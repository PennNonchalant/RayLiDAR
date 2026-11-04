"""Official 3DGS train/render orchestration for Phase 5.

This module is the only place that builds the official trainer argv, launches
the seeded launcher, and records run provenance.  It does not import torch or
the CUDA extensions at import time: the heavy imports live in the workers so
unit tests can run without a GPU.

Frozen protocol (``05-CONTEXT.md`` D-01/D-03/D-04/D-05):

* official upstream at the pinned commit, official defaults, 30,000 iterations;
* all 60 ``image_00`` frames are training views (no evaluation split);
* one fixed seed from the config, recorded in the run manifest;
* native ``1408x376`` rendering, black background;
* rendered depth is an expected z-depth ``E[z] = sum(z * alpha * T) / sum(alpha * T)``
  obtained from the official ``override_color`` pass (per-Gaussian ``z / 100``);
* accumulated alpha is the mean of the white-override render's RGB channels.

The upstream rasterizer's own ``invdepths`` is ``sum((1 / z) * alpha * T)`` (a
harmonic inverse-depth accumulator), not ``E[z]``; it is deliberately no longer
exported as the 3DGS depth map.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .. import env
from ..run_utils import (
    check_free_space,
    method_run_dir,
    read_json,
    run_logged,
    sha256_file,
    utc_now,
    write_json_atomic,
    write_run_manifest,
)

SCHEMA_VERSION = 1
METHOD = "3dgs"
CONFIG_KEY = "gs3d"
RUN_PREFIX = "3dgs"
RASTERIZER_SUBMODULE = "submodules/diff-gaussian-rasterization"
EXPECTED_RESOLUTION: tuple[int, int] = (1408, 376)
DEPTH_CONVENTION = "z_depth"
ALPHA_SOURCE = "white_override_mean_rgb"
DEPTH_SOURCE = "expected_z_color_override"
EXPECTED_Z_SCALE = 100.0
FORBIDDEN_RUN_PATH_MARKERS = (
    "artifacts/02/gt",
    "artifacts/02/rays",
    "raw/lidar",
    "derived/gt",
)
LOG_TAIL_LINES = 100


# ---------------------------------------------------------------------------
# path / provenance helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root (``reconstruct/gaussians/methods/gs3d.py``)."""
    return Path(__file__).resolve().parents[3]


def upstream_dir(cfg: dict) -> Path:
    """Return ``<paths.upstream>/<method.gs3d.upstream_subdir>``."""
    subdir = cfg["method"]["gs3d"]["upstream_subdir"]
    return Path(cfg["paths"]["upstream"]) / str(subdir)


def worker_path(name: str) -> Path:
    """Return the absolute path of a method worker script."""
    return repo_root() / "reconstruct" / "gaussians" / "methods" / name


def dataset_dir(cfg: dict) -> Path:
    """Return the frozen dataset directory for this run."""
    return Path(cfg["paths"]["artifacts_root"]) / "dataset"


def venv_python(cfg: dict) -> Path:
    """Return the pinned virtualenv interpreter."""
    return Path(cfg["paths"]["venv"]) / "bin" / "python"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(repo: Path, *arguments: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repo}",
        "-C",
        str(repo),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        raise RuntimeError(f"git {' '.join(arguments)} failed: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _submodule_head(repo: Path, relative_path: str) -> str:
    output = _git(repo, "submodule", "status", "--", relative_path)
    if not output:
        raise RuntimeError(f"{repo}: submodule {relative_path} is not initialised")
    first = output.splitlines()[0]
    tokens = first.lstrip("+- ").split()
    if not tokens:
        raise RuntimeError(f"{repo}: cannot parse submodule status for {relative_path}")
    return tokens[0]


def verify_upstream(upstream: Path) -> dict[str, str]:
    """Assert the pinned 3DGS commit and rasterizer submodule, return their heads."""
    head = _git(upstream, "rev-parse", "HEAD")
    expected = env.PINNED["gaussian_splatting"]
    if head != expected:
        raise RuntimeError(
            f"3DGS checkout {upstream} is {head}; expected pinned {expected}"
        )
    rasterizer = _submodule_head(upstream, RASTERIZER_SUBMODULE)
    rasterizer_expected = env.PINNED["diff_gaussian_rasterization"]
    if rasterizer != rasterizer_expected:
        raise RuntimeError(
            f"3DGS rasterizer is {rasterizer}; expected pinned {rasterizer_expected}"
        )
    return {
        "upstream_path": str(upstream),
        "upstream_head": head,
        "rasterizer_head": rasterizer,
    }


def _run_environment(upstream: Path) -> dict[str, str]:
    """Return the subprocess environment for a worker run in the checkout."""
    environment = dict(os.environ)
    parts = [str(upstream), str(repo_root())]
    inherited = environment.get("PYTHONPATH")
    if inherited:
        parts.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    return environment


def _gpu_name() -> str:
    try:
        import torch  # noqa: PLC0415 - lazy so unit tests need no CUDA runtime
    except Exception:
        return "unavailable"
    try:
        if not torch.cuda.is_available():
            return "unavailable"
        return str(torch.cuda.get_device_name(0))
    except Exception:  # pragma: no cover - broken GPU runtime
        return "unavailable"


def _log_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return content.splitlines()[-lines:]


def _assert_no_forbidden_paths(argv: Sequence[str]) -> None:
    rendered = "\n".join(str(argument) for argument in argv)
    for marker in FORBIDDEN_RUN_PATH_MARKERS:
        if marker in rendered:
            raise RuntimeError(
                f"training command references evaluation-only input {marker!r}"
            )


def _validate_dataset(cfg: dict) -> tuple[Path, dict]:
    root = dataset_dir(cfg)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"frozen dataset manifest missing: {manifest_path}; run --dataset first"
        )
    manifest = read_json(manifest_path)
    if manifest.get("no_gt_inputs") is not True:
        raise RuntimeError(f"{manifest_path}: no_gt_inputs is not true")
    if manifest.get("camera") != cfg["camera"]:
        raise RuntimeError(
            f"{manifest_path}: camera {manifest.get('camera')!r} != {cfg['camera']!r}"
        )
    manifest_resolution = [int(value) for value in manifest.get("resolution") or []]
    if manifest_resolution != [int(value) for value in cfg["resolution"]]:
        raise RuntimeError(
            f"{manifest_path}: resolution {manifest_resolution} != "
            f"{list(cfg['resolution'])}"
        )
    window = manifest.get("window") or {}
    if int(window.get("count", 0)) != len(cfg["frames"]):
        raise RuntimeError(
            f"{manifest_path}: window count {window.get('count')} != "
            f"{len(cfg['frames'])}"
        )
    if int(window.get("first", -1)) != min(cfg["frames"]):
        raise RuntimeError(f"{manifest_path}: unexpected window.first")
    if int(window.get("last", -1)) != max(cfg["frames"]):
        raise RuntimeError(f"{manifest_path}: unexpected window.last")
    return manifest_path, manifest


def _archive_partial(path: Path) -> str | None:
    """Move a partial output directory aside, preserving failed-attempt evidence."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return None
    if not any(path.iterdir()):
        return None
    archived = path.with_name(f"{path.name}-failed-{_utc_stamp()}")
    path.rename(archived)
    path.mkdir(parents=True, exist_ok=True)
    return str(archived)


def _require_matching_manifest(existing: dict, digest: str, smoke: bool) -> None:
    if existing.get("config_sha256") != digest:
        raise RuntimeError(
            "existing run was produced from a different config SHA-256; "
            "use a new --run-id instead of mixing runs"
        )
    if bool(existing.get("smoke")) != bool(smoke):
        raise RuntimeError(
            "existing run has a different smoke flag; use a new --run-id"
        )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def build_train_argv(
    cfg: dict, run_dir: str | Path, smoke: bool, seed: int
) -> list[str]:
    """Build the exact official-trainer argv (list, never a shell string)."""
    method = cfg["method"]["gs3d"]
    iterations = int(
        method["smoke_iterations"] if smoke else method["iterations"]
    )
    return [
        str(venv_python(cfg)),
        str(worker_path("train_gs3d_worker.py")),
        str(int(seed)),
        "-s",
        str(dataset_dir(cfg)),
        "-m",
        str(run_dir),
        "--iterations",
        str(iterations),
        "--resolution",
        str(int(method["resolution"])),
        "--save_iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
        "--checkpoint_iterations",
        str(iterations),
        "--disable_viewer",
        "--quiet",
    ]


def _train_stage_record(
    *,
    cfg: dict,
    context: dict,
    effective_run_id: str,
    run_dir: Path,
    manifest_path: Path,
    run_manifest: dict,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    iterations = int(run_manifest["iterations"])
    checkpoint = run_dir / f"chkpnt{iterations}.pth"
    point_cloud = run_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    stage = {
        "schema_version": SCHEMA_VERSION,
        "stage": "train-gs3d",
        "method": METHOD,
        "generated_at_utc": utc_now(),
        "config_sha256": context.get("config_sha256")
        or _cfg_digest(cfg),
        "run_id": effective_run_id,
        "smoke": bool(context.get("smoke")),
        "run_dir": str(run_dir),
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": run_manifest_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": run_manifest.get("checkpoint_sha256"),
        "point_cloud": str(point_cloud),
        "point_cloud_sha256": run_manifest.get("point_cloud_sha256"),
        "iterations": iterations,
        "seed": run_manifest.get("seed"),
        "gpu": run_manifest.get("gpu"),
        "duration_s": run_manifest.get("duration_s"),
        "free_space_bytes": run_manifest.get("free_space_bytes"),
    }
    write_json_atomic(
        Path(cfg["paths"]["artifacts_root"]) / "stages" / "train-gs3d.json", stage
    )
    return stage


def _cfg_digest(cfg: dict) -> str:
    from ..config import config_sha256  # noqa: PLC0415 - avoid a config import cycle

    return config_sha256(cfg)


def train(context: dict) -> dict[str, Any]:
    """Pipeline stage: train official 3DGS with one fixed seed."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("train-3dgs stage requires context['cfg']")
    smoke = bool(context.get("smoke"))
    run_id = str(context.get("run_id") or cfg["run_id"])
    digest = str(context.get("config_sha256") or _cfg_digest(cfg))
    seed = int(cfg["seed"])
    iterations = int(
        cfg["method"]["gs3d"]["smoke_iterations"]
        if smoke
        else cfg["method"]["gs3d"]["iterations"]
    )

    artifacts_root = Path(cfg["paths"]["artifacts_root"])
    artifacts_root.mkdir(parents=True, exist_ok=True)
    dataset_manifest_path, _ = _validate_dataset(cfg)
    upstream = upstream_dir(cfg)
    provenance = verify_upstream(upstream)

    env_report = artifacts_root / "env_report.json"
    if not env_report.is_file():
        raise RuntimeError(
            f"environment report missing: {env_report}; run --env before training"
        )
    env_report_sha256 = sha256_file(env_report)

    effective_run_id = f"{run_id}-smoke" if smoke else run_id
    run_dir = method_run_dir(cfg, RUN_PREFIX, effective_run_id)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        _require_matching_manifest(existing, digest, smoke)
        existing_iteration = int(existing["iterations"])
        for path in (
            run_dir / f"chkpnt{existing_iteration}.pth",
            run_dir / "point_cloud" / f"iteration_{existing_iteration}" / "point_cloud.ply",
        ):
            if not path.is_file():
                raise RuntimeError(
                    f"existing run manifest points at missing artifact: {path}"
                )
        return _train_stage_record(
            cfg=cfg,
            context=context,
            effective_run_id=effective_run_id,
            run_dir=run_dir,
            manifest_path=manifest_path,
            run_manifest=existing,
            run_manifest_sha256=sha256_file(manifest_path),
        )

    archived = _archive_partial(run_dir)
    if archived:
        print(f"archived partial run directory to {archived}", file=sys.stderr)

    free_space_before = check_free_space(artifacts_root)
    argv = build_train_argv(cfg, run_dir, smoke=smoke, seed=seed)
    _assert_no_forbidden_paths(argv)

    started_at = utc_now()
    started = time.monotonic()
    result = run_logged(
        argv,
        cwd=upstream,
        env=_run_environment(upstream),
        log_path=run_dir / "train.log",
    )
    duration_s = round(time.monotonic() - started, 3)
    finished_at = utc_now()

    if result["returncode"] != 0:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "stage": "train-3dgs",
            "run_id": effective_run_id,
            "smoke": smoke,
            "returncode": result["returncode"],
            "argv": argv,
            "cwd": str(upstream),
            "config_sha256": digest,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "duration_s": duration_s,
            "train_log": str(result["log_path"]),
            "train_log_sha256": result["log_sha256"],
            "tail": _log_tail(Path(result["log_path"])),
        }
        write_json_atomic(run_dir / "train_failure.json", failure)
        raise RuntimeError(
            f"3DGS training failed with exit {result['returncode']}; "
            f"see {run_dir / 'train_failure.json'}"
        )

    checkpoint = run_dir / f"chkpnt{iterations}.pth"
    point_cloud = run_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    missing = [str(path) for path in (checkpoint, point_cloud) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "3DGS training reported success but did not write: " + ", ".join(missing)
        )

    checkpoint_sha256 = sha256_file(checkpoint)
    point_cloud_sha256 = sha256_file(point_cloud)
    event_files = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.glob("events.out.tfevents.*"))
    ]
    free_space_after = int(shutil.disk_usage(str(artifacts_root)).free)

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "run_id": effective_run_id,
        "smoke": smoke,
        "upstream_path": provenance["upstream_path"],
        "upstream_head": provenance["upstream_head"],
        "rasterizer_head": provenance["rasterizer_head"],
        "config_sha256": digest,
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "env_report_sha256": env_report_sha256,
        "dataset_dir": str(dataset_dir(cfg)),
        "seed": seed,
        "iterations": iterations,
        "resolution": [int(value) for value in cfg["resolution"]],
        "command": argv,
        "cwd": str(upstream),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_s": duration_s,
        "gpu": _gpu_name(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "point_cloud": str(point_cloud),
        "point_cloud_sha256": point_cloud_sha256,
        "train_log": str(result["log_path"]),
        "train_log_sha256": result["log_sha256"],
        "event_files": event_files,
        "free_space_bytes_before": free_space_before,
        "free_space_bytes": free_space_after,
        "training_view": True,
        "nvs": False,
        "no_gt_inputs": True,
    }
    run_manifest_sha256 = write_run_manifest(manifest_path, run_manifest)

    return _train_stage_record(
        cfg=cfg,
        context=context,
        effective_run_id=effective_run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        run_manifest=run_manifest,
        run_manifest_sha256=run_manifest_sha256,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def invdepth_to_z(inv: Any) -> np.ndarray:
    """Convert the rasterizer's inverse-depth accumulator with ``z = 1 / inv``.

    Legacy helper kept for backward compatibility.  The upstream rasterizer
    accumulates ``sum((1 / z) * alpha * T)`` without normalising by
    ``sum(alpha * T)``, so this yields a harmonic inverse-depth mean rather
    than ``E[z]``; the render worker no longer uses it for the 3DGS depth map.
    Every non-positive/non-finite value becomes ``0``.
    """
    values = np.asarray(inv, dtype=np.float32)
    positive = np.isfinite(values) & (values > 0.0)
    result = np.zeros_like(values, dtype=np.float32)
    np.divide(1.0, values, out=result, where=positive)
    return result


def view_space_z(xyz_h: Any, world_view_transform: Any) -> Any:
    """Return each Gaussian's view-space z from a homogeneous world point.

    ``scene/cameras.py`` stores ``world_view_transform`` as the *transpose* of
    the world->camera matrix, and the CUDA rasterizer applies ``W.T @ p``.  With
    row-vector points that is ``p_hom @ W``; using ``W @ p_hom`` instead would
    rotate the translation into the wrong slot and is the bug this helper
    prevents.
    """
    return (xyz_h @ world_view_transform)[:, 2]


def expected_z_from_render(
    z_render: Any, alpha: Any, scale: float = EXPECTED_Z_SCALE
) -> np.ndarray:
    """Return ``E[z] = scale * sum((z / scale) * alpha * T) / sum(alpha * T)``.

    ``z_render`` is the official ``override_color`` pass rendered with
    ``z_color = z_view / scale`` and a black background, so every channel is
    ``sum((z / scale) * alpha * T)``.  Dividing by the accumulated alpha from
    the white-override pass and multiplying by ``scale`` recovers the expected
    z-depth.  Pixels with ``alpha <= 0`` (or non-finite inputs) become ``0``.
    """
    values = np.asarray(z_render, dtype=np.float32)
    weights = np.asarray(alpha, dtype=np.float32)
    if values.shape != weights.shape:
        raise ValueError(
            "z_render and alpha must have the same shape, got "
            f"{tuple(values.shape)} and {tuple(weights.shape)}"
        )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    positive = np.isfinite(weights) & (weights > 0.0) & np.isfinite(values)
    result = np.zeros_like(values, dtype=np.float32)
    np.divide(values * float(scale), weights, out=result, where=positive)
    result[~np.isfinite(result)] = 0.0
    return result


def alpha_from_white_render(rgb: Any) -> np.ndarray:
    """Return accumulated alpha from a ``[3, H, W]`` white-override render."""
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(
            f"expected a [3, H, W] render, got shape {tuple(values.shape)}"
        )
    return np.clip(values.mean(axis=0), 0.0, 1.0).astype(np.float32)


def render_argv(
    cfg: dict,
    run_dir: Path,
    iteration: int,
    out_dir: Path,
    frames: Iterable[int],
) -> list[str]:
    """Build the render-worker argv."""
    return [
        str(venv_python(cfg)),
        str(worker_path("render_gs3d_worker.py")),
        "-s",
        str(dataset_dir(cfg)),
        "-m",
        str(run_dir),
        "--iteration",
        str(int(iteration)),
        "--out",
        str(out_dir),
        "--frames",
        ",".join(str(int(frame)) for frame in frames),
    ]


def verify_render_output(
    render_dir: str | Path,
    frames: Iterable[int],
    *,
    iteration: int | None = None,
    point_cloud_sha256: str | None = None,
) -> dict:
    """Verify every rendered frame, hash, dtype, and native resolution."""
    root = Path(render_dir)
    manifest_path = root / "render_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"render manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)

    checks = {
        "schema_version": 1,
        "method": "3dgs",
        "depth_convention": DEPTH_CONVENTION,
        "depth_source": DEPTH_SOURCE,
        "depth_scale": EXPECTED_Z_SCALE,
        "alpha_source": ALPHA_SOURCE,
        "separate_sh": False,
        "background": [0, 0, 0],
        "resolution": [EXPECTED_RESOLUTION[0], EXPECTED_RESOLUTION[1]],
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"{manifest_path}: {key}={manifest.get(key)!r} != {expected!r}"
            )
    if iteration is not None and int(manifest.get("iteration", -1)) != int(iteration):
        raise RuntimeError(
            f"{manifest_path}: iteration {manifest.get('iteration')} != {iteration}"
        )
    if (
        point_cloud_sha256 is not None
        and manifest.get("point_cloud_sha256") != point_cloud_sha256
    ):
        raise RuntimeError(f"{manifest_path}: point-cloud hash mismatch")

    expected_frames = [int(frame) for frame in frames]
    per_frame = manifest.get("per_frame")
    if not isinstance(per_frame, list):
        raise RuntimeError(f"{manifest_path}: per_frame must be a list")
    recorded = [int(entry.get("frame", -1)) for entry in per_frame]
    if recorded != expected_frames:
        raise RuntimeError(
            f"{manifest_path}: rendered frames {recorded} != expected {expected_frames}"
        )

    width, height = EXPECTED_RESOLUTION
    for entry in per_frame:
        frame = int(entry["frame"])
        paths = {
            "rgb": root / f"{frame:010d}_rgb.png",
            "depth": root / f"{frame:010d}_depth.npy",
            "alpha": root / f"{frame:010d}_alpha.npy",
        }
        for kind, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"frame {frame}: missing {kind} file {path}")
            if sha256_file(path) != entry.get(f"{kind}_sha256"):
                raise RuntimeError(f"frame {frame}: {kind} hash mismatch for {path}")

        with Image.open(paths["rgb"]) as image:
            if image.size != (width, height):
                raise RuntimeError(
                    f"frame {frame}: RGB size {image.size} != {(width, height)}"
                )
            if image.mode != "RGB":
                raise RuntimeError(f"frame {frame}: RGB mode {image.mode} != RGB")

        depth = np.load(paths["depth"])
        alpha = np.load(paths["alpha"])
        for kind, array in (("depth", depth), ("alpha", alpha)):
            if array.dtype != np.float32:
                raise RuntimeError(f"frame {frame}: {kind} dtype {array.dtype}")
            if array.shape != (height, width):
                raise RuntimeError(
                    f"frame {frame}: {kind} shape {array.shape} != "
                    f"{(height, width)}"
                )
            if not np.isfinite(array).all():
                raise RuntimeError(f"frame {frame}: {kind} contains NaN/inf")
        if float(alpha.min()) < 0.0 or float(alpha.max()) > 1.0:
            raise RuntimeError(f"frame {frame}: alpha outside [0, 1]")

    return manifest


def _render_stage_record(
    *,
    cfg: dict,
    context: dict,
    effective_run_id: str,
    out_dir: Path,
    manifest_path: Path,
    render_manifest: dict,
    render_manifest_sha256: str,
) -> dict[str, Any]:
    stage = {
        "schema_version": SCHEMA_VERSION,
        "stage": "render-gs3d",
        "method": METHOD,
        "generated_at_utc": utc_now(),
        "config_sha256": context.get("config_sha256") or _cfg_digest(cfg),
        "run_id": effective_run_id,
        "smoke": bool(context.get("smoke")),
        "render_dir": str(out_dir),
        "render_manifest": str(manifest_path),
        "render_manifest_sha256": render_manifest_sha256,
        "iteration": int(render_manifest.get("iteration", -1)),
        "frames": len(render_manifest.get("per_frame") or []),
        "resolution": render_manifest.get("resolution"),
        "point_cloud_sha256": render_manifest.get("point_cloud_sha256"),
    }
    write_json_atomic(
        Path(cfg["paths"]["artifacts_root"]) / "stages" / "render-gs3d.json", stage
    )
    return stage


def render(context: dict) -> dict[str, Any]:
    """Pipeline stage: render RGB, alpha, and z-depth for every requested frame."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("render-3dgs stage requires context['cfg']")
    smoke = bool(context.get("smoke"))
    run_id = str(context.get("run_id") or cfg["run_id"])
    digest = str(context.get("config_sha256") or _cfg_digest(cfg))

    artifacts_root = Path(cfg["paths"]["artifacts_root"])
    artifacts_root.mkdir(parents=True, exist_ok=True)
    effective_run_id = f"{run_id}-smoke" if smoke else run_id
    run_dir = method_run_dir(cfg, RUN_PREFIX, effective_run_id)
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise RuntimeError(
            f"run manifest missing: {run_manifest_path}; run --train-3dgs first"
        )
    run_manifest = read_json(run_manifest_path)
    _require_matching_manifest(run_manifest, digest, smoke)
    iteration = int(run_manifest["iterations"])
    point_cloud_sha256 = str(run_manifest["point_cloud_sha256"])

    frames = (
        [int(frame) for frame in cfg["method"]["gs3d"]["smoke_frames"]]
        if smoke
        else [int(frame) for frame in cfg["frames"]]
    )
    out_dir = artifacts_root / "renders" / f"{RUN_PREFIX}-{effective_run_id}"
    manifest_path = out_dir / "render_manifest.json"
    if manifest_path.is_file():
        existing = verify_render_output(
            out_dir,
            frames,
            iteration=iteration,
            point_cloud_sha256=point_cloud_sha256,
        )
        return _render_stage_record(
            cfg=cfg,
            context=context,
            effective_run_id=effective_run_id,
            out_dir=out_dir,
            manifest_path=manifest_path,
            render_manifest=existing,
            render_manifest_sha256=sha256_file(manifest_path),
        )

    archived = _archive_partial(out_dir)
    if archived:
        print(f"archived partial render directory to {archived}", file=sys.stderr)

    check_free_space(artifacts_root)
    upstream = upstream_dir(cfg)
    provenance = verify_upstream(upstream)
    argv = render_argv(cfg, run_dir, iteration, out_dir, frames)
    _assert_no_forbidden_paths(argv)

    result = run_logged(
        argv,
        cwd=upstream,
        env=_run_environment(upstream),
        log_path=out_dir / "render.log",
    )
    if result["returncode"] != 0:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "stage": "render-3dgs",
            "run_id": effective_run_id,
            "smoke": smoke,
            "returncode": result["returncode"],
            "argv": argv,
            "cwd": str(upstream),
            "config_sha256": digest,
            "render_log": str(result["log_path"]),
            "render_log_sha256": result["log_sha256"],
            "tail": _log_tail(Path(result["log_path"])),
        }
        write_json_atomic(out_dir / "render_failure.json", failure)
        raise RuntimeError(
            f"3DGS rendering failed with exit {result['returncode']}; "
            f"see {out_dir / 'render_failure.json'}"
        )

    render_manifest = verify_render_output(
        out_dir,
        frames,
        iteration=iteration,
        point_cloud_sha256=point_cloud_sha256,
    )
    if render_manifest.get("upstream_head") != provenance["upstream_head"]:
        raise RuntimeError(f"{manifest_path}: upstream head mismatch")

    return _render_stage_record(
        cfg=cfg,
        context=context,
        effective_run_id=effective_run_id,
        out_dir=out_dir,
        manifest_path=manifest_path,
        render_manifest=render_manifest,
        render_manifest_sha256=sha256_file(manifest_path),
    )


__all__ = [
    "ALPHA_SOURCE",
    "DEPTH_CONVENTION",
    "DEPTH_SOURCE",
    "EXPECTED_RESOLUTION",
    "EXPECTED_Z_SCALE",
    "CONFIG_KEY",
    "METHOD",
    "RUN_PREFIX",
    "alpha_from_white_render",
    "build_train_argv",
    "dataset_dir",
    "expected_z_from_render",
    "invdepth_to_z",
    "render",
    "render_argv",
    "repo_root",
    "train",
    "upstream_dir",
    "venv_python",
    "verify_render_output",
    "verify_upstream",
    "view_space_z",
    "worker_path",
]
