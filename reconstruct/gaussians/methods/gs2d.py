"""Official 2DGS train/render orchestration for Phase 5.

This module mirrors :mod:`reconstruct.gaussians.methods.gs3d`: it builds the official
trainer argv, launches the seeded headless worker, and records run provenance.
The heavy imports (torch, the CUDA extensions) live in the workers so unit
tests run without a GPU.

Frozen protocol (``05-CONTEXT.md`` D-01/D-03/D-04/D-05):

* official upstream at the pinned commit, official defaults, 30,000 iterations;
* all 60 ``image_00`` frames are training views (no evaluation split);
* one fixed seed from the config, recorded in the run manifest;
* native ``1408x376`` rendering, black background;
* ``depth_ratio = 0.0`` so the official ``surf_depth`` is expected z-depth;
* ``rend_alpha`` is the accumulated opacity (``1 - T_final``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
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
from .gs3d import (
    FORBIDDEN_RUN_PATH_MARKERS,
    _archive_partial,
    _assert_no_forbidden_paths,
    _cfg_digest,
    _gpu_name,
    _log_tail,
    _require_matching_manifest,
    _run_environment,
    _submodule_head,
    _git,
    _validate_dataset,
    dataset_dir,
    repo_root,
    venv_python,
    worker_path,
)

SCHEMA_VERSION = 1
METHOD = "2dgs"
CONFIG_KEY = "gs2d"
RUN_PREFIX = "2dgs"
RASTERIZER_SUBMODULE = "submodules/diff-surfel-rasterization"
EXPECTED_RESOLUTION: tuple[int, int] = (1408, 376)
DEPTH_CONVENTION = "z_depth"
ALPHA_SOURCE = "rend_alpha"
DEPTH_SOURCE = "surf_depth"
DEPTH_RATIO = 0.0
NETWORK_GUI = "disabled"
LOG_TAIL_LINES = 100


# ---------------------------------------------------------------------------
# path / provenance helpers
# ---------------------------------------------------------------------------


def upstream_dir(cfg: dict) -> Path:
    """Return ``<paths.upstream>/<method.gs2d.upstream_subdir>``."""
    subdir = cfg["method"]["gs2d"]["upstream_subdir"]
    return Path(cfg["paths"]["upstream"]) / str(subdir)


def verify_upstream(upstream: Path) -> dict[str, str]:
    """Assert the pinned 2DGS commit and surfel submodule, return their heads."""
    head = _git(upstream, "rev-parse", "HEAD")
    expected = env.PINNED["2d_gaussian_splatting"]
    if head != expected:
        raise RuntimeError(
            f"2DGS checkout {upstream} is {head}; expected pinned {expected}"
        )
    submodel = _submodule_head(upstream, RASTERIZER_SUBMODULE)
    submodel_expected = env.PINNED["diff_surfel_rasterization"]
    if submodel != submodel_expected:
        raise RuntimeError(
            f"2DGS surfel rasterizer is {submodel}; expected pinned {submodel_expected}"
        )
    return {
        "upstream_path": str(upstream),
        "upstream_head": head,
        "submodel_head": submodel,
        "rasterizer_head": submodel,
    }


def surf_depth_is_z(depth_ratio: float) -> bool:
    """The official ``surf_depth`` is z-depth only for ``depth_ratio == 0.0``."""
    return float(depth_ratio) == DEPTH_RATIO


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def build_train_argv(
    cfg: dict, run_dir: str | Path, smoke: bool, seed: int
) -> list[str]:
    """Build the exact official-2DGS-trainer argv (list, never a shell string)."""
    method = cfg["method"]["gs2d"]
    iterations = int(method["smoke_iterations"] if smoke else method["iterations"])
    return [
        str(venv_python(cfg)),
        str(worker_path("train_gs2d_worker.py")),
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
        "--depth_ratio",
        str(DEPTH_RATIO),
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
        "stage": "train-gs2d",
        "method": METHOD,
        "generated_at_utc": utc_now(),
        "config_sha256": context.get("config_sha256") or _cfg_digest(cfg),
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
        "network_gui": NETWORK_GUI,
        "depth_ratio": DEPTH_RATIO,
    }
    write_json_atomic(
        Path(cfg["paths"]["artifacts_root"]) / "stages" / "train-gs2d.json", stage
    )
    return stage


def train(context: dict) -> dict[str, Any]:
    """Pipeline stage: train official 2DGS with one fixed seed and no GUI."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("train-2dgs stage requires context['cfg']")
    smoke = bool(context.get("smoke"))
    run_id = str(context.get("run_id") or cfg["run_id"])
    digest = str(context.get("config_sha256") or _cfg_digest(cfg))
    seed = int(cfg["seed"])
    iterations = int(
        cfg["method"]["gs2d"]["smoke_iterations"]
        if smoke
        else cfg["method"]["gs2d"]["iterations"]
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
            run_dir
            / "point_cloud"
            / f"iteration_{existing_iteration}"
            / "point_cloud.ply",
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
            "stage": "train-2dgs",
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
            f"2DGS training failed with exit {result['returncode']}; "
            f"see {run_dir / 'train_failure.json'}"
        )

    checkpoint = run_dir / f"chkpnt{iterations}.pth"
    point_cloud = run_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    missing = [str(path) for path in (checkpoint, point_cloud) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "2DGS training reported success but did not write: " + ", ".join(missing)
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
        "submodel_head": provenance["submodel_head"],
        "config_sha256": digest,
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "env_report_sha256": env_report_sha256,
        "dataset_dir": str(dataset_dir(cfg)),
        "seed": seed,
        "iterations": iterations,
        "resolution": [int(value) for value in cfg["resolution"]],
        "depth_ratio": DEPTH_RATIO,
        "network_gui": NETWORK_GUI,
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
        str(worker_path("render_gs2d_worker.py")),
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


def _expected_render_manifest_fields() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": METHOD,
        "depth_convention": DEPTH_CONVENTION,
        "depth_source": DEPTH_SOURCE,
        "alpha_source": ALPHA_SOURCE,
        "depth_ratio": DEPTH_RATIO,
        "background": [0, 0, 0],
        "resolution": [EXPECTED_RESOLUTION[0], EXPECTED_RESOLUTION[1]],
    }


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

    for key, expected in _expected_render_manifest_fields().items():
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
        "stage": "render-gs2d",
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
        "alpha_source": ALPHA_SOURCE,
        "depth_source": DEPTH_SOURCE,
        "depth_ratio": DEPTH_RATIO,
    }
    write_json_atomic(
        Path(cfg["paths"]["artifacts_root"]) / "stages" / "render-gs2d.json", stage
    )
    return stage


def render(context: dict) -> dict[str, Any]:
    """Pipeline stage: render RGB, ``rend_alpha``, and ``surf_depth`` for every frame."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("render-2dgs stage requires context['cfg']")
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
            f"run manifest missing: {run_manifest_path}; run --train-2dgs first"
        )
    run_manifest = read_json(run_manifest_path)
    _require_matching_manifest(run_manifest, digest, smoke)
    iteration = int(run_manifest["iterations"])
    point_cloud_sha256 = str(run_manifest["point_cloud_sha256"])

    frames = (
        [int(frame) for frame in cfg["method"]["gs2d"]["smoke_frames"]]
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
            "stage": "render-2dgs",
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
            f"2DGS rendering failed with exit {result['returncode']}; "
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
    "CONFIG_KEY",
    "DEPTH_CONVENTION",
    "DEPTH_RATIO",
    "DEPTH_SOURCE",
    "EXPECTED_RESOLUTION",
    "METHOD",
    "NETWORK_GUI",
    "RUN_PREFIX",
    "build_train_argv",
    "dataset_dir",
    "render",
    "render_argv",
    "repo_root",
    "surf_depth_is_z",
    "train",
    "upstream_dir",
    "venv_python",
    "verify_render_output",
    "verify_upstream",
    "worker_path",
]
