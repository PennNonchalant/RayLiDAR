"""Run the frozen Phase-3 metrics for both Gaussian methods.

Every evaluation goes through exactly the frozen CLI, without reimplementing a
metric formula:

    python -m reconstruct.eval real --gt-rays <bin> --hits <bin> --out-dir <dir> ...
    python -m reconstruct.eval validate --metrics <json> --csv <csv>

The runner builds ``provenance.json`` from the run/render/hit manifests, invokes
the CLI for the ``all`` and ``unoccluded`` modes, and then re-checks the emitted
metrics: mode label, ray/hit SHA-256, the frozen camera-0 coverage denominator,
and the 60-frame x 2-camera x 4-bucket row count.  Misses stay ``+inf``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import probe
from .run_utils import read_json, run_logged, sha256_file, utc_now, write_json_atomic

SCHEMA_VERSION = 1
METHODS = ("3dgs", "2dgs")
MODES = ("all", "unoccluded")
MODE_LABELS = {
    "all": "gaussian_depth_all",
    "unoccluded": "gaussian_depth_unoccluded",
}
COVERAGE_POLICY = "frozen_camera0_valid_population_per_frame_and_bucket"
PRIMARY_CAMERA = 0
ROWS_PER_FRAME = 2 * 4  # two cameras x four buckets
EXPECTED_PIXEL_ROUNDING = "half_up"
VALIDATION_DIRNAME = "phase3_validation"


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root (``reconstruct/gaussians/eval_runner.py``)."""
    return Path(__file__).resolve().parents[2]


def _eval_argv() -> list[str]:
    return [sys.executable, "-m", "reconstruct.eval"]


def protocol_config_path() -> Path:
    """Return the frozen evaluation config."""
    return repo_root() / "reconstruct" / "configs" / "eval.yaml"


def protocol_doc_path() -> Path:
    """Return the frozen protocol document."""
    return repo_root() / "reconstruct" / "configs" / "eval_protocol.md"


def run_config_path(cfg: dict) -> Path:
    """Return the gaussians run config recorded in ``metrics.json``."""
    source = cfg.get("_source")
    if source:
        return Path(str(source))
    return repo_root() / "reconstruct" / "configs" / "gaussians_1820.yaml"


def artifacts_root(cfg: dict) -> Path:
    """Return the Phase-5 artifact root."""
    return Path(cfg["paths"]["artifacts_root"])


def hits_dir(cfg: dict, method: str) -> Path:
    """Return ``<artifacts_root>/hits/<method>``."""
    return artifacts_root(cfg) / "hits" / method


def eval_inputs(cfg: dict, method: str, mode: str) -> tuple[Path, Path]:
    """Return the ``(gt_rays, hits)`` pair for one method/mode."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    root = hits_dir(cfg, method)
    if mode == "all":
        return probe.gt_rays_path(cfg), root / "hits.bin"
    return root / "gt_rays_unoccluded.bin", root / "hits_unoccluded.bin"


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{what} missing: {path}")
    return path


def build_provenance(cfg: dict, method: str, mode: str) -> dict:
    """Build the provenance object handed to ``eval real --provenance``."""
    artifacts = artifacts_root(cfg)
    run_manifest_path = _require_file(
        artifacts / "runs" / f"{method}-{cfg['run_id']}" / "run_manifest.json",
        "run manifest",
    )
    run_manifest = read_json(run_manifest_path)
    if bool(run_manifest.get("smoke")):
        raise RuntimeError(f"{run_manifest_path}: smoke runs cannot be evaluated")
    if run_manifest.get("method") != method:
        raise RuntimeError(
            f"{run_manifest_path}: method {run_manifest.get('method')!r} != "
            f"{method!r}"
        )

    hits_manifest_path = _require_file(
        hits_dir(cfg, method) / "hits_manifest.json", "hits manifest"
    )
    hits_manifest = read_json(hits_manifest_path)
    render_dir = Path(str(hits_manifest.get("render_dir") or ""))
    if not str(render_dir):
        raise RuntimeError(f"{hits_manifest_path}: render_dir is missing")
    render_manifest_path = _require_file(
        render_dir / "render_manifest.json", "render manifest"
    )
    render_manifest = read_json(render_manifest_path)

    convention_path = _require_file(
        probe.convention_path(cfg), "depth convention"
    )
    convention = read_json(convention_path)
    decision = convention.get("decision")
    if decision not in ("z_depth", "range"):
        raise RuntimeError(
            f"{convention_path}: decision {decision!r} is not frozen"
        )

    gt_rays, hits = eval_inputs(cfg, method, mode)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "mode": mode,
        "upstream_head": run_manifest.get("upstream_head"),
        "rasterizer_head": run_manifest.get("rasterizer_head"),
        "checkpoint_sha256": run_manifest.get("checkpoint_sha256"),
        "point_cloud_sha256": run_manifest.get("point_cloud_sha256"),
        "render_manifest_sha256": sha256_file(render_manifest_path),
        "hits_manifest_sha256": sha256_file(hits_manifest_path),
        "depth_convention": decision,
        "alpha_threshold": float(cfg["protocol"]["alpha_threshold"]),
        "pixel_rounding": EXPECTED_PIXEL_ROUNDING,
        "resolution": [int(value) for value in cfg["resolution"]],
        "frames": [int(frame) for frame in cfg["frames"]],
        "depth_source": render_manifest.get("depth_source"),
        "alpha_source": render_manifest.get("alpha_source"),
        "coverage_policy": COVERAGE_POLICY,
        "seed": int(cfg["seed"]),
        "ray_set": str(gt_rays),
        "hit_set": str(hits),
        "generated_at_utc": utc_now(),
    }


# ---------------------------------------------------------------------------
# post-checks
# ---------------------------------------------------------------------------


def _point_cloud_path(run_manifest: dict) -> Path:
    value = run_manifest.get("point_cloud")
    if not isinstance(value, str) or not value:
        raise RuntimeError("run manifest does not record a point-cloud path")
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"point cloud missing: {path}")
    return path


def _link(target: Path, link: Path) -> None:
    """Point ``link`` at ``target`` (idempotent, symlink so nothing is copied)."""
    link.unlink(missing_ok=True)
    link.symlink_to(Path(target).resolve())


def prepare_validation_bundle(
    out_dir: Path,
    gt_rays: Path,
    hits: Path,
    run_manifest: dict,
) -> Path:
    """Build the sidecars ``eval validate`` expects next to a real-mode run.

    ``python -m reconstruct.eval real`` writes ``metrics.json``/``metrics.csv`` and
    validates them with the Phase-3 validators, but the standalone
    ``eval validate`` path also re-hashes a placeholder-style
    ``sim_io_manifest.json`` + ``scene_manifest.json``/``scene.ply``.  Real mode
    never emitted those, so this bundle supplies them with the *real* hashes
    (symlinks, no data copies) and points ``--metrics``/``--csv`` at the
    byte-identical metrics files.  Nothing is fabricated: every hash is the
    hash of the frozen input or trained point cloud actually used.
    """
    bundle = out_dir / VALIDATION_DIRNAME
    inputs = bundle / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    _link(gt_rays, inputs / "gt_rays.bin")
    _link(hits, inputs / "hits.bin")
    point_cloud = _point_cloud_path(run_manifest)
    _link(point_cloud, bundle / "scene.ply")
    _link(out_dir / "metrics.json", bundle / "metrics.json")
    _link(out_dir / "metrics.csv", bundle / "metrics.csv")
    write_json_atomic(
        bundle / "sim_io_manifest.json",
        {
            "schema_version": 1,
            "entries": [
                {
                    "kind": "gt_rays",
                    "path": "inputs/gt_rays.bin",
                    "sha256": sha256_file(gt_rays),
                },
                {
                    "kind": "hits",
                    "path": "inputs/hits.bin",
                    "sha256": sha256_file(hits),
                },
            ],
        },
    )
    write_json_atomic(
        bundle / "scene_manifest.json",
        {"schema_version": 1, "sha256": sha256_file(point_cloud)},
    )
    return bundle


def _count_csv_rows(path: Path) -> int:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"{path}: metrics CSV is empty")
    return len(lines) - 1


def camera0_coverage(metrics: dict) -> float:
    """Return camera-0 coverage over every row, secondary bucket included.

    The frozen protocol defines coverage as ``N_valid / N_raw`` with every raw
    cast ray in the denominator.  The Phase-3 ``aggregates.per_camera`` cell is
    the *primary* accuracy cell, whose numerator excludes the ``(50, 60]``
    secondary bucket while the denominator still counts it (the frozen formula);
    summing the per-frame/per-camera/per-bucket rows restores the full
    population, which must be 100% valid for the frozen camera-0 ray set.
    """
    n_raw = 0
    n_valid = 0
    for row in metrics.get("rows") or []:
        if int(row.get("camera", -1)) != PRIMARY_CAMERA:
            continue
        n_raw += int(row.get("n_raw", 0))
        n_valid += int(row.get("n_valid", 0))
    if n_raw == 0:
        raise RuntimeError("metrics.json has no camera-0 rows")
    coverage = n_valid / n_raw
    if coverage > 1.0 or coverage < 0.0:
        raise RuntimeError(f"camera-0 coverage {coverage!r} is outside [0, 1]")
    return coverage


def camera0_primary_coverage(metrics: dict) -> float | None:
    """Return the frozen Phase-3 primary per-camera coverage (secondary excluded)."""
    aggregates = metrics.get("aggregates") or {}
    for entry in aggregates.get("per_camera") or []:
        if int(entry.get("camera", -1)) != PRIMARY_CAMERA:
            continue
        cell = entry.get("cell") or {}
        coverage = cell.get("coverage")
        return None if coverage is None else float(coverage)
    return None


def _post_check(
    cfg: dict,
    method: str,
    mode: str,
    label: str,
    metrics: dict,
    gt_rays: Path,
    hits: Path,
    csv_path: Path,
) -> tuple[float, float, int]:
    if metrics.get("mode") != label:
        raise RuntimeError(
            f"metrics.json mode {metrics.get('mode')!r} != {label!r}"
        )
    inputs = metrics.get("inputs") or {}
    if inputs.get("mode") != label:
        raise RuntimeError(f"metrics.json inputs.mode != {label!r}")
    if inputs.get("ray_set_sha256") != sha256_file(gt_rays):
        raise RuntimeError("metrics.json ray_set_sha256 does not match gt_rays")
    if inputs.get("hit_set_sha256") != sha256_file(hits):
        raise RuntimeError("metrics.json hit_set_sha256 does not match hits")
    if inputs.get("scene_generator") != f"gaussian_{method}":
        raise RuntimeError(
            f"metrics.json scene_generator {inputs.get('scene_generator')!r} "
            f"!= 'gaussian_{method}'"
        )
    coverage = camera0_coverage(metrics)
    if abs(coverage - 1.0) > 1e-12:
        raise RuntimeError(
            f"camera-0 coverage {coverage!r} != 1.0 under the frozen "
            f"population denominator"
        )
    primary_coverage = camera0_primary_coverage(metrics)
    if primary_coverage is None:
        raise RuntimeError("metrics.json has no Phase-3 primary camera-0 coverage")
    expected_rows = len(cfg["frames"]) * ROWS_PER_FRAME
    rows = metrics.get("rows") or []
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"metrics.json rows {len(rows)} != {expected_rows}"
        )
    csv_rows = _count_csv_rows(csv_path)
    if csv_rows != expected_rows:
        raise RuntimeError(f"metrics CSV rows {csv_rows} != {expected_rows}")
    return coverage, primary_coverage, expected_rows


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def run_method(cfg: dict, method: str, mode: str, cli: str | Path | None = None) -> dict:
    """Evaluate one method in one mode through the frozen Phase-3 CLI."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    gt_rays, hits = eval_inputs(cfg, method, mode)
    _require_file(gt_rays, "gt rays")
    if not hits.is_file():
        raise RuntimeError(f"hits missing: {hits}; run --hits first")

    provenance = build_provenance(cfg, method, mode)
    out_dir = artifacts_root(cfg) / "metrics" / method / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "provenance.json"
    write_json_atomic(provenance_path, provenance)

    executable = _eval_argv()
    label = MODE_LABELS[mode]
    real_argv = executable + [
        "real",
        "--gt-rays",
        str(gt_rays),
        "--hits",
        str(hits),
        "--out-dir",
        str(out_dir),
        "--mode",
        label,
        "--run-config",
        str(run_config_path(cfg)),
        "--protocol-config",
        str(protocol_config_path()),
        "--protocol-doc",
        str(protocol_doc_path()),
        "--model-sha256",
        str(provenance["point_cloud_sha256"]),
        "--scene-generator",
        f"gaussian_{method}",
        "--seed",
        str(int(cfg["seed"])),
        "--provenance",
        str(provenance_path),
    ]
    real = run_logged(
        real_argv,
        cwd=repo_root(),
        env=None,
        log_path=out_dir / "eval.log",
    )
    if real["returncode"] != 0:
        raise RuntimeError(
            f"eval real failed with exit {real['returncode']}; "
            f"see {out_dir / 'eval.log'}"
        )

    metrics_path = out_dir / "metrics.json"
    csv_path = out_dir / "metrics.csv"
    run_manifest = read_json(
        artifacts_root(cfg) / "runs" / f"{method}-{cfg['run_id']}" / "run_manifest.json"
    )
    bundle = prepare_validation_bundle(out_dir, gt_rays, hits, run_manifest)
    validate_argv = executable + [
        "validate",
        "--metrics",
        str(bundle / "metrics.json"),
        "--csv",
        str(bundle / "metrics.csv"),
    ]
    validate = run_logged(
        validate_argv,
        cwd=repo_root(),
        env=None,
        log_path=out_dir / "validate.log",
    )
    if validate["returncode"] != 0:
        raise RuntimeError(
            f"eval validate failed with exit {validate['returncode']}; "
            f"see {out_dir / 'validate.log'}"
        )

    _require_file(metrics_path, "metrics.json")
    _require_file(csv_path, "metrics.csv")
    metrics = read_json(metrics_path)
    coverage, primary_coverage, expected_rows = _post_check(
        cfg, method, mode, label, metrics, gt_rays, hits, csv_path
    )

    aggregates = metrics.get("aggregates") or {}
    real_eval_manifest = out_dir / "real_eval_manifest.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "eval",
        "method": method,
        "mode": mode,
        "mode_label": label,
        "generated_at_utc": utc_now(),
        "out_dir": str(out_dir),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "validation_bundle": str(bundle),
        "real_eval_manifest": str(real_eval_manifest),
        "real_eval_manifest_sha256": (
            sha256_file(real_eval_manifest)
            if real_eval_manifest.is_file()
            else None
        ),
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "eval_log_sha256": real["log_sha256"],
        "validate_log_sha256": validate["log_sha256"],
        "ray_set_sha256": sha256_file(gt_rays),
        "hit_set_sha256": sha256_file(hits),
        "coverage_camera0": coverage,
        "coverage_camera0_primary": primary_coverage,
        "coverage_policy": COVERAGE_POLICY,
        "rows": expected_rows,
        "micro_primary": aggregates.get("micro_primary"),
        "micro_secondary": aggregates.get("micro_secondary"),
    }
    write_json_atomic(out_dir / "eval_result.json", result)
    return result


def run(context: dict) -> dict:
    """Pipeline stage: evaluate both methods in both frozen modes."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("eval stage requires context['cfg']")

    results = {
        method: {
            mode: run_method(cfg, method, mode)
            for mode in cfg["eval"]["modes"]
        }
        for method in METHODS
    }

    convention_path = _require_file(probe.convention_path(cfg), "depth convention")
    convention = read_json(convention_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "eval",
        "generated_at_utc": utc_now(),
        "config_sha256": context.get("config_sha256"),
        "depth_convention": str(convention_path),
        "depth_convention_sha256": sha256_file(convention_path),
        "decision": convention.get("decision"),
        "protocol_config": str(protocol_config_path()),
        "protocol_config_sha256": sha256_file(protocol_config_path()),
        "protocol_doc": str(protocol_doc_path()),
        "protocol_doc_sha256": sha256_file(protocol_doc_path()),
        "coverage_policy": COVERAGE_POLICY,
        "expected_csv_rows": len(cfg["frames"]) * ROWS_PER_FRAME,
        "methods": results,
    }
    write_json_atomic(artifacts_root(cfg) / "metrics" / "eval_summary.json", summary)
    return summary


__all__ = [
    "COVERAGE_POLICY",
    "METHODS",
    "MODES",
    "MODE_LABELS",
    "PRIMARY_CAMERA",
    "ROWS_PER_FRAME",
    "artifacts_root",
    "build_provenance",
    "camera0_coverage",
    "camera0_primary_coverage",
    "eval_inputs",
    "hits_dir",
    "protocol_config_path",
    "protocol_doc_path",
    "repo_root",
    "run",
    "run_config_path",
    "run_method",
]
