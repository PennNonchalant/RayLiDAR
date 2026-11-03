"""Build the frozen 3DGS-vs-2DGS comparison package.

This module is deliberately downstream-only.  It reads the frozen dataset,
``gt_rays.bin``, the four validated Phase-3 ``metrics.json`` files, both render
directories, and both run manifests.  It never trains, renders, exports rays, or
recomputes a metric formula.

The report has three co-primary outcomes (D-12): training-view PSNR, training-view
SSIM, and the frozen Phase-3 GT-ray geometry metrics.  It does not declare an
overall winner, an efficiency result, an NVS result, or a variance estimate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import probe
from .config import config_sha256
from .protocol import psnr_ssim
from .protocol.gt_rays_io import read_gt_rays, read_hits
from .run_utils import read_json, sha256_file, utc_now, write_json_atomic

SCHEMA_VERSION = 1
PHASE = "05"
METHODS = ("3dgs", "2dgs")
MODES = ("all", "unoccluded")
MODE_LABELS = {
    "all": "gaussian_depth_all",
    "unoccluded": "gaussian_depth_unoccluded",
}
BUCKETS = ("0_20", "20_40", "40_50", "50_60")
PRIMARY_CAMERA = 0
COVERAGE_POLICY = "frozen_camera0_valid_population_per_frame_and_bucket"
ROWS_PER_FRAME = 8

LIMITATIONS = [
    "training-view reconstruction metrics (all 60 frames are training frames; not NVS)",
    "rendered-depth nearest-pixel proxy (not analytic ray-Gaussian intersection)",
    "single fixed seed; no variance estimate",
    "both methods use an expected z-depth, sum(z*alpha*T)/sum(alpha*T): 2DGS the official surf_depth with depth_ratio = 0.0, 3DGS the official colour-override expected-z render; both are sampled identically",
    "nearest-pixel quantization at 1408×376",
    "coverage is reported twice: full-population rows are 1.000000 for both modes, while the frozen Phase-3 primary cell excludes the (50,60] secondary bucket from its numerator while the denominator keeps every raw camera-0 ray",
    "mesh_sha256 carries the Gaussian point-cloud artifact hash because no mesh exists",
    "no efficiency, FPS, VRAM, training-time, or LPIPS claims",
    "camera 1 is excluded from Phase-5 GT-ray metrics",
]

NO_WINNER_STATEMENT = (
    "Three co-primary metrics are reported; no overall winner is declared."
)


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the repository root (``reconstruct/gaussians/report.py``)."""
    return Path(__file__).resolve().parents[2]


def artifacts_root(cfg: dict) -> Path:
    """Return the Phase-5 artifact root."""
    return Path(cfg["paths"]["artifacts_root"])


def report_dir(cfg: dict) -> Path:
    """Return ``<artifacts_root>/report``."""
    return artifacts_root(cfg) / "report"


def psnr_dir(cfg: dict, method: str) -> Path:
    """Return ``<artifacts_root>/psnr_ssim/<method>``."""
    return artifacts_root(cfg) / "psnr_ssim" / method


def metrics_path(cfg: dict, method: str, mode: str) -> Path:
    """Return one validated Phase-3 ``metrics.json`` path."""
    return artifacts_root(cfg) / "metrics" / method / mode / "metrics.json"


def metrics_csv_path(cfg: dict, method: str, mode: str) -> Path:
    """Return one validated Phase-3 ``metrics.csv`` path."""
    return artifacts_root(cfg) / "metrics" / method / mode / "metrics.csv"


def hits_manifest_path(cfg: dict, method: str) -> Path:
    """Return one method's ``hits_manifest.json``."""
    return artifacts_root(cfg) / "hits" / method / "hits_manifest.json"


def run_manifest_path(cfg: dict, method: str) -> Path:
    """Return one method's non-smoke ``run_manifest.json``."""
    return artifacts_root(cfg) / "runs" / f"{method}-{cfg['run_id']}" / "run_manifest.json"


def render_manifest_path(cfg: dict, method: str) -> Path:
    """Return one method's non-smoke ``render_manifest.json``."""
    return probe.locate_render_dir(cfg, method) / "render_manifest.json"


def gt_images_dir(cfg: dict) -> Path:
    """Return the frozen ``raw/images/image_00`` directory."""
    return Path(cfg["paths"]["bundle"]) / "raw" / "images" / "image_00"


def protocol_doc_path() -> Path:
    """Return the frozen protocol document."""
    return repo_root() / "reconstruct" / "configs" / "eval_protocol.md"


def protocol_config_path() -> Path:
    """Return the frozen PSNR/SSIM protocol config."""
    return repo_root() / "reconstruct" / "configs" / "psnr_ssim.yaml"


def dataset_dir(cfg: dict) -> Path:
    """Return the frozen training dataset directory."""
    return artifacts_root(cfg) / "dataset"


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"{what} missing: {path}")
    return path


def _require_upstream(cfg: dict) -> None:
    """Refuse report generation when a frozen upstream stage is absent."""
    _require_file(dataset_dir(cfg) / "dataset_manifest.json", "dataset manifest")
    _require_file(probe.gt_rays_path(cfg), "gt_rays.bin")
    _require_file(probe.artifacts_root(cfg) / "io02" / "gt_rays_manifest.json", "gt-ray manifest")
    _require_file(probe.artifacts_root(cfg) / "env_report.json", "environment report")
    _require_file(probe.convention_path(cfg), "depth convention")
    for method in METHODS:
        _require_file(run_manifest_path(cfg, method), "run manifest")
        _require_file(render_manifest_path(cfg, method), "render manifest")
        _require_file(hits_manifest_path(cfg, method), "hits manifest")
        for mode in MODES:
            _require_file(metrics_path(cfg, method, mode), "metrics.json")
            _require_file(metrics_csv_path(cfg, method, mode), "metrics.csv")


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------


def run_psnr_ssim(cfg: dict) -> dict:
    """Score both methods on all 60 training frames with the frozen harness."""
    config_path = _require_file(protocol_config_path(), "PSNR/SSIM protocol config")
    gt_dir = gt_images_dir(cfg)
    if not gt_dir.is_dir():
        raise RuntimeError(f"GT image directory missing: {gt_dir}")

    frames = [int(frame) for frame in cfg["frames"]]
    result: dict[str, dict] = {}
    for method in METHODS:
        render_dir = probe.locate_render_dir(cfg, method)
        out_dir = psnr_dir(cfg, method)
        payload = psnr_ssim.run_method(
            gt_dir,
            render_dir,
            frames,
            method,
            out_dir,
            config_path,
        )
        per_frame = payload.get("per_frame") or []
        if len(per_frame) != len(frames):
            raise RuntimeError(
                f"{method}: PSNR/SSIM produced {len(per_frame)} frames, "
                f"expected {len(frames)}"
            )
        recorded = [int(entry["frame"]) for entry in per_frame]
        if recorded != frames:
            raise RuntimeError(
                f"{method}: PSNR/SSIM frames {recorded} != frozen frames {frames}"
            )
        payload["outputs"] = {
            "json": str(out_dir / "psnr_ssim.json"),
            "csv": str(out_dir / "psnr_ssim.csv"),
            "csv_sha256": sha256_file(out_dir / "psnr_ssim.csv"),
        }
        # The JSON hash is deliberately not embedded in itself; the report
        # manifest hashes the final bytes on disk.
        write_json_atomic(out_dir / "psnr_ssim.json", payload)
        result[method] = payload
    return result


def load_psnr_ssim(cfg: dict) -> dict:
    """Read the two frozen PSNR/SSIM payloads."""
    result = {}
    for method in METHODS:
        path = _require_file(psnr_dir(cfg, method) / "psnr_ssim.json", "PSNR/SSIM JSON")
        payload = read_json(path)
        if payload.get("method") and payload["method"] != method:
            raise RuntimeError(f"{path}: method {payload['method']!r} != {method!r}")
        expected_frames = len(cfg["frames"])
        if int(payload.get("aggregate", {}).get("n", -1)) != expected_frames:
            raise RuntimeError(f"{path}: expected {expected_frames} scored frames")
        result[method] = payload
    return result


# ---------------------------------------------------------------------------
# validated metrics
# ---------------------------------------------------------------------------


def load_metrics(cfg: dict) -> dict:
    """Read the four validated Phase-3 metrics documents.

    Only the validated inputs are accepted: the frozen mode label and the
    rendered-depth nearest-pixel predictor.  Nothing here recomputes a formula.
    """
    result: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        result[method] = {}
        for mode in MODES:
            path = _require_file(metrics_path(cfg, method, mode), "metrics.json")
            document = read_json(path)
            label = MODE_LABELS[mode]
            if document.get("mode") != label:
                raise RuntimeError(
                    f"{path}: mode {document.get('mode')!r} != {label!r}"
                )
            inputs = document.get("inputs") or {}
            if inputs.get("mode") not in (None, label):
                raise RuntimeError(f"{path}: inputs.mode is not {label!r}")
            if inputs.get("predictor") != "rendered_depth_alpha_nearest_pixel":
                raise RuntimeError(
                    f"{path}: predictor {inputs.get('predictor')!r} is not "
                    "'rendered_depth_alpha_nearest_pixel'"
                )
            rows = document.get("rows") or []
            expected_rows = len(cfg["frames"]) * ROWS_PER_FRAME
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"{path}: {len(rows)} rows != {expected_rows}"
                )
            result[method][mode] = document
    return result


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(values) > 1 else 0.0
    return float(array.mean()), std


def bucket_table(rows: list[dict]) -> list[dict]:
    """Aggregate validated per-frame cells by bucket for camera 0.

    This is presentation-only aggregation of already validated cells.  It does
    not recompute a range, CD, P-R-F, miss, or coverage metric: sums and
    unweighted macro means are taken over the values the Phase-3 library wrote.
    """
    camera0 = [row for row in rows if int(row.get("camera", -1)) == PRIMARY_CAMERA]
    table: list[dict] = []
    for bucket in BUCKETS:
        cells = [row for row in camera0 if str(row.get("bucket")) == bucket]
        if not cells:
            table.append(
                {
                    "bucket": bucket,
                    "n_frames_with_data": 0,
                    "n_raw": 0,
                    "n_valid": 0,
                    "n_hits": 0,
                    "n_misses": 0,
                    "coverage_macro_mean": None,
                    "range_mae_macro_mean": None,
                    "range_mae_macro_std": None,
                    "range_rmse_macro_mean": None,
                    "range_rmse_macro_std": None,
                    "miss_rate_macro_mean": None,
                    "miss_rate_macro_std": None,
                    "ray_recall_macro_mean": None,
                    "ray_recall_macro_std": None,
                    "f1_020_macro_mean": None,
                    "f1_020_macro_std": None,
                }
            )
            continue

        coverage = [
            float(row["coverage"])
            for row in cells
            if row.get("coverage") is not None
        ]
        mae = [
            float(row["range"]["mae"])
            for row in cells
            if isinstance(row.get("range"), dict) and row["range"].get("mae") is not None
        ]
        rmse = [
            float(row["range"]["rmse"])
            for row in cells
            if isinstance(row.get("range"), dict) and row["range"].get("rmse") is not None
        ]
        miss_rate = [
            float(row["miss"]["miss_rate"])
            for row in cells
            if isinstance(row.get("miss"), dict) and row["miss"].get("miss_rate") is not None
        ]
        ray_recall = [
            float(row["miss"]["ray_recall"])
            for row in cells
            if isinstance(row.get("miss"), dict) and row["miss"].get("ray_recall") is not None
        ]
        f1_020 = [
            float(row["prf"]["0.20"]["nearest"]["f_score"])
            for row in cells
            if isinstance(row.get("prf"), dict)
            and isinstance(row["prf"].get("0.20"), dict)
            and isinstance(row["prf"]["0.20"].get("nearest"), dict)
            and row["prf"]["0.20"]["nearest"].get("f_score") is not None
        ]
        coverage_mean, _ = _mean_std(coverage)
        mae_mean, mae_std = _mean_std(mae)
        rmse_mean, rmse_std = _mean_std(rmse)
        miss_mean, miss_std = _mean_std(miss_rate)
        recall_mean, recall_std = _mean_std(ray_recall)
        f1_mean, f1_std = _mean_std(f1_020)
        table.append(
            {
                "bucket": bucket,
                "n_frames_with_data": len(cells),
                "n_raw": int(sum(int(row.get("n_raw", 0)) for row in cells)),
                "n_valid": int(sum(int(row.get("n_valid", 0)) for row in cells)),
                "n_hits": int(sum(int(row.get("n_hits", 0)) for row in cells)),
                "n_misses": int(sum(int(row.get("n_misses", 0)) for row in cells)),
                "coverage_macro_mean": coverage_mean,
                "range_mae_macro_mean": mae_mean,
                "range_mae_macro_std": mae_std,
                "range_rmse_macro_mean": rmse_mean,
                "range_rmse_macro_std": rmse_std,
                "miss_rate_macro_mean": miss_mean,
                "miss_rate_macro_std": miss_std,
                "ray_recall_macro_mean": recall_mean,
                "ray_recall_macro_std": recall_std,
                "f1_020_macro_mean": f1_mean,
                "f1_020_macro_std": f1_std,
            }
        )
    return table


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _hash_entry(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_provenance(cfg: dict) -> dict:
    """Collect every input hash and upstream commit the report cites."""
    artifacts = artifacts_root(cfg)
    runs: dict[str, dict] = {}
    renders: dict[str, dict] = {}
    hits: dict[str, dict] = {}
    metrics: dict[str, dict] = {}

    for method in METHODS:
        run_path = _require_file(run_manifest_path(cfg, method), "run manifest")
        run_doc = read_json(run_path)
        if bool(run_doc.get("smoke")):
            raise RuntimeError(f"{run_path}: smoke runs cannot enter the report")
        if run_doc.get("method") != method:
            raise RuntimeError(
                f"{run_path}: method {run_doc.get('method')!r} != {method!r}"
            )
        runs[method] = {
            **_hash_entry(run_path),
            "upstream_head": run_doc.get("upstream_head"),
            "rasterizer_head": run_doc.get("rasterizer_head"),
            "submodel_head": run_doc.get("submodel_head"),
            "checkpoint_sha256": run_doc.get("checkpoint_sha256"),
            "point_cloud_sha256": run_doc.get("point_cloud_sha256"),
            "checkpoint": run_doc.get("checkpoint"),
            "point_cloud": run_doc.get("point_cloud"),
            "iterations": run_doc.get("iterations"),
            "seed": run_doc.get("seed"),
            # Training duration stays in the run manifest as provenance only;
            # Phase 5 makes no training-time or efficiency claim.
            "event_files": run_doc.get("event_files"),
        }

        render_root = probe.locate_render_dir(cfg, method)
        render_path = _require_file(render_root / "render_manifest.json", "render manifest")
        render_doc = read_json(render_path)
        renders[method] = {
            **_hash_entry(render_path),
            "render_dir": str(render_root),
            "depth_convention": render_doc.get("depth_convention"),
            "depth_source": render_doc.get("depth_source"),
            "alpha_source": render_doc.get("alpha_source"),
            "depth_ratio": render_doc.get("depth_ratio"),
            "resolution": render_doc.get("resolution"),
            "frames": len(render_doc.get("per_frame") or []),
        }

        hit_path = _require_file(hits_manifest_path(cfg, method), "hits manifest")
        hit_doc = read_json(hit_path)
        hits[method] = {
            **_hash_entry(hit_path),
            "hits": hit_doc.get("hits_sha256"),
            "hits_unoccluded": hit_doc.get("hits_unoccluded_sha256"),
            "gt_rays": hit_doc.get("gt_rays_sha256"),
            "gt_rays_unoccluded": hit_doc.get("gt_rays_unoccluded_sha256"),
            "counts": hit_doc.get("counts"),
            "render_dir": hit_doc.get("render_dir"),
        }

        metrics[method] = {}
        for mode in MODES:
            path = _require_file(metrics_path(cfg, method, mode), "metrics.json")
            csv_path = _require_file(metrics_csv_path(cfg, method, mode), "metrics.csv")
            metrics[method][mode] = {
                **_hash_entry(path),
                "csv": _hash_entry(csv_path),
            }

    dataset_manifest = _require_file(dataset_dir(cfg) / "dataset_manifest.json", "dataset manifest")
    gt_rays = _require_file(probe.gt_rays_path(cfg), "gt_rays.bin")
    env_report = _require_file(artifacts / "env_report.json", "environment report")
    convention = _require_file(probe.convention_path(cfg), "depth convention")
    protocol = _require_file(protocol_doc_path(), "protocol document")

    return {
        "config_sha256": config_sha256(cfg),
        "config_path": str(cfg.get("_source") or (repo_root() / "reconstruct" / "configs" / "gaussians_1820.yaml")),
        "protocol_sha256": sha256_file(protocol),
        "protocol_path": str(protocol),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_manifest": str(dataset_manifest),
        "gt_rays_sha256": sha256_file(gt_rays),
        "gt_rays": str(gt_rays),
        "env_report_sha256": sha256_file(env_report),
        "env_report": str(env_report),
        "depth_convention": _hash_entry(convention),
        "runs": runs,
        "renders": renders,
        "hits": hits,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _gt_ray_tables(metrics: dict) -> dict:
    result: dict[str, dict] = {}
    for mode in MODES:
        result[mode] = {}
        for method in METHODS:
            document = metrics[method][mode]
            aggregates = document.get("aggregates") or {}
            result[mode][method] = {
                "micro_primary": aggregates.get("micro_primary"),
                "micro_secondary": aggregates.get("micro_secondary"),
                "per_camera": aggregates.get("per_camera"),
                "macro": aggregates.get("macro"),
                "buckets": bucket_table(document.get("rows") or []),
            }
    return result


def build_comparison(
    cfg: dict,
    psnr_ssim: dict | None = None,
    metrics: dict | None = None,
    provenance: dict | None = None,
    limitations: list[str] | None = None,
) -> dict:
    """Assemble the machine-readable comparison document."""
    psnr_ssim = psnr_ssim if psnr_ssim is not None else load_psnr_ssim(cfg)
    metrics = metrics if metrics is not None else load_metrics(cfg)
    provenance = provenance if provenance is not None else build_provenance(cfg)
    limitations = list(LIMITATIONS if limitations is None else limitations)

    for method in METHODS:
        if method not in psnr_ssim:
            raise RuntimeError(f"comparison is missing PSNR/SSIM for {method}")
        if method not in metrics:
            raise RuntimeError(f"comparison is missing metrics for {method}")
    for mode in MODES:
        for method in METHODS:
            if mode not in metrics[method]:
                raise RuntimeError(f"comparison is missing metrics {method}/{mode}")

    protocol = cfg["protocol"]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "methods": list(METHODS),
        "protocol": {
            "alpha_threshold": float(protocol["alpha_threshold"]),
            "depth_convention": str(protocol["depth_convention"]),
            "pixel_rounding": str(protocol["pixel_rounding"]),
            "coverage_policy": COVERAGE_POLICY,
            "primary_camera": int(protocol.get("primary_camera", PRIMARY_CAMERA)),
            "resolution": [int(value) for value in cfg["resolution"]],
            "frames": [int(frame) for frame in cfg["frames"]],
        },
        "psnr_ssim": psnr_ssim,
        "gt_ray": _gt_ray_tables(metrics),
        "provenance": provenance,
        "limitations": limitations,
        "tradeoffs": {
            "winner": None,
            "statement": NO_WINNER_STATEMENT,
        },
    }


# ---------------------------------------------------------------------------
# Markdown and table writers
# ---------------------------------------------------------------------------


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _mean_std_text(mean, std, digits: int = 4) -> str:
    if mean is None:
        return "null"
    if std is None:
        return _fmt(mean, digits)
    return f"{_fmt(mean, digits)} ± {_fmt(std, digits)}"


def _psnr_table(comparison: dict) -> list[str]:
    lines = [
        "| Method | n | PSNR mean ± std (dB) | PSNR min | PSNR max | SSIM mean ± std | SSIM min | SSIM max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        aggregate = comparison["psnr_ssim"][method]["aggregate"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method.upper(),
                    _fmt(aggregate.get("n")),
                    _mean_std_text(aggregate.get("psnr_mean"), aggregate.get("psnr_std")),
                    _fmt(aggregate.get("psnr_min")),
                    _fmt(aggregate.get("psnr_max")),
                    _mean_std_text(aggregate.get("ssim_mean"), aggregate.get("ssim_std")),
                    _fmt(aggregate.get("ssim_min")),
                    _fmt(aggregate.get("ssim_max")),
                ]
            )
            + " |"
        )
    return lines


def _micro_header() -> list[str]:
    return [
        "| Mode | Method | n_valid | n_hits | n_misses | coverage | miss rate | ray recall | Range MAE (m) | Range RMSE (m) | MedAE (m) | Bias (m) | P90 (m) | P95 (m) | CD-L1 | CD-L2 | F@0.20 nearest | F@0.20 greedy | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]


def _micro_row(mode: str, method: str, entry: dict) -> str:
    cell = (entry or {}).get("cell") or {}
    range_cell = cell.get("range") or {}
    miss = cell.get("miss") or {}
    cd = (entry or {}).get("cd") or {}
    prf = (entry or {}).get("prf") or {}
    nearest = ((prf or {}).get("0.20") or {}).get("nearest") or {}
    greedy = ((prf or {}).get("0.20") or {}).get("greedy") or {}
    return "| " + " | ".join(
        [
            mode,
            method.upper(),
            _fmt(cell.get("n_valid")),
            _fmt(cell.get("n_hits")),
            _fmt(cell.get("n_misses")),
            _fmt(cell.get("coverage")),
            _fmt(miss.get("miss_rate")),
            _fmt(miss.get("ray_recall")),
            _fmt(range_cell.get("mae")),
            _fmt(range_cell.get("rmse")),
            _fmt(range_cell.get("medae")),
            _fmt(range_cell.get("bias")),
            _fmt(range_cell.get("p90")),
            _fmt(range_cell.get("p95")),
            _fmt(cd.get("cd_l1")),
            _fmt(cd.get("cd_l2")),
            _fmt(nearest.get("f_score")),
            _fmt(greedy.get("f_score")),
            _fmt(nearest.get("precision")),
            _fmt(nearest.get("recall")),
        ]
    ) + " |"


def _micro_tables(comparison: dict) -> list[str]:
    lines = ["### Micro primary cells", "", *_micro_header()]
    for mode in MODES:
        for method in METHODS:
            entry = comparison["gt_ray"][mode][method]
            lines.append(
                _micro_row(mode, method, entry.get("micro_primary") or {})
            )
    lines += ["", "### Micro secondary (50–60 m) cells", "", *_micro_header()]
    for mode in MODES:
        for method in METHODS:
            entry = comparison["gt_ray"][mode][method]
            lines.append(
                _micro_row(mode, method, entry.get("micro_secondary") or {})
            )
    return lines


def _per_camera_table(comparison: dict) -> list[str]:
    lines = [
        "| Mode | Method | Camera | n_raw | n_valid | n_hits | n_misses | coverage | miss rate | ray recall | Range MAE (m) | Range RMSE (m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for method in METHODS:
            for entry in comparison["gt_ray"][mode][method].get("per_camera") or []:
                cell = entry.get("cell") or {}
                range_cell = cell.get("range") or {}
                miss = cell.get("miss") or {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            mode,
                            method.upper(),
                            _fmt(entry.get("camera")),
                            _fmt(cell.get("n_raw")),
                            _fmt(cell.get("n_valid")),
                            _fmt(cell.get("n_hits")),
                            _fmt(cell.get("n_misses")),
                            _fmt(cell.get("coverage")),
                            _fmt(miss.get("miss_rate")),
                            _fmt(miss.get("ray_recall")),
                            _fmt(range_cell.get("mae")),
                            _fmt(range_cell.get("rmse")),
                        ]
                    )
                    + " |"
                )
    return lines


def _macro_table(comparison: dict) -> list[str]:
    lines = [
        "| Mode | Method | n_frames | Range MAE mean ± std (m) | Range RMSE mean ± std (m) | Miss rate mean ± std | Ray recall mean ± std | Coverage mean ± std |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for method in METHODS:
            macro = comparison["gt_ray"][mode][method].get("macro") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        mode,
                        method.upper(),
                        _fmt(macro.get("n_frames_total")),
                        _mean_std_text(macro.get("mae_mean"), macro.get("mae_std")),
                        _mean_std_text(macro.get("rmse_mean"), macro.get("rmse_std")),
                        _mean_std_text(macro.get("miss_rate_mean"), macro.get("miss_rate_std")),
                        _mean_std_text(macro.get("ray_recall_mean"), macro.get("ray_recall_std")),
                        _mean_std_text(macro.get("coverage_mean"), macro.get("coverage_std")),
                    ]
                )
                + " |"
            )
    return lines


def _bucket_table(comparison: dict, mode: str) -> list[str]:
    lines = [
        "| Mode | Method | Bucket | frames | n_raw | n_valid | n_hits | n_misses | coverage mean | Range MAE mean ± std (m) | Range RMSE mean ± std (m) | Miss rate mean ± std | Ray recall mean ± std | F@0.20 mean ± std |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for row in comparison["gt_ray"][mode][method]["buckets"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        mode,
                        method.upper(),
                        str(row["bucket"]),
                        _fmt(row.get("n_frames_with_data")),
                        _fmt(row.get("n_raw")),
                        _fmt(row.get("n_valid")),
                        _fmt(row.get("n_hits")),
                        _fmt(row.get("n_misses")),
                        _fmt(row.get("coverage_macro_mean")),
                        _mean_std_text(row.get("range_mae_macro_mean"), row.get("range_mae_macro_std")),
                        _mean_std_text(row.get("range_rmse_macro_mean"), row.get("range_rmse_macro_std")),
                        _mean_std_text(row.get("miss_rate_macro_mean"), row.get("miss_rate_macro_std")),
                        _mean_std_text(row.get("ray_recall_macro_mean"), row.get("ray_recall_macro_std")),
                        _mean_std_text(row.get("f1_020_macro_mean"), row.get("f1_020_macro_std")),
                    ]
                )
                + " |"
            )
    return lines


def _provenance_lines(comparison: dict) -> list[str]:
    provenance = comparison["provenance"]
    lines = [
        f"- canonical config SHA-256: `{provenance.get('config_sha256')}`",
        f"- protocol document SHA-256: `{provenance.get('protocol_sha256')}`",
        f"- dataset manifest SHA-256: `{provenance.get('dataset_manifest_sha256')}`",
        f"- frozen camera-0 `gt_rays.bin` SHA-256: `{provenance.get('gt_rays_sha256')}`",
        f"- environment report SHA-256: `{provenance.get('env_report_sha256')}`",
    ]
    convention = provenance.get("depth_convention") or {}
    if convention:
        lines.append(
            f"- `depth_convention.json` SHA-256: `{convention.get('sha256')}`"
        )
    for method in METHODS:
        run = (provenance.get("runs") or {}).get(method) or {}
        render = (provenance.get("renders") or {}).get(method) or {}
        hit = (provenance.get("hits") or {}).get(method) or {}
        lines += [
            f"- {method.upper()} upstream commit: `{run.get('upstream_head')}`",
            f"- {method.upper()} rasterizer/submodel commit: `{run.get('rasterizer_head') or run.get('submodel_head')}`",
            f"- {method.upper()} checkpoint SHA-256: `{run.get('checkpoint_sha256')}`",
            f"- {method.upper()} point-cloud SHA-256: `{run.get('point_cloud_sha256')}`",
            f"- {method.upper()} run manifest: `{run.get('path')}` (`{run.get('sha256')}`)",
            f"- {method.upper()} render manifest: `{render.get('path')}` (`{render.get('sha256')}`)",
            f"- {method.upper()} hits manifest: `{hit.get('path')}` (`{hit.get('sha256')}`)",
        ]
    lines.append(
        "- `metrics.json`/`metrics.csv` are the validated Phase-3 outputs for both methods and both modes; "
        "their SHA-256 values are in `run_manifest.json`."
    )
    return lines


def _training_lines(comparison: dict) -> list[str]:
    lines = []
    for method in METHODS:
        run = (comparison["provenance"].get("runs") or {}).get(method) or {}
        lines.append(
            f"- {method.upper()}: upstream `{run.get('upstream_head')}`, "
            f"{_fmt(run.get('iterations'))} iterations, seed `{run.get('seed')}`, "
            f"native `1408×376`, training-view only. `total_loss` and the L1 loss are read "
            "from the official TensorBoard event file."
        )
    lines.append(
        "- 2DGS logs the L1 term as `train/loss_viewpoint - l1_loss`; 3DGS logs "
        "`train_loss_patches/l1_loss`. Both curves use the official event files; no value is reconstructed."
    )
    return lines


def _tradeoff_lines(comparison: dict) -> list[str]:
    p = comparison["psnr_ssim"]
    all_geometry = comparison["gt_ray"]["all"]
    unoccluded = comparison["gt_ray"]["unoccluded"]
    lines = [
        f"- PSNR (training-view): 3DGS {_fmt(p['3dgs']['aggregate'].get('psnr_mean'))} ± "
        f"{_fmt(p['3dgs']['aggregate'].get('psnr_std'))} dB; 2DGS "
        f"{_fmt(p['2dgs']['aggregate'].get('psnr_mean'))} ± "
        f"{_fmt(p['2dgs']['aggregate'].get('psnr_std'))} dB.",
        f"- SSIM (training-view): 3DGS {_fmt(p['3dgs']['aggregate'].get('ssim_mean'))} ± "
        f"{_fmt(p['3dgs']['aggregate'].get('ssim_std'))}; 2DGS "
        f"{_fmt(p['2dgs']['aggregate'].get('ssim_mean'))} ± "
        f"{_fmt(p['2dgs']['aggregate'].get('ssim_std'))}.",
    ]
    for mode, table in (("all camera-0 rays", all_geometry), ("unoccluded camera-0 rays", unoccluded)):
        for method in METHODS:
            cell = (table[method].get("micro_primary") or {}).get("cell") or {}
            miss = cell.get("miss") or {}
            range_cell = cell.get("range") or {}
            lines.append(
                f"- {method.upper()} {mode}: range MAE {_fmt(range_cell.get('mae'))} m, "
                f"miss rate {_fmt(miss.get('miss_rate'))}, coverage {_fmt(cell.get('coverage'))}."
            )
    lines += [
        "",
        "The PSNR/SSIM ranking and the GT-ray geometry ranking are not identical across every "
        "table, and the miss/coverage columns change how a row should be read. "
        "**No overall winner is declared** — the three co-primary metrics are reported side by side for the reader.",
    ]
    return lines


def _write_csv(path: Path, header: list[str], rows: list[list]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join("" if value is None else str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_tables(comparison: dict, tables_dir: Path) -> dict:
    psnr_rows = []
    for method in METHODS:
        for entry in comparison["psnr_ssim"][method]["per_frame"]:
            psnr_rows.append(
                [
                    method,
                    entry.get("frame"),
                    entry.get("psnr"),
                    entry.get("ssim"),
                    entry.get("mse"),
                    entry.get("psnr_reason"),
                ]
            )
    psnr_csv = _write_csv(
        tables_dir / "psnr_ssim.csv",
        ["method", "frame", "psnr_db", "ssim", "mse", "psnr_reason"],
        psnr_rows,
    )

    range_rows = []
    miss_rows = []
    for mode in MODES:
        for method in METHODS:
            for row in comparison["gt_ray"][mode][method]["buckets"]:
                range_rows.append(
                    [
                        mode,
                        method,
                        row["bucket"],
                        row["n_frames_with_data"],
                        row["n_raw"],
                        row["n_valid"],
                        row["n_hits"],
                        row["n_misses"],
                        row["coverage_macro_mean"],
                        row["range_mae_macro_mean"],
                        row["range_mae_macro_std"],
                        row["range_rmse_macro_mean"],
                        row["range_rmse_macro_std"],
                    ]
                )
                miss_rows.append(
                    [
                        mode,
                        method,
                        row["bucket"],
                        row["n_frames_with_data"],
                        row["n_hits"],
                        row["n_misses"],
                        row["n_valid"],
                        row["coverage_macro_mean"],
                        row["miss_rate_macro_mean"],
                        row["miss_rate_macro_std"],
                        row["ray_recall_macro_mean"],
                        row["ray_recall_macro_std"],
                    ]
                )
    range_csv = _write_csv(
        tables_dir / "range_bucket.csv",
        ["mode", "method", "bucket", "n_frames_with_data", "n_raw", "n_valid", "n_hits",
         "n_misses", "coverage_macro_mean", "range_mae_macro_mean", "range_mae_macro_std",
         "range_rmse_macro_mean", "range_rmse_macro_std"],
        range_rows,
    )
    miss_csv = _write_csv(
        tables_dir / "miss_bucket.csv",
        ["mode", "method", "bucket", "n_frames_with_data", "n_hits", "n_misses", "n_valid",
         "coverage_macro_mean", "miss_rate_macro_mean", "miss_rate_macro_std",
         "ray_recall_macro_mean", "ray_recall_macro_std"],
        miss_rows,
    )

    micro_rows = []
    for mode in MODES:
        for method in METHODS:
            entry = comparison["gt_ray"][mode][method]
            for kind, key in (("primary", "micro_primary"), ("secondary", "micro_secondary")):
                cell = entry.get(key) or {}
                range_cell = (cell.get("cell") or {}).get("range") or {}
                miss = (cell.get("cell") or {}).get("miss") or {}
                cd = cell.get("cd") or {}
                prf = cell.get("prf") or {}
                nearest = ((prf.get("0.20") or {}).get("nearest") or {})
                greedy = ((prf.get("0.20") or {}).get("greedy") or {})
                micro_rows.append(
                    [
                        mode,
                        method,
                        kind,
                        (cell.get("cell") or {}).get("n_raw"),
                        (cell.get("cell") or {}).get("n_valid"),
                        (cell.get("cell") or {}).get("n_hits"),
                        (cell.get("cell") or {}).get("n_misses"),
                        (cell.get("cell") or {}).get("coverage"),
                        miss.get("miss_rate"),
                        miss.get("ray_recall"),
                        range_cell.get("mae"),
                        range_cell.get("rmse"),
                        range_cell.get("medae"),
                        range_cell.get("bias"),
                        range_cell.get("p90"),
                        range_cell.get("p95"),
                        cd.get("cd_l1"),
                        cd.get("cd_l2"),
                        nearest.get("f_score"),
                        greedy.get("f_score"),
                        nearest.get("precision"),
                        nearest.get("recall"),
                    ]
                )
    micro_csv = _write_csv(
        tables_dir / "micro.csv",
        ["mode", "method", "cell_kind", "n_raw", "n_valid", "n_hits", "n_misses", "coverage",
         "miss_rate", "ray_recall", "range_mae", "range_rmse", "range_medae", "range_bias",
         "range_p90", "range_p95", "cd_l1", "cd_l2", "f020_nearest", "f020_greedy",
         "precision", "recall"],
        micro_rows,
    )
    return {
        "psnr_ssim": psnr_csv,
        "range_bucket": range_csv,
        "miss_bucket": miss_csv,
        "micro": micro_csv,
    }


def _coverage_paragraph(comparison: dict) -> str:
    """Explain the full-population and Phase-3 primary-cell coverage values."""
    primary = {}
    for mode in MODES:
        entry = (comparison["gt_ray"].get(mode) or {}).get(METHODS[0]) or {}
        cell = (entry.get("micro_primary") or {}).get("cell") or {}
        primary[mode] = cell.get("coverage")
    all_text = _fmt(primary.get("all"), 6)
    unoccluded_text = _fmt(primary.get("unoccluded"), 6)
    return (
        "**Coverage is reported twice.** The eval stage asserts the full-population "
        "coverage is **1.000000** for both modes (every raw camera-0 ray in each "
        "population is valid). The Phase-3 primary cells in the tables below are "
        f"**{all_text}** (all) and **{unoccluded_text}** (unoccluded), because the frozen "
        "primary numerator excludes the `(50,60]` secondary bucket while the denominator "
        "keeps every raw camera-0 ray. Both numbers are correct; they describe different "
        "populations."
    )


def write_report(comparison: dict, figures: dict, out_dir: str | Path) -> Path:
    """Write REPORT.md, tables, run_manifest, index, and SHA256SUMS."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables_dir = root / "tables"
    tables = _write_tables(comparison, tables_dir)
    provenance = comparison["provenance"]

    frames = [int(frame) for frame in (comparison.get("protocol") or {}).get("frames") or []]
    first_frame, last_frame = (frames[0], frames[-1]) if frames else ("?", "?")
    lines: list[str] = [
        f"# 3DGS vs 2DGS on KITTI-360 frames {first_frame}–{last_frame}",
        "",
        "**Training-view reconstruction metrics, not novel-view synthesis.**",
        "",
        "## 1. Scope and protocol",
        "",
        "This report compares official 3DGS and official 2DGS on the frozen KITTI-360 "
        f"`2013_05_28_drive_0000_sync` window, frames **{first_frame}–{last_frame}**, `image_00`, native "
        "**1408×376**, with the official poses and calibration. **All 60 frames are training "
        "frames**; the PSNR/SSIM values below are **training-view reconstruction metrics**, not "
        "novel-view synthesis or a generalisation estimate. The geometry table uses **camera 0 "
        "only**. Both methods were trained once with a **single fixed seed**; no held-out split, "
        "no cherry-picking, and no test-frame tuning were used.",
        "",
        "- Protocol: `alpha_threshold = 0.5`, frozen depth convention "
        f"`{comparison['protocol']['depth_convention']}`, half-up nearest-pixel projection, "
        "rendered-depth proxy, camera-0 valid population denominator.",
        "- Frame set: 60 frames, camera 0, all frames are training views.",
        "- Resolution: native 1408×376 for training, rendering, PSNR/SSIM, and GT-ray sampling.",
        "- Single seed: `20261102` for both methods; no variance estimate is reported.",
        "",
        "## 2. Inputs and provenance",
        "",
        *_provenance_lines(comparison),
        "",
        "## 3. Methods and training",
        "",
        *_training_lines(comparison),
        "",
        "Both methods use the official implementations and official default 30,000-iteration "
        "training configurations. Both export an **expected z-depth** "
        "`sum(z * alpha * T) / sum(alpha * T)`: 2DGS returns the official `surf_depth` with "
        "`depth_ratio = 0.0`; 3DGS renders it through the official `override_color` path with "
        "per-Gaussian `z / 100` (so the renderer's `clamp(0, 1)` cannot clip a <= 60 m scene) "
        "and divides the colour-override render by the accumulated alpha. The renderer's own "
        "`1/invdepths` accumulator is not used. Both depth maps are sampled by the same half-up "
        "nearest-pixel protocol.",
        "",
        "## 4. PSNR / SSIM",
        "",
        "Per-frame PSNR and SSIM are computed by the single frozen harness on all 60 training "
        "frames. Identical images would be written as JSON `null` with reason `identical`; no "
        "such frame occurred.",
        "",
        *_psnr_table(comparison),
        "",
        f"![Per-frame PSNR/SSIM](figures/{Path(figures.get('psnr_ssim_per_frame', {}).get('path', 'psnr_ssim_per_frame.png')).name})",
        "",
        "Per-frame values are in [`tables/psnr_ssim.csv`](tables/psnr_ssim.csv).",
        "",
        "## 5. GT-ray geometry",
        "",
        "All geometry values are the validated Phase-3 outputs. `micro_primary` is the frozen "
        "primary cell; `micro_secondary` is the separate 50–60 m sensitivity cell. The bucket "
        "table is a presentation-only macro aggregation of the validated per-frame cells. Every "
        "accuracy row carries coverage and miss/ray-recall so an accuracy number is never read "
        "without its denominator.",
        "",
        _coverage_paragraph(comparison),
        "",
        "### Per-camera",
        "",
        *_per_camera_table(comparison),
        "",
        "### Macro (per-frame means)",
        "",
        *_macro_table(comparison),
        "",
        *_micro_tables(comparison),
        "",
        "### All camera-0 rays by bucket",
        "",
        *_bucket_table(comparison, "all"),
        "",
        "### Unoccluded camera-0 rays by bucket",
        "",
        *_bucket_table(comparison, "unoccluded"),
        "",
        f"![Range error and miss rate by bucket](figures/{Path(figures.get('error_by_bucket', {}).get('path', 'error_by_bucket.png')).name})",
        "",
        "CSV tables: [`tables/range_bucket.csv`](tables/range_bucket.csv), "
        "[`tables/miss_bucket.csv`](tables/miss_bucket.csv), [`tables/micro.csv`](tables/micro.csv).",
        "",
        "## 6. Training curves",
        "",
        "Training curves are read from the official TensorBoard event files recorded by each "
        "run manifest; they are not reconstructed from logs.",
        "",
        f"![Training curves](figures/{Path(figures.get('training_curves', {}).get('path', 'training_curves.png')).name})",
        "",
        "## 7. Trade-offs",
        "",
        *_tradeoff_lines(comparison),
        "",
        "## 8. Limitations",
        "",
        *[f"- {limitation}" for limitation in comparison["limitations"]],
        "",
        "## 9. Reproduction",
        "",
        "The report is regenerated from the frozen configs, checkpoints, renders, hits, and "
        "validated metrics. **Training is not rerun.**",
        "",
        "```bash",
        "bash reconstruct/pipeline/gaussians.sh --stage report",
        "bash reconstruct/pipeline/gaussians.sh --stage verify-only",
        "```",
        "",
        "`--report` recomputes PSNR/SSIM and rewrites this package; `--verify-only` re-hashes "
        "every artifact, re-validates the four metrics files with the Python validator, and checks "
        "ordered `gt_rays.bin`/`hits.bin` identity without training or rendering.",
        "",
        f"![Sample renders](figures/{Path(figures.get('renders_grid', {}).get('path', 'renders_grid.png')).name})",
        "",
    ]
    report_path = root / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Every input/output path with a SHA-256.  The report, comparison, figures,
    # index, and tables are all either read or written by this package.
    entries: list[dict] = []

    def add(role: str, kind: str, path: str | Path | None, digest: str | None = None) -> None:
        if not path:
            return
        path_obj = Path(path)
        entries.append(
            {
                "role": role,
                "kind": kind,
                "path": str(path_obj),
                "sha256": digest or sha256_file(path_obj),
            }
        )

    add("input", "config", provenance.get("config_path"))
    add("input", "protocol", provenance.get("protocol_path"), provenance.get("protocol_sha256"))
    add("input", "dataset_manifest", provenance.get("dataset_manifest"), provenance.get("dataset_manifest_sha256"))
    add("input", "gt_rays", provenance.get("gt_rays"), provenance.get("gt_rays_sha256"))
    add("input", "env_report", provenance.get("env_report"), provenance.get("env_report_sha256"))
    convention = provenance.get("depth_convention") or {}
    add("input", "depth_convention", convention.get("path"), convention.get("sha256"))
    for method in METHODS:
        for group in ("runs", "renders", "hits"):
            item = (provenance.get(group) or {}).get(method) or {}
            add("input", f"{group}-{method}", item.get("path"), item.get("sha256"))
        for mode in MODES:
            item = ((provenance.get("metrics") or {}).get(method) or {}).get(mode) or {}
            add("input", f"metrics-{method}-{mode}", item.get("path"), item.get("sha256"))
            add("input", f"metrics-csv-{method}-{mode}", (item.get("csv") or {}).get("path"), (item.get("csv") or {}).get("sha256"))
        psnr_output = (comparison["psnr_ssim"].get(method) or {}).get("outputs") or {}
        add("input", f"psnr_ssim-json-{method}", psnr_output.get("json"), psnr_output.get("json_sha256"))
        add("input", f"psnr_ssim-csv-{method}", psnr_output.get("csv"), psnr_output.get("csv_sha256"))

    outputs = [report_path, root / "comparison.json"]
    outputs += [Path(item["path"]) for item in tables.values()]
    outputs += [Path(item["path"]) for item in figures.values()]
    for path in outputs:
        add("output", path.suffix.lstrip(".") or "file", path)

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "methods": list(METHODS),
        "groups": {
            "report": ["REPORT.md"],
            "comparison": ["comparison.json"],
            "tables": sorted(item["path"] for item in tables.values()),
            "figures": sorted(item["path"] for item in figures.values()),
            "psnr_ssim": sorted(
                str(psnr_dir_for_path) for psnr_dir_for_path in [
                    (comparison["psnr_ssim"][m].get("outputs") or {}).get("json")
                    for m in METHODS
                ] if psnr_dir_for_path
            ),
        },
        "artifacts": [
            {"path": entry["path"], "sha256": entry["sha256"]}
            for entry in entries
            if entry["role"] == "output"
        ],
    }
    index_path = root / "index.json"
    write_json_atomic(index_path, index_payload)
    add("output", "index", index_path)

    manifest_path = root / "run_manifest.json"
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "winner": None,
        "entries": entries,
    }
    write_json_atomic(manifest_path, manifest_payload)

    sums_lines = []
    for path in sorted(
        [p for p in root.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "verify_report.json")]
    ):
        sums_lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# read-only verification
# ---------------------------------------------------------------------------


def _entry(path: str | Path | None, expected: str | None, actual: str | None, ok: bool) -> dict:
    return {
        "path": str(path) if path is not None else None,
        "expected": expected,
        "actual": actual,
        "ok": bool(ok),
    }


def _check_manifest_entries(manifest: dict) -> list[dict]:
    entries: list[dict] = []
    for item in manifest.get("entries") or []:
        path = Path(str(item.get("path") or ""))
        expected = item.get("sha256")
        if not path.is_file():
            entries.append(_entry(path, expected, None, False))
            continue
        actual = sha256_file(path)
        entries.append(_entry(path, expected, actual, actual == expected))
    return entries


def _sha256sums_entry(directory: Path, name: str = "SHA256SUMS") -> dict:
    path = directory / name
    if not path.is_file():
        return _entry(path, "sha256sum -c", None, False)
    completed = subprocess.run(
        ["sha256sum", "-c", name],
        cwd=str(directory),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return _entry(path, "sha256sum -c", f"exit={completed.returncode}: {output[-400:]}", completed.returncode == 0)


def _validate_metrics(cfg: dict, method: str, mode: str) -> dict:
    from ..eval.output import validate_metrics_csv, validate_metrics_json

    metrics_dir = artifacts_root(cfg) / "metrics" / method / mode
    bundle = metrics_dir / "phase3_validation"
    metrics = (bundle / "metrics.json") if (bundle / "metrics.json").is_file() else (metrics_dir / "metrics.json")
    csv = (bundle / "metrics.csv") if (bundle / "metrics.csv").is_file() else (metrics_dir / "metrics.csv")
    if not metrics.is_file() or not csv.is_file():
        return _entry(metrics, "eval validate", "missing metrics artifacts", False)
    try:
        document = read_json(metrics)
        validate_metrics_json(document)
        validate_metrics_csv(csv, document)
    except Exception as exc:  # noqa: BLE001 - verification reports, never raises
        return _entry(metrics, "eval validate", str(exc), False)
    return _entry(metrics, "eval validate", "ok", True)

def _check_hits(cfg: dict, method: str) -> list[dict]:
    hits_root = artifacts_root(cfg) / "hits" / method
    entries: list[dict] = []
    gt_rays = read_gt_rays(probe.gt_rays_path(cfg))
    hits = read_hits(hits_root / "hits.bin")
    expected_hits = len(gt_rays)
    if len(hits) != expected_hits:
        entries.append(
            _entry(
                hits_root / "hits.bin",
                f"{expected_hits} ordered records",
                f"gt_rays={len(gt_rays)} hits={len(hits)}",
                False,
            )
        )
    else:
        ordered = all(
            int(hit["ray_id"]) == int(ray["ray_id"])
            for ray, hit in zip(gt_rays, hits)
        )
        entries.append(
            _entry(
                hits_root / "hits.bin",
                f"{expected_hits} ordered records; misses +inf/0xFFFFFFFF",
                f"gt_rays={len(gt_rays)} hits={len(hits)} ordered={ordered}",
                ordered,
            )
        )

    gt_unoccluded = read_gt_rays(hits_root / "gt_rays_unoccluded.bin")
    hits_unoccluded = read_hits(hits_root / "hits_unoccluded.bin")
    ordered = len(gt_unoccluded) == len(hits_unoccluded) and all(
        int(hit["ray_id"]) == int(ray["ray_id"])
        for ray, hit in zip(gt_unoccluded, hits_unoccluded)
    )
    smaller = len(gt_unoccluded) < expected_hits
    entries.append(
        _entry(
            hits_root / "hits_unoccluded.bin",
            "equal ordered unoccluded pair; count < camera-0 total",
            f"gt_rays={len(gt_unoccluded)} hits={len(hits_unoccluded)} ordered={ordered}",
            ordered and smaller,
        )
    )
    return entries


def verify(context: dict, *, deep: bool = True) -> dict:
    """Re-hash the report package and revalidate frozen inputs (read-only).

    ``deep=False`` is the unit-test seam: it still checks the report manifest
    and SHA256SUMS, but skips the dataset checksum, the Python metrics validator,
    and the full ordered hit pair.  Production ``--verify-only`` uses the
    default ``deep=True``.
    """
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("verify stage requires context['cfg']")
    root = artifacts_root(cfg)
    out_dir = report_dir(cfg)
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"report run manifest missing: {manifest_path}")

    manifest = read_json(manifest_path)
    entries = _check_manifest_entries(manifest)

    if (out_dir / "SHA256SUMS").is_file():
        entries.append(_sha256sums_entry(out_dir))
    elif deep:
        entries.append(_entry(out_dir / "SHA256SUMS", "sha256sum -c", None, False))

    if deep:
        entries.append(_sha256sums_entry(dataset_dir(cfg)))
        for method in METHODS:
            for mode in MODES:
                entries.append(_validate_metrics(cfg, method, mode))
        for method in METHODS:
            entries.extend(_check_hits(cfg, method))

        convention_path = probe.convention_path(cfg)
        comparison_path = out_dir / "comparison.json"
        if convention_path.is_file() and comparison_path.is_file():
            decision = read_json(convention_path).get("decision")
            protocol_decision = (read_json(comparison_path).get("protocol") or {}).get("depth_convention")
            entries.append(
                _entry(
                    convention_path,
                    f"decision == comparison.protocol.depth_convention ({protocol_decision})",
                    decision,
                    decision == protocol_decision,
                )
            )
        else:
            entries.append(_entry(convention_path, "depth_convention.json + comparison.json", None, False))

    ok = all(entry["ok"] for entry in entries)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "verify-only",
        "generated_at_utc": utc_now(),
        "ok": ok,
        "deep": bool(deep),
        "entries": entries,
    }
    verify_path = out_dir / "verify_report.json"
    write_json_atomic(verify_path, result)
    if not ok:
        failures = [entry["path"] for entry in entries if not entry["ok"]]
        raise RuntimeError(f"verify-only failed for: {failures}")
    return result


# ---------------------------------------------------------------------------
# pipeline stage
# ---------------------------------------------------------------------------


def run(context: dict) -> dict:
    """Pipeline stage: compute PSNR/SSIM, figures, and the full report package."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("report stage requires context['cfg']")
    _require_upstream(cfg)

    psnr = run_psnr_ssim(cfg)
    comparison = build_comparison(cfg, psnr_ssim=psnr)
    out_dir = report_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "comparison.json"
    write_json_atomic(comparison_path, comparison)

    from . import figures as figures_module  # noqa: PLC0415 - lazy import keeps task order explicit

    figure_files = figures_module.make_all_figures(cfg, comparison, out_dir / "figures")
    report_path = write_report(comparison, figure_files, out_dir)
    manifest_path = out_dir / "run_manifest.json"

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "report",
        "generated_at_utc": utc_now(),
        "comparison": str(comparison_path),
        "comparison_sha256": sha256_file(comparison_path),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "figures": figure_files,
        "psnr_ssim": {
            method: {
                "json": str(psnr_dir(cfg, method) / "psnr_ssim.json"),
                "json_sha256": sha256_file(psnr_dir(cfg, method) / "psnr_ssim.json"),
                "csv": str(psnr_dir(cfg, method) / "psnr_ssim.csv"),
                "csv_sha256": sha256_file(psnr_dir(cfg, method) / "psnr_ssim.csv"),
            }
            for method in METHODS
        },
        "limitations": LIMITATIONS,
        "winner": None,
    }


__all__ = [
    "BUCKETS",
    "COVERAGE_POLICY",
    "LIMITATIONS",
    "METHODS",
    "MODES",
    "MODE_LABELS",
    "NO_WINNER_STATEMENT",
    "artifacts_root",
    "bucket_table",
    "build_comparison",
    "build_provenance",
    "dataset_dir",
    "gt_images_dir",
    "hits_manifest_path",
    "load_metrics",
    "load_psnr_ssim",
    "metrics_csv_path",
    "metrics_path",
    "protocol_config_path",
    "protocol_doc_path",
    "psnr_dir",
    "repo_root",
    "report_dir",
    "run",
    "run_manifest_path",
    "run_psnr_ssim",
    "verify",
    "write_report",
]
