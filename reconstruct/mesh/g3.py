"""G3 diagnostics for the mesh reconstruction package.

Three measurements are computed from local artifacts only:

1. ``bbox_ratio`` — the x/y mesh footprint over the camera-visible GT footprint
   (``camera < 2``, ``valid == 1``; the GT cache decoded read-only), plus a z
   sanity band;
2. ``components`` — welded-vertex connected components and the largest
   component fraction;
3. ``degenerate`` — triangles with a repeated index or area below ``1e-12 m²``.

These are diagnostics, not a hard pass/fail gate.  The reconstruction package
does not reject a mesh because a reference value is missed; the numbers are
written to ``g3_report.json`` for the consumer.  No geometric correction,
fitting, transform, or rescaling is applied.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import cameras, cleanup
from .config import config_sha256

# --------------------------------------------------------------------------- #
# Frozen constants (D-06, D-07, D-08, D-09)
# --------------------------------------------------------------------------- #


GT_MIN_POINTS = 1_000_000

# G3-AMEND-01 (approved 2026-11-01): the GT surface is evaluated on the points
# both rectified cameras can see.  ``camera`` is 0/1 for a rectified camera and
# 255 for a return outside every camera frustum.
GT_VISIBLE_CAMERA_MAX = 2
BBOX_AXIS_NAMES = ("x", "y")
MESH_Z_MIN_MARGIN_M = 5.0
MESH_Z_MAX_MARGIN_M = 20.0

BBOX_RATIO_MIN = 0.8
BBOX_RATIO_MAX = 1.2
LARGEST_COMPONENT_MIN = 0.90
DEGENERATE_FRACTION_MAX = 0.001

G3_AMENDMENT_ID = "G3-AMEND-01"
G3_AMENDMENT_APPROVED_AT_UTC = "2026-11-01"


# Camera-frame directions; ``+z`` is the optical axis (D-07).

# Fixed binary layouts (frozen; see core/include/raylidar/{gt/cache,rays/rays}.h).
_GT_HEADER_SIZE = 64
_GT_RECORD_SIZE = 32
_GT_HEADER_DTYPE = np.dtype(
    {
        "names": ["magic", "version", "record_size", "count", "frame", "reserved"],
        "formats": ["S4", "<u2", "<u2", "<u4", "<u4", "V48"],
        "offsets": [0, 4, 6, 8, 12, 16],
        "itemsize": _GT_HEADER_SIZE,
    }
)
_GT_RECORD_DTYPE = np.dtype(
    {
        "names": ["p_L", "r_gt", "frame", "semantic_id", "camera", "valid", "pad"],
        "formats": [
            ("<f4", (3,)),
            "<f4",
            "<u4",
            "<u2",
            "u1",
            "u1",
            "V8",
        ],
        "offsets": [0, 12, 16, 20, 22, 23, 24],
        "itemsize": _GT_RECORD_SIZE,
    }
)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_mesh(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly one triangle mesh with the pure-Python reader."""
    from .meshio import MeshFormatError, read_mesh

    path = Path(mesh_path)
    if not path.is_file():
        raise G3Error(f"mesh is missing: {path}")
    try:
        vertices, faces = read_mesh(path)
    except MeshFormatError as exc:
        raise G3Error(str(exc)) from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise G3Error(f"mesh has no usable vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise G3Error(f"mesh has no triangles: {path}")
    if not np.isfinite(vertices).all():
        raise G3Error(f"mesh has non-finite vertices: {path}")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise G3Error(f"mesh has an out-of-range face index: {path}")
    return vertices, faces


def _read_gt_frame(
    path: Path, expected_frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode one ``RLGT`` frame with the frozen header/record validation.

    Returns ``(p_L, camera, valid, count)``.  ``camera`` is the per-record
    camera id (0/1 for a rectified camera, 255 when no camera sees the point);
    ``valid`` is the frozen static-validity flag.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < _GT_HEADER_SIZE:
        raise G3Error(f"{path}: truncated RLGT header")
    header = raw[:_GT_HEADER_SIZE].view(_GT_HEADER_DTYPE)[0]
    if bytes(header["magic"]) != b"RLGT":
        raise G3Error(f"{path}: bad RLGT magic")
    if int(header["version"]) != 1:
        raise G3Error(f"{path}: unsupported RLGT version {int(header['version'])}")
    if int(header["record_size"]) != _GT_RECORD_SIZE:
        raise G3Error(f"{path}: RLGT record size is not 32")
    if int(header["frame"]) != int(expected_frame):
        raise G3Error(
            f"{path}: header frame {int(header['frame'])} != {int(expected_frame)}"
        )
    count = int(header["count"])
    if count <= 0:
        raise G3Error(f"{path}: RLGT frame has no points")
    expected_bytes = _GT_HEADER_SIZE + count * _GT_RECORD_SIZE
    if raw.size != expected_bytes:
        raise G3Error(
            f"{path}: length {raw.size} != 64 + {count} * 32 = {expected_bytes}"
        )
    records = raw[_GT_HEADER_SIZE:].view(_GT_RECORD_DTYPE)
    if not np.all(records["frame"] == np.uint32(expected_frame)):
        raise G3Error(f"{path}: a record frame disagrees with the header frame")
    points = np.asarray(records["p_L"], dtype=np.float64)
    if not np.isfinite(points).all():
        raise G3Error(f"{path}: RLGT has non-finite coordinates")
    camera = np.asarray(records["camera"], dtype=np.int64)
    valid = np.asarray(records["valid"], dtype=np.int64)
    return points, camera, valid, count


def _nearest_rank(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = int(math.ceil(percentile * ordered.size)) - 1
    index = min(max(index, 0), ordered.size - 1)
    return float(ordered[index])


# --------------------------------------------------------------------------- #
# Metric producers
# --------------------------------------------------------------------------- #


def gt_extent(
    bundle: cameras.Bundle,
    frames: Iterable[int],
    *,
    phase2_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the W-frame AABB of the camera-visible static GT points.

    G3-AMEND-01: only records with ``camera < GT_VISIBLE_CAMERA_MAX`` and
    ``valid == 1`` contribute.  ``p_W = T_W_V(frame) @ p_L`` for every visible
    point in every requested frame.  The returned ``total`` is the visible
    count.  The result is used only to measure the mesh; no construction module
    calls this.
    """
    requested = [int(frame) for frame in frames]
    if not requested:
        raise G3Error("gt_extent requires at least one frame")
    root = Path(phase2_dir) / "gt"

    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    total = 0
    per_frame: dict[int, int] = {}
    for frame in requested:
        path = root / f"frame_{frame:010d}.bin"
        if not path.is_file():
            raise G3Error(f"missing frozen GT frame: {path}")
        p_L, camera, valid, _count = _read_gt_frame(path, frame)
        visible = (camera < GT_VISIBLE_CAMERA_MAX) & (valid == 1)
        visible_count = int(np.count_nonzero(visible))
        if visible_count == 0:
            raise G3Error(
                f"{path}: no camera-visible static GT points "
                f"(camera < {GT_VISIBLE_CAMERA_MAX} and valid == 1)"
            )
        T_W_V = cameras.T_W_V(bundle, frame)
        p_W = p_L[visible] @ T_W_V[:3, :3].T + T_W_V[:3, 3]
        minimum = np.minimum(minimum, p_W.min(axis=0))
        maximum = np.maximum(maximum, p_W.max(axis=0))
        per_frame[frame] = visible_count
        total += visible_count

    if total <= GT_MIN_POINTS:
        raise G3Error(
            f"GT extent needs more than {GT_MIN_POINTS} camera-visible points, "
            f"got {total}"
        )
    return minimum, maximum, total


def _bbox_ratios(
    mesh_min: Sequence[float],
    mesh_max: Sequence[float],
    gt_min: Sequence[float],
    gt_max: Sequence[float],
) -> list[float | None]:
    """Return the x/y mesh-over-GT footprint ratios (``None`` if undefined)."""
    mesh_extent = np.asarray(mesh_max, dtype=np.float64) - np.asarray(
        mesh_min, dtype=np.float64
    )
    gt_extent = np.asarray(gt_max, dtype=np.float64) - np.asarray(
        gt_min, dtype=np.float64
    )
    ratios: list[float | None] = []
    for name in BBOX_AXIS_NAMES:
        axis = "xyz".index(name)
        if not math.isfinite(float(gt_extent[axis])) or gt_extent[axis] <= 0.0:
            ratios.append(None)
        else:
            ratios.append(float(mesh_extent[axis] / gt_extent[axis]))
    return ratios


def bbox_check(
    mesh_min: Sequence[float],
    mesh_max: Sequence[float],
    gt_min: Sequence[float],
    gt_max: Sequence[float],
) -> dict[str, Any]:
    """Measure the x/y footprint ratio and the z band (no pass/fail verdict)."""
    ratios = _bbox_ratios(mesh_min, mesh_max, gt_min, gt_max)
    return {
        "measured": {
            "ratio": ratios,
            "axes": list(BBOX_AXIS_NAMES),
            "height_band": {
                "mesh_z_min_m": float(mesh_min[2]),
                "mesh_z_max_m": float(mesh_max[2]),
                "gt_z_min_m": float(gt_min[2]),
                "gt_z_max_m": float(gt_max[2]),
                "z_min_margin_m": MESH_Z_MIN_MARGIN_M,
                "z_max_margin_m": MESH_Z_MAX_MARGIN_M,
            },
        },
        "threshold": {
            "min": BBOX_RATIO_MIN,
            "max": BBOX_RATIO_MAX,
            "axes": list(BBOX_AXIS_NAMES),
            "z_min_margin_m": MESH_Z_MIN_MARGIN_M,
            "z_max_margin_m": MESH_Z_MAX_MARGIN_M,
        },
    }


def components_check(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Measure connected-component statistics (no pass/fail verdict)."""
    return {
        "measured": {
            "component_count": int(stats.get("components_count") or 0),
            "largest_component_fraction": float(stats.get("largest_component_fraction") or 0.0),
        },
        "threshold": {"largest_component_fraction_min": LARGEST_COMPONENT_MIN},
    }


def degenerate_check(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Measure degenerate-triangle statistics (no pass/fail verdict)."""
    return {
        "measured": {
            "degenerate_count": int(stats.get("degenerate_removed_count") or 0),
            "degenerate_fraction": float(stats.get("degenerate_fraction") or 0.0),
        },
        "threshold": {"degenerate_fraction_max": DEGENERATE_FRACTION_MAX},
    }


def _tool_versions(env: Mapping[str, Any] | None) -> dict[str, Any]:
    env = env or {}
    packages = env.get("packages") or {}
    return {
        "numpy": packages.get("numpy"),
        "openmvs": (env.get("openmvs") or {}).get("version"),
        "colmap": (env.get("colmap") or {}).get("version"),
        "colmap_cuda": (env.get("colmap_cuda") or {}).get("build"),
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def evaluate(
    artifact: Mapping[str, Any],
    bundle_root: str,
    window_name: str,
    frames: Sequence[int],
    cfg: Mapping[str, Any],
    workdir: str,
    env: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate G3 on ``artifact`` and write ``<workdir>/g3_report.json``."""
    started = time.time()
    if not isinstance(artifact, Mapping) or not artifact.get("path"):
        raise G3Error("evaluate requires an artifact with a canonical mesh path")
    mesh_path = Path(str(artifact["path"]))
    if not mesh_path.is_file():
        raise G3Error(f"canonical mesh is missing: {mesh_path}")

    windows = cfg.get("windows") or {}
    full_frames = tuple(int(value) for value in windows.get("full") or ())
    if not full_frames:
        raise G3Error("config.windows.full is required")

    bundle = cameras.load_bundle(bundle_root)
    phase2_dir = cfg.get("phase2_dir")
    if not phase2_dir:
        raise G3Error("config.phase2_dir is required for G3 diagnostics")

    gt_min, gt_max, gt_points = gt_extent(
        bundle, full_frames, phase2_dir=phase2_dir
    )

    # Independent recomputation from the canonical PLY (D-08), never from the
    # manifest's claimed counts.
    vertices, faces = _load_mesh(mesh_path)
    cleanup_params = (cfg.get("params") or {}).get("cleanup") or {}
    cleaned = cleanup.clean_mesh(
        vertices,
        faces,
        min_component_triangles=int(
            cleanup_params.get(
                "min_component_triangles", cleanup.DEFAULT_MIN_COMPONENT_TRIANGLES
            )
        ),
    )
    stats = cleaned["stats"]

    mesh_min = [float(value) for value in artifact.get("bbox_min_m") or []]
    mesh_max = [float(value) for value in artifact.get("bbox_max_m") or []]
    if len(mesh_min) != 3 or len(mesh_max) != 3:
        mesh_min = [float(value) for value in vertices.min(axis=0)]
        mesh_max = [float(value) for value in vertices.max(axis=0)]

    checks = {
        "bbox_ratio": bbox_check(mesh_min, mesh_max, gt_min, gt_max),
        "components": components_check(stats),
        "degenerate": degenerate_check(stats),
    }

    # Diagnostics only: keep the measured values, drop the reference pass/fail
    # booleans from the report.  The constants remain available as reference
    # values, but the reconstruction package never rejects a mesh because of them.
    measurements = {
        name: {
            "measured": check.get("measured") or {},
            "reference": check.get("threshold") or {},
        }
        for name, check in checks.items()
    }

    metrics = {
        "mesh_extent_m": [float(mesh_max[i] - mesh_min[i]) for i in range(3)],
        "gt_extent_m": [float(gt_max[i] - gt_min[i]) for i in range(3)],
        "bbox_ratio": checks["bbox_ratio"]["measured"]["ratio"],
        "components": {
            "count": int(stats["components_count"]),
            "largest_component_fraction": float(
                stats["largest_component_fraction"]
            ),
        },
        "degenerate": {
            "count": int(stats["degenerate_removed_count"]),
            "fraction": float(stats["degenerate_fraction"]),
        },
    }

    mesh_sha256 = artifact.get("sha256") or _sha256_file(mesh_path)
    report = {
        "schema_version": 2,
        "status": "measured",
        "mode": "diagnostics",
        "window": window_name,
        "frames": [int(frame) for frame in frames],
        "gt_window": "full",
        "gt_frames": [int(frame) for frame in full_frames],
        "artifact_root": str(Path(str(phase2_dir))),
        "bundle_root": str(bundle_root),
        "mesh": {
            "path": str(mesh_path),
            "sha256": str(mesh_sha256),
            "vertex_count": int(artifact.get("vertex_count") or vertices.shape[0]),
            "triangle_count": int(artifact.get("triangle_count") or faces.shape[0]),
            "bbox_min_m": mesh_min,
            "bbox_max_m": mesh_max,
        },
        "gt": {
            "camera_visible": True,
            "camera_visible_max": GT_VISIBLE_CAMERA_MAX,
            "point_count": int(gt_points),
            "min_m": [float(value) for value in gt_min],
            "max_m": [float(value) for value in gt_max],
        },
        "amendment": {
            "id": G3_AMENDMENT_ID,
            "approved_at_utc": G3_AMENDMENT_APPROVED_AT_UTC,
            "camera_visible_gt": True,
        },
        "config_sha256": str(
            artifact.get("config_sha256") or config_sha256(dict(cfg))
        ),
        "gate_config_sha256": config_sha256(dict(cfg)),
        "alignment_applied": False,
        "evaluation_frames_used_for_tuning": False,
        "reference_thresholds": {
            "bbox_ratio_min": BBOX_RATIO_MIN,
            "bbox_ratio_max": BBOX_RATIO_MAX,
            "bbox_axes": list(BBOX_AXIS_NAMES),
            "mesh_z_min_margin_m": MESH_Z_MIN_MARGIN_M,
            "mesh_z_max_margin_m": MESH_Z_MAX_MARGIN_M,
            "largest_component_fraction_min": LARGEST_COMPONENT_MIN,
            "degenerate_fraction_max": DEGENERATE_FRACTION_MAX,
            "gt_visible_camera_max": GT_VISIBLE_CAMERA_MAX,
        },
        "measurements": measurements,
        "metrics": metrics,
        "tool_versions": _tool_versions(env),
        "wall_time_s": round(time.time() - started, 3),
    }

    report_path = Path(workdir) / "g3_report.json"
    _write_json(report_path, report)
    return {
        "status": "measured",
        "report_path": str(report_path),
        "measurements": measurements,
        "metrics": metrics,
        "g3_report": report,
    }


__all__ = [
    "BBOX_AXIS_NAMES",
    "BBOX_RATIO_MAX",
    "BBOX_RATIO_MIN",
    "DEGENERATE_FRACTION_MAX",
    "G3_AMENDMENT_APPROVED_AT_UTC",
    "G3_AMENDMENT_ID",
    "G3Error",
    "GT_MIN_POINTS",
    "GT_VISIBLE_CAMERA_MAX",
    "LARGEST_COMPONENT_MIN",
    "MESH_Z_MAX_MARGIN_M",
    "MESH_Z_MIN_MARGIN_M",
    "bbox_check",
    "components_check",
    "degenerate_check",
    "evaluate",
    "gt_extent",
]
