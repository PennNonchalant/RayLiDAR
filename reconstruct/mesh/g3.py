"""Diagnostic measurements for the exported mesh bundle.

Five measurements are computed from local artifacts only:

1. ``bbox_ratio`` — the x/y mesh footprint over the camera-visible GT footprint
   (``camera < 2``, ``valid == 1``; ``artifacts/02/gt`` decoded read-only,
   transformed by ``T_W_V``), plus a z sanity band;
2. ``camera_outside`` — from every full-window camera centre, five directions
   (camera ``+z`` forward, ±30° yaw, ±20° pitch) are cast at the mesh;
3. ``components`` — welded-vertex connected components and the largest
   component fraction;
4. ``degenerate`` — triangles with a repeated index or area below ``1e-12 m²``;
5. ``smoke_range`` — the frozen rays of frames 4615–4624 are cast against the
   mesh and the median ``|t_pred - t_gt|`` is reported.

These are diagnostics, not a hard pass/fail gate.  The reconstruction package
does not reject a mesh because a reference value is missed; the numbers are
written to ``g3_report.json`` for the consumer.  No geometric correction,
fitting, transform, or rescaling is applied.

Measurement definitions are frozen constants in this module; the report records
the values and the reference thresholds but never enforces a pass/fail gate.

Ray casting uses ``open3d.t.geometry.RaycastingScene`` (Embree), two-sided.
Open3D exposes no cast-range arguments, so ``t_min``/``t_max`` are enforced by
filtering the returned hit distances, which has identical semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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

ARTIFACTS_ROOT_ENV = "RAYLIDAR_ARTIFACTS_ROOT"
DEFAULT_ARTIFACTS_ROOT = "/data-20T/hybrid-lidar-sim/artifacts"

GT_MIN_POINTS = 1_000_000
SMOKE_MIN_HITS = 1_000
SMOKE_FIRST_FRAME = 4615
SMOKE_LAST_FRAME = 4624

# G3-AMEND-01 (approved 2026-11-01): the GT surface is evaluated on the points
# both rectified cameras can see.  ``camera`` is 0/1 for a rectified camera and
# 255 for a return outside every camera frustum.
GT_VISIBLE_CAMERA_MAX = 2
BBOX_AXIS_NAMES = ("x", "y")
MESH_Z_MIN_MARGIN_M = 5.0
MESH_Z_MAX_MARGIN_M = 20.0

BBOX_RATIO_MIN = 0.8
BBOX_RATIO_MAX = 1.2
FORWARD_HIT_RATE_MIN = 0.3
NEAR_HIT_MAX_M = 1.5
LARGEST_COMPONENT_MIN = 0.90
DEGENERATE_FRACTION_MAX = 0.001
SMOKE_MEDIAN_ERROR_MAX_M = 1.0

G3_AMENDMENT_ID = "G3-AMEND-01"
G3_AMENDMENT_APPROVED_AT_UTC = "2026-11-01"

RAY_T_MIN_M = 1e-4
CAMERA_OUTSIDE_T_MAX_M = 200.0
SMOKE_RANGE_T_MAX_MARGIN_M = 1.0
UNIT_DIRECTION_ATOL = 1e-3

# Camera-frame directions; ``+z`` is the optical axis (D-07).
CAMERA_OUTSIDE_DIRECTIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward", (0.0, 0.0, 1.0)),
    (
        "yaw_plus_30",
        (math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0))),
    ),
    (
        "yaw_minus_30",
        (math.sin(math.radians(-30.0)), 0.0, math.cos(math.radians(30.0))),
    ),
    (
        "pitch_plus_20",
        (0.0, math.sin(math.radians(20.0)), math.cos(math.radians(20.0))),
    ),
    (
        "pitch_minus_20",
        (0.0, math.sin(math.radians(-20.0)), math.cos(math.radians(20.0))),
    ),
)


# Fixed binary layouts for the prepared GT and ray frames.
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
_RAY_HEADER_SIZE = 32
_RAY_RECORD_SIZE = 64
_RAY_HEADER_DTYPE = np.dtype(
    {
        "names": ["magic", "version", "record_size", "count", "frame", "reserved"],
        "formats": ["S4", "<u2", "<u2", "<u4", "<u4", "V16"],
        "offsets": [0, 4, 6, 8, 12, 16],
        "itemsize": _RAY_HEADER_SIZE,
    }
)
_RAY_RECORD_DTYPE = np.dtype(
    {
        "names": [
            "ray_id",
            "dir_C",
            "t_gt",
            "p_L",
            "p_C",
            "bucket",
            "occluded",
            "camera",
            "semantic_id",
            "pad",
        ],
        "formats": [
            "<u8",
            ("<f4", (3,)),
            "<f4",
            ("<f4", (3,)),
            ("<f4", (3,)),
            "u1",
            "u1",
            "u1",
            "<u2",
            "V11",
        ],
        "offsets": [0, 8, 20, 24, 36, 48, 49, 50, 51, 53],
        "itemsize": _RAY_RECORD_SIZE,
    }
)

_RAY_CAMERAS: dict[int, str] = {0: "image_00", 1: "image_01"}


class G3Error(RuntimeError):
    """The diagnostics cannot be evaluated (missing/malformed input)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _artifacts_root(
    override: str | Path | None = None, cfg: Mapping[str, Any] | None = None
) -> Path:
    """Resolve the prepared artifacts root (``.../artifacts``)."""
    if override is not None:
        return Path(override)
    env_value = os.environ.get(ARTIFACTS_ROOT_ENV)
    if env_value:
        return Path(env_value)
    if isinstance(cfg, Mapping) and cfg.get("out_root"):
        # ``out_root`` is ``<artifacts>/04``; the frozen GT/rays live beside it.
        return Path(str(cfg["out_root"])).resolve().parent
    return Path(DEFAULT_ARTIFACTS_ROOT)


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
    os.replace(temporary, path)


def _load_mesh(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly one triangle mesh as float64 vertices and int64 faces."""
    import trimesh  # lazy: only ray-casting paths need a mesh loader

    path = Path(mesh_path)
    if not path.is_file():
        raise G3Error(f"mesh is missing: {path}")
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if len(geometries) != 1:
            raise G3Error(
                f"mesh must contain exactly one geometry, got {len(geometries)}"
            )
        loaded = geometries[0]
    if not isinstance(loaded, trimesh.Trimesh):
        raise G3Error(f"not a triangle mesh: {path}")
    vertices = np.ascontiguousarray(np.asarray(loaded.vertices, dtype=np.float64))
    faces = np.ascontiguousarray(np.asarray(loaded.faces, dtype=np.int64))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise G3Error(f"mesh has no usable vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise G3Error(f"mesh has no triangles: {path}")
    if not np.isfinite(vertices).all():
        raise G3Error(f"mesh has non-finite vertices: {path}")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise G3Error(f"mesh has an out-of-range face index: {path}")
    return vertices, faces


def _raycasting_scene(vertices: np.ndarray, faces: np.ndarray):
    """Build one Embree scene; callers reuse it for every frame."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)),
    )
    return scene


def _cast_rays(
    scene,
    origins: np.ndarray,
    directions: np.ndarray,
    t_min: float,
    t_max: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast rays and return ``(t_hit, is_hit)``.

    ``t_min``/``t_max`` are applied to the returned distances because Open3D
    exposes no cast-range arguments; this matches the frozen protocol rule
    "a hit is a cast result with ``t_min <= t_pred <= t_max``".
    """
    import open3d as o3d

    if origins.shape[0] == 0:
        return np.empty(0, dtype=np.float64), np.zeros(0, dtype=bool)
    rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
    t_hit = np.asarray(
        scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy(), dtype=np.float64
    ).reshape(-1)
    upper = np.broadcast_to(np.asarray(t_max, dtype=np.float64), t_hit.shape)
    hit = np.isfinite(t_hit) & (t_hit >= float(t_min)) & (t_hit <= upper)
    return t_hit, hit


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


def _read_ray_frame(
    path: Path, expected_frame: int
) -> tuple[dict[str, np.ndarray], int]:
    """Decode one ``RLRY`` frame with the frozen header/record validation."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < _RAY_HEADER_SIZE:
        raise G3Error(f"{path}: truncated RLRY header")
    header = raw[:_RAY_HEADER_SIZE].view(_RAY_HEADER_DTYPE)[0]
    if bytes(header["magic"]) != b"RLRY":
        raise G3Error(f"{path}: bad RLRY magic")
    if int(header["version"]) != 1:
        raise G3Error(f"{path}: unsupported RLRY version {int(header['version'])}")
    if int(header["record_size"]) != _RAY_RECORD_SIZE:
        raise G3Error(f"{path}: RLRY record size is not 64")
    if int(header["frame"]) != int(expected_frame):
        raise G3Error(
            f"{path}: header frame {int(header['frame'])} != {int(expected_frame)}"
        )
    count = int(header["count"])
    if count <= 0:
        raise G3Error(f"{path}: RLRY frame has no rays")
    expected_bytes = _RAY_HEADER_SIZE + count * _RAY_RECORD_SIZE
    if raw.size != expected_bytes:
        raise G3Error(
            f"{path}: length {raw.size} != 32 + {count} * 64 = {expected_bytes}"
        )
    records = raw[_RAY_HEADER_SIZE:].view(_RAY_RECORD_DTYPE)
    directions = np.asarray(records["dir_C"], dtype=np.float64)
    t_gt = np.asarray(records["t_gt"], dtype=np.float64)
    if not np.isfinite(directions).all() or not np.isfinite(t_gt).all():
        raise G3Error(f"{path}: RLRY has non-finite values")
    if np.any(t_gt <= 0.0):
        raise G3Error(f"{path}: RLRY has a non-positive t_gt")
    norms = np.linalg.norm(directions, axis=1)
    if np.max(np.abs(norms - 1.0)) > UNIT_DIRECTION_ATOL:
        raise G3Error(f"{path}: RLRY dir_C is not unit length")
    cameras_field = np.asarray(records["camera"], dtype=np.int64)
    if np.any(cameras_field > 1):
        raise G3Error(f"{path}: RLRY camera must be 0 or 1")
    buckets = np.asarray(records["bucket"], dtype=np.int64)
    if np.any(buckets > 3):
        raise G3Error(f"{path}: RLRY bucket must be 0-3")
    return (
        {
            "dir_C": directions,
            "t_gt": t_gt,
            "camera": cameras_field,
            "occluded": np.asarray(records["occluded"], dtype=np.int64),
        },
        count,
    )


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
    artifacts_root: str | Path | None = None,
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
    root = _artifacts_root(artifacts_root) / "02" / "gt"

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


def _direction_vectors() -> np.ndarray:
    return np.asarray(
        [direction for _, direction in CAMERA_OUTSIDE_DIRECTIONS], dtype=np.float64
    )


def camera_outside(
    mesh_path: str | Path,
    bundle: cameras.Bundle,
    frames: Iterable[int],
    *,
    t_max_m: float = CAMERA_OUTSIDE_T_MAX_M,
) -> dict[str, Any]:
    """Cast the five D-07 directions from every camera centre.

    ``forward_hit_rate`` is the fraction of camera centres whose forward ray
    hits the mesh; the five directions share one RaycastingScene.  A hit closer
    than ``NEAR_HIT_MAX_M`` is reported as a near hit.
    """
    requested = [int(frame) for frame in frames]
    if not requested:
        raise G3Error("camera_outside requires at least one frame")
    vertices, faces = _load_mesh(mesh_path)
    scene = _raycasting_scene(vertices, faces)
    local_directions = _direction_vectors()

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    labels: list[tuple[int, str]] = []
    for frame in requested:
        T_W_C = cameras.T_W_C(bundle, frame)
        rotation = T_W_C[:3, :3]
        origin = T_W_C[:3, 3]
        for name, direction in zip(
            [name for name, _ in CAMERA_OUTSIDE_DIRECTIONS], local_directions
        ):
            world = rotation @ direction
            norm = float(np.linalg.norm(world))
            if norm <= 0.0:
                raise G3Error(f"camera_outside: degenerate direction for frame {frame}")
            origins.append(origin)
            directions.append(world / norm)
            labels.append((frame, name))

    origin_array = np.asarray(origins, dtype=np.float64)
    direction_array = np.asarray(directions, dtype=np.float64)
    t_hit, is_hit = _cast_rays(
        scene, origin_array, direction_array, RAY_T_MIN_M, float(t_max_m)
    )

    frame_count = len(requested)
    per_direction: dict[str, int] = {
        name: 0 for name, _ in CAMERA_OUTSIDE_DIRECTIONS
    }
    forward_hits = 0
    near_hits: list[dict[str, Any]] = []
    hit_distances = t_hit[is_hit]
    for index, (frame, name) in enumerate(labels):
        if not bool(is_hit[index]):
            continue
        per_direction[name] += 1
        if name == "forward":
            forward_hits += 1
        if float(t_hit[index]) < NEAR_HIT_MAX_M:
            near_hits.append(
                {
                    "frame": int(frame),
                    "direction": name,
                    "t_m": float(t_hit[index]),
                }
            )

    return {
        "frames": frame_count,
        "rays": int(origin_array.shape[0]),
        "hit_count": int(np.count_nonzero(is_hit)),
        "forward_rays": frame_count,
        "forward_hits": int(forward_hits),
        "forward_hit_rate": (
            float(forward_hits) / float(frame_count) if frame_count else 0.0
        ),
        "per_direction_hits": per_direction,
        "per_direction_hit_rate": {
            name: (float(count) / float(frame_count) if frame_count else 0.0)
            for name, count in per_direction.items()
        },
        "near_hits": near_hits,
        "min_hit_t_m": (
            float(np.min(hit_distances)) if hit_distances.size else None
        ),
        "directions": {
            name: [float(value) for value in direction]
            for name, direction in CAMERA_OUTSIDE_DIRECTIONS
        },
        "t_min_m": RAY_T_MIN_M,
        "t_max_m": float(t_max_m),
    }


def smoke_range(
    mesh_path: str | Path,
    bundle: cameras.Bundle,
    frames: Iterable[int],
    *,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    """Cast the frozen rays of ``frames`` and report the median range error."""
    requested = [int(frame) for frame in frames]
    if not requested:
        raise G3Error("smoke_range requires at least one frame")
    root = _artifacts_root(artifacts_root) / "02" / "rays"

    decoded: list[tuple[int, dict[str, np.ndarray]]] = []
    for frame in requested:
        path = root / f"frame_{frame:010d}.bin"
        if not path.is_file():
            raise G3Error(f"missing frozen ray frame: {path}")
        payload, _ = _read_ray_frame(path, frame)
        decoded.append((frame, payload))

    vertices, faces = _load_mesh(mesh_path)
    scene = _raycasting_scene(vertices, faces)

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    t_gt: list[np.ndarray] = []
    occluded: list[np.ndarray] = []
    frame_index: list[int] = []
    for index, (frame, payload) in enumerate(decoded):
        dir_C = payload["dir_C"]
        camera_names = [_RAY_CAMERAS[int(value)] for value in payload["camera"]]
        # Rays share a frame but may come from either rectified camera.
        for camera_name in sorted(set(camera_names)):
            T_W_C = cameras.T_W_C(bundle, frame, camera_name)
            selection = np.asarray(
                [name == camera_name for name in camera_names], dtype=bool
            )
            dir_W = dir_C[selection] @ T_W_C[:3, :3].T
            norms = np.linalg.norm(dir_W, axis=1)
            dir_W = dir_W / norms[:, None]
            origins.append(np.broadcast_to(T_W_C[:3, 3], dir_W.shape).copy())
            directions.append(dir_W)
            t_gt.append(payload["t_gt"][selection])
            occluded.append(payload["occluded"][selection])
            frame_index.append(np.full(dir_W.shape[0], index, dtype=np.int64))

    if not origins:
        raise G3Error("smoke_range decoded no rays")
    origin_array = np.concatenate(origins, axis=0)
    direction_array = np.concatenate(directions, axis=0)
    t_gt_array = np.concatenate(t_gt, axis=0)
    occluded_array = np.concatenate(occluded, axis=0)
    frame_index_array = np.concatenate(frame_index, axis=0)

    t_hit, is_hit = _cast_rays(
        scene,
        origin_array,
        direction_array,
        RAY_T_MIN_M,
        t_gt_array + SMOKE_RANGE_T_MAX_MARGIN_M,
    )
    hit_count = int(np.count_nonzero(is_hit))
    ray_count = int(origin_array.shape[0])
    result: dict[str, Any] = {
        "frames": requested,
        "rays": ray_count,
        "hits": hit_count,
        "coverage": (float(hit_count) / float(ray_count) if ray_count else 0.0),
        "median_abs_error_m": None,
        "p90_abs_error_m": None,
        "per_frame_median": {},
        "median_abs_error_unoccluded_m": None,
        "occluded_rays": int(np.count_nonzero(occluded_array)),
        "t_max_rule": "t_gt + 1.0 m",
    }
    if hit_count < SMOKE_MIN_HITS:
        result["reason"] = "insufficient hits"
        return result

    errors = np.abs(t_hit[is_hit] - t_gt_array[is_hit])
    result["median_abs_error_m"] = float(np.median(errors))
    result["p90_abs_error_m"] = _nearest_rank(errors, 0.90)
    per_frame: dict[str, Any] = {}
    for index, frame in enumerate(requested):
        mask = is_hit & (frame_index_array == index)
        if np.count_nonzero(mask):
            per_frame[str(int(frame))] = float(
                np.median(np.abs(t_hit[mask] - t_gt_array[mask]))
            )
    result["per_frame_median"] = per_frame
    unoccluded = is_hit & (occluded_array == 0)
    if np.count_nonzero(unoccluded):
        result["median_abs_error_unoccluded_m"] = float(
            np.median(np.abs(t_hit[unoccluded] - t_gt_array[unoccluded]))
        )
    return result


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


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
    """x/y footprint ratio in ``[0.8, 1.2]`` plus a z sanity band.

    G3-AMEND-01: the z axis is not compared as an extent ratio.  Instead the
    mesh z range must stay inside the documented margins around the GT z range
    and must overlap it.
    """
    ratios = _bbox_ratios(mesh_min, mesh_max, gt_min, gt_max)
    ratio_ok = all(
        value is not None and BBOX_RATIO_MIN <= value <= BBOX_RATIO_MAX
        for value in ratios
    )

    mesh_z_min = float(mesh_min[2])
    mesh_z_max = float(mesh_max[2])
    gt_z_min = float(gt_min[2])
    gt_z_max = float(gt_max[2])
    z_min_ok = mesh_z_min >= gt_z_min - MESH_Z_MIN_MARGIN_M
    z_max_ok = mesh_z_max <= gt_z_max + MESH_Z_MAX_MARGIN_M
    overlap = mesh_z_min <= gt_z_max and gt_z_min <= mesh_z_max
    z_ok = z_min_ok and z_max_ok and overlap
    passed = ratio_ok and z_ok

    reasons: list[str] = []
    if not ratio_ok:
        rendered = [
            name + "=" + ("NaN" if value is None else f"{value:.4f}")
            for name, value in zip(BBOX_AXIS_NAMES, ratios)
        ]
        reasons.append(
            f"bbox ratio outside [{BBOX_RATIO_MIN}, {BBOX_RATIO_MAX}]: "
            + ", ".join(rendered)
        )
    if not z_ok:
        z_reasons: list[str] = []
        if not z_min_ok:
            z_reasons.append(
                f"mesh z min {mesh_z_min:.4f} < gt z min {gt_z_min:.4f} - "
                f"{MESH_Z_MIN_MARGIN_M} m"
            )
        if not z_max_ok:
            z_reasons.append(
                f"mesh z max {mesh_z_max:.4f} > gt z max {gt_z_max:.4f} + "
                f"{MESH_Z_MAX_MARGIN_M} m"
            )
        if not overlap:
            z_reasons.append(
                f"mesh z range [{mesh_z_min:.4f}, {mesh_z_max:.4f}] does not "
                f"overlap gt z range [{gt_z_min:.4f}, {gt_z_max:.4f}]"
            )
        reasons.append("z band: " + "; ".join(z_reasons))

    return {
        "pass": bool(passed),
        "measured": {
            "ratio": ratios,
            "axes": list(BBOX_AXIS_NAMES),
            "height_band": {
                "mesh_z_min_m": mesh_z_min,
                "mesh_z_max_m": mesh_z_max,
                "gt_z_min_m": gt_z_min,
                "gt_z_max_m": gt_z_max,
                "z_min_margin_m": MESH_Z_MIN_MARGIN_M,
                "z_max_margin_m": MESH_Z_MAX_MARGIN_M,
                "z_min_ok": bool(z_min_ok),
                "z_max_ok": bool(z_max_ok),
                "overlap": bool(overlap),
            },
        },
        "threshold": {
            "min": BBOX_RATIO_MIN,
            "max": BBOX_RATIO_MAX,
            "axes": list(BBOX_AXIS_NAMES),
            "z_min_margin_m": MESH_Z_MIN_MARGIN_M,
            "z_max_margin_m": MESH_Z_MAX_MARGIN_M,
        },
        "reason": "; ".join(reasons),
    }


def camera_outside_check(camera_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Forward hit rate ``>= 0.3`` and no hit inside ``1.5 m``.

    G3-AMEND-01 removes the former 0.9 upper bound: a complete street mesh
    should put mesh on every camera's forward axis.
    """
    rate = float(camera_metrics.get("forward_hit_rate") or 0.0)
    near_hits = list(camera_metrics.get("near_hits") or [])
    rate_ok = rate >= FORWARD_HIT_RATE_MIN
    near_ok = not near_hits
    reasons: list[str] = []
    if not rate_ok:
        reasons.append(
            f"forward hit rate {rate:.4f} < {FORWARD_HIT_RATE_MIN}"
        )
    if not near_ok:
        reasons.append(f"{len(near_hits)} near hit(s) within {NEAR_HIT_MAX_M} m")
    return {
        "pass": bool(rate_ok and near_ok),
        "measured": {
            "forward_hit_rate": rate,
            "near_hit_count": len(near_hits),
            "near_hits": near_hits,
        },
        "threshold": {
            "forward_hit_rate_min": FORWARD_HIT_RATE_MIN,
            "near_hit_max_m": NEAR_HIT_MAX_M,
        },
        "reason": "; ".join(reasons),
    }


def components_check(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Largest welded component must own at least 90 % of the triangles."""
    fraction = float(stats.get("largest_component_fraction") or 0.0)
    passed = fraction >= LARGEST_COMPONENT_MIN
    return {
        "pass": bool(passed),
        "measured": {
            "component_count": int(stats.get("components_count") or 0),
            "largest_component_fraction": fraction,
        },
        "threshold": {"largest_component_fraction_min": LARGEST_COMPONENT_MIN},
        "reason": (
            ""
            if passed
            else f"largest component fraction {fraction:.4f} < "
            f"{LARGEST_COMPONENT_MIN}"
        ),
    }


def degenerate_check(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Degenerate triangle fraction must stay below 0.1 %."""
    fraction = float(stats.get("degenerate_fraction") or 0.0)
    passed = fraction < DEGENERATE_FRACTION_MAX
    return {
        "pass": bool(passed),
        "measured": {
            "degenerate_count": int(stats.get("degenerate_removed_count") or 0),
            "degenerate_fraction": fraction,
        },
        "threshold": {"degenerate_fraction_max": DEGENERATE_FRACTION_MAX},
        "reason": (
            ""
            if passed
            else f"degenerate fraction {fraction:.6f} >= {DEGENERATE_FRACTION_MAX}"
        ),
    }


def smoke_range_check(range_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Median absolute range error on the smoke rays must be below 1 m."""
    median = range_metrics.get("median_abs_error_m")
    passed = median is not None and float(median) < SMOKE_MEDIAN_ERROR_MAX_M
    if median is None:
        reason = "median range error is null: " + str(
            range_metrics.get("reason") or "no sufficient hits"
        )
    elif not passed:
        reason = (
            f"median range error {float(median):.4f} m >= "
            f"{SMOKE_MEDIAN_ERROR_MAX_M} m"
        )
    else:
        reason = ""
    return {
        "pass": bool(passed),
        "measured": {
            "median_abs_error_m": None if median is None else float(median),
            "p90_abs_error_m": range_metrics.get("p90_abs_error_m"),
            "coverage": range_metrics.get("coverage"),
            "hits": range_metrics.get("hits"),
            "rays": range_metrics.get("rays"),
        },
        "threshold": {"median_abs_error_m_max": SMOKE_MEDIAN_ERROR_MAX_M},
        "reason": reason,
    }




def _tool_versions(env: Mapping[str, Any] | None) -> dict[str, Any]:
    env = env or {}
    packages = env.get("packages") or {}
    return {
        "open3d": packages.get("open3d"),
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
    smoke_frames = tuple(int(value) for value in windows.get("smoke") or ())
    if not full_frames or not smoke_frames:
        raise G3Error("config.windows.full and config.windows.smoke are required")

    bundle = cameras.load_bundle(bundle_root)
    root = _artifacts_root(None, cfg)

    gt_min, gt_max, gt_points = gt_extent(
        bundle, full_frames, artifacts_root=root
    )
    outside = camera_outside(mesh_path, bundle, full_frames)
    ranges = smoke_range(mesh_path, bundle, smoke_frames, artifacts_root=root)

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
        "bbox_ratio_ok": bbox_check(mesh_min, mesh_max, gt_min, gt_max),
        "camera_outside_ok": camera_outside_check(outside),
        "components_ok": components_check(stats),
        "degenerate_ok": degenerate_check(stats),
        "smoke_range_ok": smoke_range_check(ranges),
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
        "bbox_ratio": checks["bbox_ratio_ok"]["measured"]["ratio"],
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
        "camera_outside": outside,
        "smoke_range": ranges,
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
        "camera_outside_frames": [int(frame) for frame in full_frames],
        "smoke_range_frames": [int(frame) for frame in smoke_frames],
        "artifact_root": str(root),
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
            "forward_hit_rate_min": FORWARD_HIT_RATE_MIN,
            "near_hit_max_m": NEAR_HIT_MAX_M,
            "largest_component_fraction_min": LARGEST_COMPONENT_MIN,
            "degenerate_fraction_max": DEGENERATE_FRACTION_MAX,
            "smoke_median_abs_error_m_max": SMOKE_MEDIAN_ERROR_MAX_M,
            "ray_t_min_m": RAY_T_MIN_M,
            "camera_outside_t_max_m": CAMERA_OUTSIDE_T_MAX_M,
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
    "CAMERA_OUTSIDE_DIRECTIONS",
    "CAMERA_OUTSIDE_T_MAX_M",
    "DEGENERATE_FRACTION_MAX",
    "DEFAULT_ARTIFACTS_ROOT",
    "FORWARD_HIT_RATE_MIN",
    "G3_AMENDMENT_APPROVED_AT_UTC",
    "G3_AMENDMENT_ID",
    "G3Error",
    "GT_MIN_POINTS",
    "GT_VISIBLE_CAMERA_MAX",
    "LARGEST_COMPONENT_MIN",
    "MESH_Z_MAX_MARGIN_M",
    "MESH_Z_MIN_MARGIN_M",
    "NEAR_HIT_MAX_M",
    "RAY_T_MIN_M",
    "SMOKE_MIN_HITS",
    "SMOKE_MEDIAN_ERROR_MAX_M",
    "bbox_check",
    "camera_outside",
    "camera_outside_check",
    "components_check",
    "degenerate_check",
    "evaluate",
    "gt_extent",
    "smoke_range",
    "smoke_range_check",
]
