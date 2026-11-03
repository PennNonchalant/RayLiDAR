"""Frozen camera-origin ray set (``RLRY``) and W-frame ``gt_rays.bin`` (``RLGR``).

``RLRY`` header 32 bytes: magic, version u16, record size u16, count u32,
frame u32, reserved 16. Record 64 bytes: ``ray_id u64``, ``dir_C f32[3]``,
``t_gt f32``, ``p_L f32[3]``, ``p_C f32[3]``, ``bucket u8``, ``occluded u8``,
``camera u8``, ``semantic_id u16``, pad[11].

``RLGR`` header 32 bytes: magic ``RLGR``, version u16, record size u16,
count u32, reserved 20. Record 96 bytes: ``ray_id u64``,
``origin_W f64[3]``, ``dir_W f64[3]``, ``t_gt f64``, ``t_min f64``,
``t_max f64``, ``frame u32``, ``camera u16``, pad[10].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gt_cache import GtPoint
from .io_utils import PreprocessError, canonical_sha256, ensure_dir, sha256_file, utc_now, write_json

RAY_MAGIC = b"RLRY"
RAY_VERSION = 1
RAY_RECORD_SIZE = 64
RAY_HEADER_SIZE = 32

GT_RAY_MAGIC = b"RLGR"
GT_RAY_VERSION = 1
GT_RAY_RECORD_SIZE = 96
GT_RAY_HEADER_SIZE = 32

T_MIN_M = 1.0e-4
R_MIN_M = 1.5
R_MAX_M = 60.0
IMAGE_MARGIN_PX = 5

BUCKET_NAMES = ("0_20", "20_40", "40_50", "50_60")


def bucket_of(range_m: float) -> int:
    if not np.isfinite(range_m) or range_m < 0.0 or range_m > 60.0:
        raise PreprocessError(f"range {range_m} is outside the frozen [0, 60] m domain")
    if range_m < 20.0:
        return 0
    if range_m < 40.0:
        return 1
    if range_m <= 50.0:
        return 2
    return 3


@dataclass
class Ray:
    ray_id: int
    p_L: np.ndarray
    p_C: np.ndarray
    dir_C: np.ndarray
    t_gt: float
    bucket: int
    occluded: int
    camera: int
    semantic_id: int


def write_ray_frame(path: str | Path, frame: int, rays: list[Ray]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    header = bytearray(RAY_HEADER_SIZE)
    header[0:4] = RAY_MAGIC
    header[4:6] = np.uint16(RAY_VERSION).tobytes()
    header[6:8] = np.uint16(RAY_RECORD_SIZE).tobytes()
    header[8:12] = np.uint32(len(rays)).tobytes()
    header[12:16] = np.uint32(frame).tobytes()

    body = bytearray(len(rays) * RAY_RECORD_SIZE)
    for index, ray in enumerate(rays):
        base = index * RAY_RECORD_SIZE
        body[base : base + 8] = np.uint64(ray.ray_id).tobytes()
        body[base + 8 : base + 20] = np.asarray(ray.dir_C, dtype="<f4").tobytes()
        body[base + 20 : base + 24] = np.float32(ray.t_gt).tobytes()
        body[base + 24 : base + 36] = np.asarray(ray.p_L, dtype="<f4").tobytes()
        body[base + 36 : base + 48] = np.asarray(ray.p_C, dtype="<f4").tobytes()
        body[base + 48] = ray.bucket
        body[base + 49] = ray.occluded
        body[base + 50] = ray.camera
        body[base + 51 : base + 53] = np.uint16(ray.semantic_id).tobytes()
    target.write_bytes(bytes(header) + bytes(body))
    return target


def read_ray_frame(path: str | Path) -> tuple[int, list[Ray]]:
    target = Path(path)
    raw = target.read_bytes()
    if len(raw) < RAY_HEADER_SIZE:
        raise PreprocessError(f"ray frame is shorter than its header: {target}")
    if raw[0:4] != RAY_MAGIC:
        raise PreprocessError(f"ray frame has a wrong magic: {target}")
    version = int(np.frombuffer(raw[4:6], dtype="<u2")[0])
    record_size = int(np.frombuffer(raw[6:8], dtype="<u2")[0])
    count = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    frame = int(np.frombuffer(raw[12:16], dtype="<u4")[0])
    if version != RAY_VERSION or record_size != RAY_RECORD_SIZE:
        raise PreprocessError(f"ray frame has an unsupported version or record size: {target}")
    if len(raw) != RAY_HEADER_SIZE + count * RAY_RECORD_SIZE:
        raise PreprocessError(f"ray frame length does not match its header count: {target}")
    rays: list[Ray] = []
    for index in range(count):
        base = RAY_HEADER_SIZE + index * RAY_RECORD_SIZE
        record = raw[base : base + RAY_RECORD_SIZE]
        rays.append(
            Ray(
                ray_id=int(np.frombuffer(record[0:8], dtype="<u8")[0]),
                dir_C=np.frombuffer(record[8:20], dtype="<f4").astype(np.float64),
                t_gt=float(np.frombuffer(record[20:24], dtype="<f4")[0]),
                p_L=np.frombuffer(record[24:36], dtype="<f4").astype(np.float64),
                p_C=np.frombuffer(record[36:48], dtype="<f4").astype(np.float64),
                bucket=int(record[48]),
                occluded=int(record[49]),
                camera=int(record[50]),
                semantic_id=int(np.frombuffer(record[51:53], dtype="<u2")[0]),
            )
        )
    return frame, rays


def build_rays(gt_points: list[GtPoint], frame: int, transforms: dict[str, np.ndarray]) -> list[Ray]:
    if frame < 0:
        raise PreprocessError("ray frame id must be non-negative")
    camera_names = {0: "image_00", 1: "image_01"}
    rays: list[Ray] = []
    for point in gt_points:
        if point.valid != 1 or point.camera == 255:
            continue
        camera_name = camera_names.get(int(point.camera))
        if camera_name is None:
            raise PreprocessError(f"GT point has an unknown camera id {point.camera}")
        transform = transforms.get(camera_name)
        if transform is None:
            raise PreprocessError(f"missing T_Ck_rect_V for {camera_name}")
        p_l = np.asarray(point.p_L, dtype=np.float64)
        p_c = (np.asarray(transform) @ np.append(p_l, 1.0))[:3]
        range_m = float(np.linalg.norm(p_c))
        if not np.isfinite(p_c).all() or not np.isfinite(range_m) or range_m <= 0.0:
            raise PreprocessError("GT point projects to a non-finite camera position")
        if range_m < R_MIN_M or range_m > R_MAX_M:
            raise PreprocessError("GT point camera-frame range is outside [1.5, 60] m")
        if point.r_gt < 0.0 or abs(range_m - point.r_gt) > 1.0e-3:
            raise PreprocessError("GT cache r_gt disagrees with the recomputed camera-frame range")
        rays.append(
            Ray(
                ray_id=(frame << 32) | len(rays),
                p_L=p_l,
                p_C=p_c,
                dir_C=p_c / range_m,
                t_gt=range_m,
                bucket=bucket_of(range_m),
                occluded=0,
                camera=int(point.camera),
                semantic_id=int(point.semantic_id),
            )
        )
    return rays


def compute_occlusion_flags(
    gt_points: list[GtPoint],
    rays: list[Ray],
    camera: int,
    transform: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    neighbor_px: int = 3,
    sigma_lidar_m: float = 0.02,
    sigma_multiple: float = 3.0,
    depth_threshold_m: float = 0.06,
) -> np.ndarray:
    threshold = depth_threshold_m if depth_threshold_m > 0.0 else sigma_lidar_m * sigma_multiple
    flags = np.zeros(len(rays), dtype=np.uint8)
    if not gt_points:
        return flags

    points = np.asarray([point.p_L for point in gt_points], dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera_points = (homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3]
    z_all = camera_points[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u_all = float(K[0, 0]) * camera_points[:, 0] / z_all + float(K[0, 2])
        v_all = float(K[1, 1]) * camera_points[:, 1] / z_all + float(K[1, 2])
    usable = np.isfinite(camera_points).all(axis=1) & np.isfinite(z_all) & (z_all > 0.0)
    usable &= np.isfinite(u_all) & np.isfinite(v_all)

    buckets: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for index in np.flatnonzero(usable):
        px = int(np.floor(u_all[index]))
        py = int(np.floor(v_all[index]))
        if px < 0 or px >= width or py < 0 or py >= height:
            continue
        entry = (float(z_all[index]), int(index))
        bucket = buckets.setdefault((px, py), [])
        bucket.append(entry)
        bucket.sort(key=lambda item: item[0])
        if len(bucket) > 8:
            bucket.pop()

    point_index: dict[tuple[float, ...], int] = {}
    for index, point in enumerate(gt_points):
        key = tuple(float(v) for v in point.p_L)
        point_index.setdefault(key, index)

    for ray_index, ray in enumerate(rays):
        if ray.camera != camera:
            continue
        z = float(ray.p_C[2])
        if not np.isfinite(z) or z <= 0.0:
            continue
        u = float(K[0, 0]) * float(ray.p_C[0]) / z + float(K[0, 2])
        v = float(K[1, 1]) * float(ray.p_C[1]) / z + float(K[1, 2])
        px = int(np.floor(u))
        py = int(np.floor(v))
        if not (np.isfinite(u) and np.isfinite(v)) or px < 0 or px >= width or py < 0 or py >= height:
            continue
        own = point_index.get(tuple(float(v_) for v_ in ray.p_L))
        occluded = False
        for oy in range(max(0, py - neighbor_px), min(height - 1, py + neighbor_px) + 1):
            for ox in range(max(0, px - neighbor_px), min(width - 1, px + neighbor_px) + 1):
                for depth, index in buckets.get((ox, oy), []):
                    if own is not None and index == own:
                        continue
                    if z - depth > threshold:
                        occluded = True
                        break
                if occluded:
                    break
            if occluded:
                break
        flags[ray_index] = 1 if occluded else 0
    return flags


def build_rays_manifest(
    rays_dir: str | Path,
    frame_entries: list[dict],
    window: list[int],
    sequence: str,
    config_sha256: str,
    filename: str = "rays_manifest.json",
) -> dict:
    totals = {
        "frames": len(frame_entries),
        "n_rays": 0,
        "occluded_count": 0,
        "occluded_excluded_count": 0,
        "camera_counts": {"image_00": 0, "image_01": 0},
        "bucket_counts": {name: 0 for name in BUCKET_NAMES},
    }
    for entry in frame_entries:
        totals["n_rays"] += int(entry["n_rays"])
        totals["occluded_count"] += int(entry["occluded_count"])
        totals["occluded_excluded_count"] += int(entry["occluded_excluded_count"])
        for camera, count in entry["camera_counts"].items():
            totals["camera_counts"][camera] += int(count)
        for bucket, count in entry["bucket_counts"].items():
            totals["bucket_counts"][bucket] += int(count)
    document = {
        "schema_version": 1,
        "sequence": sequence,
        "generated_at_utc": utc_now(),
        "config_sha256": config_sha256,
        "window": list(window),
        "frames": frame_entries,
        "totals": totals,
    }
    write_json(Path(rays_dir) / filename, document)
    return document


def freeze_ray_set(
    rays_dir: str | Path,
    manifest: dict,
    config_sha256: str,
    filename: str = "RAY_SET_FREEZE.json",
) -> dict:
    frame_hashes = {str(entry["frame"]): entry["sha256"]["ray"] for entry in manifest["frames"]}
    freeze = {
        "schema_version": 1,
        "code_version": "0.1.0",
        "config_sha256": config_sha256,
        "frozen_at_utc": utc_now(),
        "counts": manifest["totals"],
        "manifest_sha256": canonical_sha256(manifest),
        "per_frame_meta_sha256": frame_hashes,
        "ray_set_sha256": canonical_sha256(frame_hashes),
    }
    write_json(Path(rays_dir) / filename, freeze)
    return freeze


def write_gt_rays(path: str | Path, rays: list[dict]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    header = bytearray(GT_RAY_HEADER_SIZE)
    header[0:4] = GT_RAY_MAGIC
    header[4:6] = np.uint16(GT_RAY_VERSION).tobytes()
    header[6:8] = np.uint16(GT_RAY_RECORD_SIZE).tobytes()
    header[8:12] = np.uint32(len(rays)).tobytes()

    body = bytearray(len(rays) * GT_RAY_RECORD_SIZE)
    seen: set[int] = set()
    for index, ray in enumerate(rays):
        ray_id = int(ray["ray_id"])
        if ray_id in seen:
            raise PreprocessError(f"duplicate gt_rays ray_id {ray_id}")
        seen.add(ray_id)
        direction = np.asarray(ray["dir_W"], dtype=np.float64)
        if abs(np.linalg.norm(direction) - 1.0) > 1.0e-6:
            raise PreprocessError(f"gt_rays ray {ray_id} is not unit length")
        base = index * GT_RAY_RECORD_SIZE
        body[base : base + 8] = np.uint64(ray_id).tobytes()
        body[base + 8 : base + 32] = np.asarray(ray["origin_W"], dtype="<f8").tobytes()
        body[base + 32 : base + 56] = direction.astype("<f8").tobytes()
        body[base + 56 : base + 64] = np.float64(ray["t_gt"]).tobytes()
        body[base + 64 : base + 72] = np.float64(ray["t_min"]).tobytes()
        body[base + 72 : base + 80] = np.float64(ray["t_max"]).tobytes()
        body[base + 80 : base + 84] = np.uint32(ray["frame"]).tobytes()
        body[base + 84 : base + 86] = np.uint16(ray["camera"]).tobytes()
    target.write_bytes(bytes(header) + bytes(body))
    return target


def read_gt_rays(path: str | Path) -> list[dict]:
    target = Path(path)
    raw = target.read_bytes()
    if len(raw) < GT_RAY_HEADER_SIZE:
        raise PreprocessError(f"gt_rays.bin is shorter than its header: {target}")
    if raw[0:4] != GT_RAY_MAGIC:
        raise PreprocessError(f"gt_rays.bin has a wrong magic: {target}")
    version = int(np.frombuffer(raw[4:6], dtype="<u2")[0])
    record_size = int(np.frombuffer(raw[6:8], dtype="<u2")[0])
    count = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    if version != GT_RAY_VERSION or record_size != GT_RAY_RECORD_SIZE:
        raise PreprocessError("gt_rays.bin has an unsupported version or record size")
    if len(raw) != GT_RAY_HEADER_SIZE + count * GT_RAY_RECORD_SIZE:
        raise PreprocessError("gt_rays.bin length does not match its header count")
    result: list[dict] = []
    for index in range(count):
        base = GT_RAY_HEADER_SIZE + index * GT_RAY_RECORD_SIZE
        record = raw[base : base + GT_RAY_RECORD_SIZE]
        result.append(
            {
                "ray_id": int(np.frombuffer(record[0:8], dtype="<u8")[0]),
                "origin_W": np.frombuffer(record[8:32], dtype="<f8").tolist(),
                "dir_W": np.frombuffer(record[32:56], dtype="<f8").tolist(),
                "t_gt": float(np.frombuffer(record[56:64], dtype="<f8")[0]),
                "t_min": float(np.frombuffer(record[64:72], dtype="<f8")[0]),
                "t_max": float(np.frombuffer(record[72:80], dtype="<f8")[0]),
                "frame": int(np.frombuffer(record[80:84], dtype="<u4")[0]),
                "camera": int(np.frombuffer(record[84:86], dtype="<u2")[0]),
            }
        )
    return result
