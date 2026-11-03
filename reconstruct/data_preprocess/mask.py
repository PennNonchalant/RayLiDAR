"""Per-point and per-pixel dynamic/ego masks.

Bit ABI (frozen):
  bit0 dynamic-2D, bit1 dynamic-bbox, bit2 dynamic-union, bit3 ego ROI,
  bit4 kept-static, bit5 invalid/non-finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bbox import DynamicBBox, applicable_boxes, point_in_bbox
from .config import MaskOptions
from .io_utils import PreprocessError, ensure_dir, sha256_file, utc_now, write_json
from .projection import project_points
from .semantics import SemanticImage, is_dynamic_label, label_at

DYNAMIC_2D = 1 << 0
DYNAMIC_BBOX = 1 << 1
DYNAMIC_UNION = 1 << 2
EGO_ROI = 1 << 3
KEPT_STATIC = 1 << 4
INVALID_NONFINITE = 1 << 5


def is_kept_static(bits: int) -> bool:
    return bool(int(bits) & KEPT_STATIC)


def is_dynamic(bits: int) -> bool:
    return bool(int(bits) & DYNAMIC_UNION)


@dataclass
class MaskResult:
    bits: np.ndarray
    image_00_pixels: np.ndarray | None
    image_01_pixels: np.ndarray | None
    image_sizes: dict[str, tuple[int, int]]
    counts: dict[str, int]
    residual: dict[str, float | int]


def _in_ego_roi(point: np.ndarray, options: MaskOptions) -> bool:
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    return (
        options.ego_roi_x[0] < x < options.ego_roi_x[1]
        and options.ego_roi_y[0] < y < options.ego_roi_y[1]
        and options.ego_roi_z[0] < z < options.ego_roi_z[1]
    )


def _semantic_dynamic_mask(
    uncurled: np.ndarray,
    transforms: dict[str, np.ndarray],
    intrinsics: dict[str, np.ndarray],
    semantics: dict[str, SemanticImage],
    image_sizes: dict[str, tuple[int, int]],
) -> np.ndarray:
    flag = np.zeros(uncurled.shape[0], dtype=bool)
    for camera in transforms:
        semantics_image = semantics.get(camera)
        size = image_sizes.get(camera)
        internals = intrinsics.get(camera)
        if semantics_image is None or size is None or internals is None:
            continue
        projection = project_points(
            uncurled,
            transforms[camera],
            internals,
            size[0],
            size[1],
            min_range_m=0.0,
            max_range_m=1.0e9,
            margin_px=0,
            clamp_range=False,
        )
        usable = (
            projection.valid
            & (projection.z > 0.0)
            & np.isfinite(projection.uv).all(axis=1)
        )
        for index in np.flatnonzero(usable):
            u, v = projection.uv[index]
            if u < 0.0 or u >= size[0] or v < 0.0 or v >= size[1]:
                continue
            if is_dynamic_label(label_at(semantics_image, float(u), float(v))):
                flag[index] = True
    return flag


def _rasterize_boxes(
    box: DynamicBBox,
    transform: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    pixels: np.ndarray,
) -> None:
    fx, cx = float(intrinsics[0, 0]), float(intrinsics[0, 2])
    fy, cy = float(intrinsics[1, 1]), float(intrinsics[1, 2])
    us: list[float] = []
    vs: list[float] = []
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            for z_sign in (-1, 1):
                corner = np.array([0.5 * x_sign, 0.5 * y_sign, 0.5 * z_sign, 1.0])
                world = (box.T_W_B @ corner)[:3]
                camera = (np.asarray(transform) @ np.append(world, 1.0))[:3]
                if not (np.isfinite(camera).all() and camera[2] > 0.0):
                    continue
                u = fx * camera[0] / camera[2] + cx
                v = fy * camera[1] / camera[2] + cy
                if np.isfinite(u) and np.isfinite(v):
                    us.append(u)
                    vs.append(v)
    if not us:
        return
    x0 = int(np.clip(np.floor(min(us)), 0, width - 1))
    x1 = int(np.clip(np.ceil(max(us)), 0, width - 1))
    y0 = int(np.clip(np.floor(min(vs)), 0, height - 1))
    y1 = int(np.clip(np.ceil(max(vs)), 0, height - 1))
    if x0 <= x1 and y0 <= y1:
        pixels[y0 : y1 + 1, x0 : x1 + 1] = 1


def build_mask(
    raw_points: np.ndarray,
    uncurled_points: np.ndarray,
    frame: int,
    T_W_V: np.ndarray,
    boxes: list[DynamicBBox],
    semantics: dict[str, SemanticImage],
    image_sizes: dict[str, tuple[int, int]],
    transforms: dict[str, np.ndarray],
    intrinsics: dict[str, np.ndarray],
    options: MaskOptions,
) -> MaskResult:
    raw = np.asarray(raw_points, dtype=np.float64)[:, :3]
    uncurled = np.asarray(uncurled_points, dtype=np.float64)[:, :3]
    if raw.shape != uncurled.shape:
        raise PreprocessError("raw and uncurled point arrays must be index-aligned")

    count = raw.shape[0]
    bits = np.zeros(count, dtype=np.uint8)
    finite = np.isfinite(raw).all(axis=1) & np.isfinite(uncurled).all(axis=1)
    bits[~finite] = INVALID_NONFINITE

    ego = np.array([_in_ego_roi(point, options) for point in raw], dtype=bool)
    bits[ego] |= EGO_ROI

    dynamic_2d = _semantic_dynamic_mask(uncurled, transforms, intrinsics, semantics, image_sizes)
    dynamic_2d &= finite

    applicable = applicable_boxes(boxes, frame, options.bbox_max_time_gap_frames)
    world = (np.concatenate([uncurled, np.ones((count, 1))], axis=1) @ np.asarray(T_W_V).T)[:, :3]
    dynamic_bbox = np.zeros(count, dtype=bool)
    if applicable:
        for index in np.flatnonzero(finite):
            if any(point_in_bbox(world[index], box) for box in applicable):
                dynamic_bbox[index] = True

    dynamic = dynamic_2d | dynamic_bbox
    kept = finite & ~ego & ~dynamic
    bits[dynamic_2d] |= DYNAMIC_2D
    bits[dynamic_bbox] |= DYNAMIC_BBOX
    bits[dynamic] |= DYNAMIC_UNION
    bits[kept] |= KEPT_STATIC

    image_pixels: dict[str, np.ndarray] = {}
    for camera, size in image_sizes.items():
        semantic_image = semantics.get(camera)
        if semantic_image is None:
            continue
        pixels = np.zeros((size[1], size[0]), dtype=np.uint8)
        labels = semantic_image.labels[: size[1], : size[0]]
        dynamic_pixels = np.isin(labels, [24, 25, 26, 27, 28, 32, 33])
        pixels[dynamic_pixels] = 1
        transform = transforms.get(camera)
        K = intrinsics.get(camera)
        if transform is not None and K is not None:
            for box in applicable:
                _rasterize_boxes(box, transform, K, size[0], size[1], pixels)
        image_pixels[camera] = pixels

    residual = _residual_dynamic(world, kept, dynamic, applicable, options.residual_dynamic_radius_m)

    counts = {
        "nonfinite": int((~finite).sum()),
        "ego_roi": int(ego.sum()),
        "dynamic_2d": int(dynamic_2d.sum()),
        "dynamic_bbox": int(dynamic_bbox.sum()),
        "dynamic_union": int(dynamic.sum()),
        "kept_static": int(kept.sum()),
        "overlap_2d_only": int((dynamic_2d & ~dynamic_bbox).sum()),
        "overlap_bbox_only": int((dynamic_bbox & ~dynamic_2d).sum()),
        "overlap_both": int((dynamic_2d & dynamic_bbox).sum()),
    }
    return MaskResult(
        bits=bits,
        image_00_pixels=image_pixels.get("image_00"),
        image_01_pixels=image_pixels.get("image_01"),
        image_sizes=dict(image_sizes),
        counts=counts,
        residual=residual,
    )


def _residual_dynamic(
    world: np.ndarray,
    kept: np.ndarray,
    dynamic: np.ndarray,
    applicable: list[DynamicBBox],
    radius: float,
) -> dict[str, float | int]:
    kept_indices = np.flatnonzero(kept)
    if kept_indices.size == 0:
        return {"kept_static": 0, "inside_bbox": 0, "within_radius": 0, "either": 0, "fraction": 0.0, "radius_m": radius}
    dynamic_points = world[dynamic]
    inside = 0
    near = 0
    either = 0
    radius_sq = radius * radius
    for index in kept_indices:
        point = world[index]
        inside_flag = any(point_in_bbox(point, box) for box in applicable)
        near_flag = False
        if dynamic_points.size:
            delta = dynamic_points - point
            near_flag = bool(np.any(np.einsum("ij,ij->i", delta, delta) <= radius_sq))
        inside += int(inside_flag)
        near += int(near_flag)
        either += int(inside_flag or near_flag)
    return {
        "kept_static": int(kept_indices.size),
        "inside_bbox": int(inside),
        "within_radius": int(near),
        "either": int(either),
        "fraction": float(either) / float(kept_indices.size),
        "radius_m": float(radius),
    }


def write_mask_artifact(
    masks_dir: str | Path,
    frame: int,
    result: MaskResult,
    raw_points: np.ndarray,
    sequence: str,
) -> dict:
    frame_dir = ensure_dir(Path(masks_dir) / f"frame_{frame:010d}")
    points_path = frame_dir / "points.bin"
    result.bits.tofile(points_path)

    files: dict[str, str] = {"points_sha256": sha256_file(points_path)}
    for camera in ("image_00", "image_01"):
        pixels = result.image_00_pixels if camera == "image_00" else result.image_01_pixels
        size = result.image_sizes.get(camera)
        if pixels is None or size is None:
            raise PreprocessError(f"mask result is missing per-camera pixels for {camera}")
        path = frame_dir / f"{camera}_dynamic.png"
        from PIL import Image

        Image.fromarray((pixels * 255).astype(np.uint8), mode="L").save(path)
        files[f"{camera}_dynamic_sha256"] = sha256_file(path)

    document = {
        "schema_version": 1,
        "frame": int(frame),
        "sequence": sequence,
        "generated_at_utc": utc_now(),
        "point_count": int(result.bits.size),
        "counts": result.counts,
        "residual_dynamic": result.residual,
        "applies_to": ["images", "gt", "projections", "evaluation"],
        "files": files,
    }
    write_json(frame_dir / "mask.json", document)
    return document


def build_mask_manifest(
    masks_dir: str | Path,
    frames: list[dict],
    sequence: str,
    config_sha256: str,
    filename: str = "mask_manifest.json",
) -> dict:
    keys = (
        "nonfinite",
        "ego_roi",
        "dynamic_2d",
        "dynamic_bbox",
        "dynamic_union",
        "kept_static",
        "overlap_2d_only",
        "overlap_bbox_only",
        "overlap_both",
    )
    totals = {key: 0 for key in keys}
    for frame in frames:
        for key in keys:
            totals[key] += int(frame["counts"].get(key, 0))
    document = {
        "schema_version": 1,
        "sequence": sequence,
        "generated_at_utc": utc_now(),
        "applies_to": ["images", "gt", "projections", "evaluation"],
        "frames": frames,
        "totals": totals,
        "config_sha256": config_sha256,
    }
    write_json(Path(masks_dir) / filename, document)
    return document
