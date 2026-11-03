"""Pinhole projection and deterministic primary-camera assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Projection:
    uv: np.ndarray
    z: np.ndarray
    range_m: np.ndarray
    valid: np.ndarray


def project_points(
    points_v: np.ndarray,
    T_Ck_rect_V: np.ndarray,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    min_range_m: float = 0.0,
    max_range_m: float = float("inf"),
    margin_px: int = 0,
    clamp_range: bool = False,
) -> Projection:
    points = np.asarray(points_v, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    homogeneous = np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1)
    camera = (homogeneous @ np.asarray(T_Ck_rect_V, dtype=np.float64).T)[:, :3]
    z = camera[:, 2]
    range_m = np.linalg.norm(camera, axis=1)

    finite = np.isfinite(camera).all(axis=1) & np.isfinite(z) & np.isfinite(range_m) & (z > 0.0)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = float(K[0, 0]) * camera[:, 0] / z + float(K[0, 2])
        v = float(K[1, 1]) * camera[:, 1] / z + float(K[1, 2])
    uv[finite] = np.stack([u[finite], v[finite]], axis=1)

    margin = float(max(margin_px, 0))
    max_u = float(image_width - 1) - margin
    max_v = float(image_height - 1) - margin
    in_bounds = finite & (u >= margin) & (u <= max_u) & (v >= margin) & (v <= max_v)
    in_range = ~clamp_range | ((range_m >= min_range_m) & (range_m <= max_range_m))
    return Projection(uv=uv, z=z, range_m=range_m, valid=finite & in_bounds & in_range)


@dataclass(frozen=True)
class CameraEligibility:
    z_positive: bool
    in_image_margin: bool
    range_ok: bool
    u: float
    v: float
    z: float
    range_m: float

    @property
    def valid(self) -> bool:
        return self.z_positive and self.in_image_margin and self.range_ok


def evaluate_camera_eligibility(
    point_v: np.ndarray,
    T_Ck_rect_V: np.ndarray,
    K: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    r_min_m: float = 1.5,
    r_max_m: float = 60.0,
    margin_px: int = 5,
) -> CameraEligibility:
    projection = project_points(
        np.asarray(point_v, dtype=np.float64).reshape(1, 3),
        T_Ck_rect_V,
        K,
        image_width,
        image_height,
        min_range_m=0.0,
        max_range_m=1.0e9,
        margin_px=0,
        clamp_range=False,
    )
    z = float(projection.z[0])
    range_m = float(projection.range_m[0])
    u, v = (float(projection.uv[0, 0]), float(projection.uv[0, 1])) if np.isfinite(projection.uv[0]).all() else (
        float("nan"),
        float("nan"),
    )
    margin = float(max(margin_px, 0))
    max_u = float(image_width - 1) - margin
    max_v = float(image_height - 1) - margin
    return CameraEligibility(
        z_positive=np.isfinite(z) and z > 0.0,
        in_image_margin=bool(np.isfinite(u) and np.isfinite(v)) and margin <= u <= max_u and margin <= v <= max_v,
        range_ok=np.isfinite(range_m) and r_min_m <= range_m <= r_max_m,
        u=u,
        v=v,
        z=z,
        range_m=range_m,
    )


CAMERA_ORDER = ("image_00", "image_01")


@dataclass(frozen=True)
class CameraAssignment:
    camera: int = 255
    r_gt: float = -1.0
    u: float = 0.0
    v: float = 0.0
    p_C: np.ndarray | None = None


def assign_primary_camera(
    point_v: np.ndarray,
    transforms: dict[str, np.ndarray],
    intrinsics: dict[str, np.ndarray],
    image_sizes: dict[str, tuple[int, int]],
    *,
    r_min_m: float = 1.5,
    r_max_m: float = 60.0,
    margin_px: int = 5,
) -> CameraAssignment:
    best: CameraAssignment | None = None
    best_score = float("inf")
    point = np.asarray(point_v, dtype=np.float64)
    for index, camera in enumerate(CAMERA_ORDER):
        transform = transforms.get(camera)
        K = intrinsics.get(camera)
        size = image_sizes.get(camera)
        if transform is None or K is None or size is None:
            continue
        eligibility = evaluate_camera_eligibility(
            point, transform, K, size[0], size[1], r_min_m=r_min_m, r_max_m=r_max_m, margin_px=margin_px
        )
        if not eligibility.valid:
            continue
        dx = (eligibility.u - float(K[0, 2])) / float(K[0, 0])
        dy = (eligibility.v - float(K[1, 2])) / float(K[1, 1])
        score = dx * dx + dy * dy
        if not np.isfinite(score):
            continue
        if best is None or score < best_score:
            best_score = score
            homogeneous = np.append(point, 1.0)
            best = CameraAssignment(
                camera=index,
                r_gt=eligibility.range_m,
                u=eligibility.u,
                v=eligibility.v,
                p_C=(np.asarray(transform) @ homogeneous)[:3],
            )
    return best if best is not None else CameraAssignment()


def assign_primary_camera_batch(
    points_v: np.ndarray,
    transforms: dict[str, np.ndarray],
    intrinsics: dict[str, np.ndarray],
    image_sizes: dict[str, tuple[int, int]],
    *,
    r_min_m: float = 1.5,
    r_max_m: float = 60.0,
    margin_px: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised camera assignment.

    Returns ``(camera, r_gt, u, v)`` arrays; ``camera`` is 255 where no camera is
    eligible and ``r_gt`` is ``-1`` there.
    """

    points = np.asarray(points_v, dtype=np.float64)
    count = points.shape[0]
    camera = np.full(count, 255, dtype=np.uint8)
    r_gt = np.full(count, -1.0, dtype=np.float64)
    u_out = np.zeros(count, dtype=np.float64)
    v_out = np.zeros(count, dtype=np.float64)
    best_score = np.full(count, np.inf, dtype=np.float64)

    margin = float(max(margin_px, 0))
    for index, name in enumerate(CAMERA_ORDER):
        transform = transforms.get(name)
        K = intrinsics.get(name)
        size = image_sizes.get(name)
        if transform is None or K is None or size is None:
            continue
        width, height = size
        projection = project_points(
            points,
            transform,
            K,
            width,
            height,
            min_range_m=0.0,
            max_range_m=1.0e9,
            margin_px=0,
            clamp_range=False,
        )
        z = projection.z
        range_m = projection.range_m
        u = projection.uv[:, 0]
        v = projection.uv[:, 1]
        z_positive = np.isfinite(z) & (z > 0.0)
        max_u = float(width - 1) - margin
        max_v = float(height - 1) - margin
        in_margin = np.isfinite(u) & np.isfinite(v) & (u >= margin) & (u <= max_u) & (v >= margin) & (v <= max_v)
        range_ok = np.isfinite(range_m) & (range_m >= r_min_m) & (range_m <= r_max_m)
        eligible = z_positive & in_margin & range_ok
        dx = (u - float(K[0, 2])) / float(K[0, 0])
        dy = (v - float(K[1, 2])) / float(K[1, 1])
        score = dx * dx + dy * dy
        score = np.where(eligible & np.isfinite(score), score, np.inf)
        replace = score < best_score
        best_score[replace] = score[replace]
        camera[replace] = index
        r_gt[replace] = range_m[replace]
        u_out[replace] = u[replace]
        v_out[replace] = v[replace]
    return camera, r_gt, u_out, v_out
