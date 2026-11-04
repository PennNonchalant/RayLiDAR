"""Frozen camera projection, nearest-pixel sampling, and the hit rule.

Both Gaussian methods use exactly these functions.  The protocol is:

1. rotate the W-frame unit ray into the camera frame (``T_C_W``);
2. project it with the frozen pinhole ``K`` and reject rays behind the camera;
3. round to the nearest pixel with **half-up** rounding;
4. require a finite ``alpha >= alpha_threshold``;
5. require finite positive depth;
6. convert z-depth to ray range by ``z * ||((u-cx)/fx, (v-cy)/fy, 1)||``;
7. accept the sample only when ``t_min <= t_pred <= t_max``.

Every failure is a miss with ``t_pred = +inf`` and a reason string from
``projection``, ``pixel_bounds``, ``alpha``, ``depth``, ``range`` (``none`` is
a hit).
"""

from __future__ import annotations

import math

import numpy as np

MISS = math.inf
REASONS = ("projection", "pixel_bounds", "alpha", "depth", "range", "none")


def _intrinsics(K: object) -> np.ndarray:
    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("K must be a 3x3 intrinsic matrix")
    return matrix


def project_to_pixel(dir_W: object, T_C_W: object, K: object) -> tuple[float, float] | None:
    """Project a W-frame direction into pixel coordinates.

    Returns ``None`` when the ray is not in front of the camera (``d_C.z <= 0``)
    or when the projection is not finite.
    """
    transform = np.asarray(T_C_W, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_C_W must be a 4x4 matrix")
    direction = np.asarray(dir_W, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("dir_W must be a 3-vector")
    intrinsics = _intrinsics(K)

    d_C = transform[:3, :3] @ direction
    if not np.all(np.isfinite(d_C)):
        return None
    if d_C[2] <= 0.0:
        return None

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    u = fx * d_C[0] / d_C[2] + cx
    v = fy * d_C[1] / d_C[2] + cy
    if not (math.isfinite(float(u)) and math.isfinite(float(v))):
        return None
    return float(u), float(v)


def nearest_pixel(
    u: float, v: float, width: int, height: int
) -> tuple[int, int] | None:
    """Return the half-up nearest pixel, or ``None`` when it is outside."""
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    iu = math.floor(u + 0.5)
    iv = math.floor(v + 0.5)
    if iu < 0 or iu >= int(width) or iv < 0 or iv >= int(height):
        return None
    return int(iu), int(iv)


def pixel_ray(iu: int, iv: int, K: object) -> np.ndarray:
    """Return the camera-frame direction through pixel ``(iu, iv)``."""
    intrinsics = _intrinsics(K)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    return np.asarray(
        [(float(iu) - cx) / fx, (float(iv) - cy) / fy, 1.0], dtype=np.float64
    )


def z_to_range(z: float, iu: int, iv: int, K: object) -> float:
    """Convert camera z-depth to ray range at pixel ``(iu, iv)``."""
    return float(z) * float(np.linalg.norm(pixel_ray(iu, iv, K)))


def _empty_result() -> dict:
    return {
        "hit": False,
        "t_pred": MISS,
        "reason": "none",
        "u": None,
        "v": None,
        "pixel": None,
        "alpha": None,
        "z": None,
    }


def sample_ray(
    depth: object,
    alpha: object,
    ray: dict,
    T_C_W: object,
    K: object,
    alpha_threshold: float = 0.5,
    convention: str = "z_depth",
) -> dict:
    """Sample a rendered depth/alpha map for one GT ray.

    ``depth`` and ``alpha`` must be 2-D arrays of equal shape.  Production
    callers use ``(376, 1408)``; the synthetic unit tests use tiny maps.
    """
    depth_array = np.asarray(depth)
    alpha_array = np.asarray(alpha)
    if depth_array.ndim != 2 or alpha_array.ndim != 2:
        raise ValueError("depth and alpha must be 2-D arrays")
    if depth_array.shape != alpha_array.shape:
        raise ValueError(
            f"depth shape {depth_array.shape} does not match alpha shape "
            f"{alpha_array.shape}"
        )
    if convention not in ("z_depth", "range"):
        raise ValueError(f"unsupported depth convention: {convention!r}")

    height, width = depth_array.shape
    result = _empty_result()

    projected = project_to_pixel(ray["dir_W"], T_C_W, K)
    if projected is None:
        result["reason"] = "projection"
        return result
    u, v = projected
    result["u"] = u
    result["v"] = v

    pixel = nearest_pixel(u, v, width, height)
    if pixel is None:
        result["reason"] = "pixel_bounds"
        return result
    iu, iv = pixel
    result["pixel"] = [iu, iv]

    alpha_value = float(alpha_array[iv, iu])
    result["alpha"] = alpha_value
    if not (math.isfinite(alpha_value) and alpha_value >= float(alpha_threshold)):
        result["reason"] = "alpha"
        return result

    z = float(depth_array[iv, iu])
    result["z"] = z
    if not (math.isfinite(z) and z > 0.0):
        result["reason"] = "depth"
        return result

    if convention == "z_depth":
        t_pred = z_to_range(z, iu, iv, K)
    else:
        t_pred = z

    if not (
        math.isfinite(t_pred)
        and float(ray["t_min"]) <= t_pred <= float(ray["t_max"])
    ):
        result["reason"] = "range"
        return result

    result["hit"] = True
    result["t_pred"] = float(t_pred)
    result["reason"] = "none"
    return result


__all__ = [
    "MISS",
    "REASONS",
    "nearest_pixel",
    "pixel_ray",
    "project_to_pixel",
    "sample_ray",
    "z_to_range",
]
