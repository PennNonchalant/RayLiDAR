"""Velodyne motion compensation ("uncurl") using the KITTI-360 convention.

The official ``curlVelodyneData`` convention assigns each point the fraction
``u = 0.5 + atan2(y, x) / 2pi`` of the current-to-next sweep, interpolates the
ego pose by ``u``, and re-expresses the point in the current sensor frame.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 6.283185307179586476925286766559
AZIMUTH_CLAMP = 1.0 - 1e-12


def azimuth_fraction(points: np.ndarray) -> np.ndarray:
    """Per-point sweep fraction for an ``(N, 3)`` array, clamped to ``[0, 1)``."""

    points = np.asarray(points, dtype=np.float64)
    u = 0.5 + np.arctan2(points[:, 1], points[:, 0]) / TWO_PI
    return np.clip(u, 0.0, AZIMUTH_CLAMP)


def interpolate_rigid(T_a: np.ndarray, T_b: np.ndarray, u: float) -> np.ndarray:
    """SE(3) interpolation between two rigid transforms."""

    relative = np.linalg.inv(T_a) @ T_b
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) <= 1e-12:
        scaled_rotation = np.eye(3)
    else:
        axis = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * np.sin(angle))
        theta = angle * u
        kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        scaled_rotation = np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)
    delta = np.eye(4)
    delta[:3, :3] = scaled_rotation
    delta[:3, 3] = u * translation
    return T_a @ delta


def uncurl_scan(
    points: np.ndarray,
    T_W_V_curr: np.ndarray,
    T_W_V_next: np.ndarray,
    t_curr_ns: int,
    t_next_ns: int,
) -> np.ndarray:
    """Return an ``(N, 3)`` array of uncurled points (``NaN`` for non-finite cells)."""

    points = np.asarray(points, dtype=np.float64)[:, :3]
    if t_next_ns <= t_curr_ns:
        raise ValueError(f"uncurl sweep interval is not strictly positive: [{t_curr_ns}, {t_next_ns})")

    finite = np.isfinite(points).all(axis=1)
    result = np.full(points.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return result

    relative = np.linalg.inv(T_W_V_curr) @ T_W_V_next
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))

    p = points[finite]
    u = azimuth_fraction(p)
    if abs(angle) <= 1e-12:
        result[finite] = p + u[:, None] * translation
        return result

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    kx2 = kx @ kx
    theta = angle * u
    sin_t = np.sin(theta)[:, None, None]
    cos_t = (1.0 - np.cos(theta))[:, None, None]
    rotations = np.eye(3)[None, :, :] + sin_t * kx[None, :, :] + cos_t * kx2[None, :, :]
    result[finite] = np.einsum("nij,nj->ni", rotations, p) + u[:, None] * translation
    return result
