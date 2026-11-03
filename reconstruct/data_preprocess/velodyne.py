"""Velodyne scan loading and timestamp parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_utils import PreprocessError
from .layout import load_timestamps

RECORD_BYTES = 16
DEFAULT_MAX_POINTS = 2_000_000


def load_scan(path: str | Path, max_points: int = DEFAULT_MAX_POINTS) -> np.ndarray:
    """Return an ``(N, 4)`` float32 array of ``x, y, z, intensity`` rows."""

    target = Path(path)
    if not target.is_file():
        raise PreprocessError(f"cannot read Velodyne scan {target}")
    size = target.stat().st_size
    if size == 0:
        raise PreprocessError(f"Velodyne scan is empty: {target}")
    if size % RECORD_BYTES != 0:
        raise PreprocessError(f"Velodyne scan size is not a multiple of 16 bytes: {target}")
    count = size // RECORD_BYTES
    if count > max_points:
        raise PreprocessError(f"Velodyne scan has {count} points, above the {max_points} point cap")
    return np.fromfile(target, dtype="<f4").reshape(count, 4)


def load_lidar_timestamps(path: str | Path) -> list[int]:
    return load_timestamps(path)
