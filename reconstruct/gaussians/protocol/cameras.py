"""Frozen ``geometry/cameras.json`` loader for the Phase-5 protocol.

Camera 0 (``image_00``) is the only camera that enters the Phase-5 GT-ray
metrics.  The loader validates the schema, the 60-frame window and the
``T_C_W == inv(T_W_C)`` round trip so a sampling run cannot silently use a
pose that disagrees with the frozen calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXPECTED_SCHEMA_VERSION = 1
DEFAULT_EXPECTED_FRAMES = 60
EXPECTED_WIDTH = 1408
EXPECTED_HEIGHT = 376
# The frozen export stores both poses with finite decimal precision: the
# rotation round trip is accurate to about 1.1e-6 and the translation to about
# 3.7 mm.  Sampling only uses the rotation, so it is checked at 1e-5 and the
# translation at 1 cm.
POSE_ROTATION_TOLERANCE = 1e-5
POSE_TRANSLATION_TOLERANCE = 1e-2


def _as_matrix(value: object, *, path: Path, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: {name} must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: {name} contains a non-finite value")
    return matrix


def load_cameras(
    path: str | Path, expected_frames: int = DEFAULT_EXPECTED_FRAMES
) -> dict:
    """Load camera 0 intrinsics and the per-frame poses.

    ``expected_frames`` is 60 for production; tests may pass a smaller value
    for a synthetic fixture.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {document.get('schema_version')!r}"
        )

    cameras = document.get("cameras")
    if not isinstance(cameras, dict) or "image_00" not in cameras:
        raise ValueError(f"{path}: missing cameras.image_00")
    camera = cameras["image_00"]
    K = np.asarray(camera.get("K"), dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{path}: cameras.image_00.K must be a 3x3 matrix")
    if not np.all(np.isfinite(K)):
        raise ValueError(f"{path}: cameras.image_00.K contains a non-finite value")
    width = int(camera.get("width", EXPECTED_WIDTH))
    height = int(camera.get("height", EXPECTED_HEIGHT))

    entries = document.get("frames")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: frames must be a list")
    if len(entries) != int(expected_frames):
        raise ValueError(
            f"{path}: expected {expected_frames} frames, found {len(entries)}"
        )

    frames: dict[int, dict] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "frame" not in entry:
            raise ValueError(f"{path}: frame entry {index} has no frame id")
        frame = int(entry["frame"])
        if frame in frames:
            raise ValueError(f"{path}: duplicate frame {frame}")
        T_W_C = _as_matrix(
            entry.get("T_W_image_00_rect"), path=path, name=f"frame {frame} T_W_C"
        )
        T_C_W = _as_matrix(
            entry.get("T_image_00_rect_W"), path=path, name=f"frame {frame} T_C_W"
        )
        inverse = np.linalg.inv(T_W_C)
        rotation_delta = float(np.max(np.abs(T_C_W[:3, :3] - inverse[:3, :3])))
        translation_delta = float(
            np.max(np.abs(T_C_W[:3, 3] - inverse[:3, 3]))
        )
        if rotation_delta > POSE_ROTATION_TOLERANCE:
            raise ValueError(
                f"{path}: T_C_W rotation does not invert T_W_C for frame "
                f"{frame} (max |delta|={rotation_delta!r})"
            )
        if translation_delta > POSE_TRANSLATION_TOLERANCE:
            raise ValueError(
                f"{path}: T_C_W translation disagrees with inv(T_W_C) for "
                f"frame {frame} (max |delta|={translation_delta!r} m)"
            )
        frames[frame] = {
            "T_W_C": T_W_C.tolist(),
            "T_C_W": T_C_W.tolist(),
        }

    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "K": K.tolist(),
        "width": width,
        "height": height,
        "frames": frames,
    }


__all__ = [
    "DEFAULT_EXPECTED_FRAMES",
    "EXPECTED_HEIGHT",
    "EXPECTED_WIDTH",
    "load_cameras",
]
