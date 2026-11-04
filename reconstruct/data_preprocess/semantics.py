"""KITTI-360 semantic label images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import PreprocessError

DYNAMIC_LABELS = frozenset({24, 25, 26, 27, 28, 32, 33})
UNKNOWN_LABEL = 65535


@dataclass(frozen=True)
class SemanticImage:
    width: int
    height: int
    labels: np.ndarray


def load_semantics(path: str | Path, max_pixels: int = 50_000_000) -> SemanticImage:
    target = Path(path)
    if not target.is_file():
        raise PreprocessError(f"cannot read semantic image: {target}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise PreprocessError("Pillow is required to read semantic PNGs") from exc

    with Image.open(target) as image:
        if image.width * image.height > max_pixels:
            raise PreprocessError(f"semantic image {target} exceeds {max_pixels} pixels")
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            array = np.asarray(image, dtype=np.uint16)
        else:
            array = np.asarray(image.convert("L"), dtype=np.uint16)
    if array.ndim != 2:
        raise PreprocessError(f"semantic image {target} is not single-channel")
    return SemanticImage(width=int(array.shape[1]), height=int(array.shape[0]), labels=array)


def is_dynamic_label(label_id: int) -> bool:
    return int(label_id) in DYNAMIC_LABELS


def label_at(image: SemanticImage, u: float, v: float) -> int:
    if not (np.isfinite(u) and np.isfinite(v)):
        return UNKNOWN_LABEL
    x = int(np.floor(u + 0.5))
    y = int(np.floor(v + 0.5))
    if x < 0 or y < 0 or x >= image.width or y >= image.height:
        return UNKNOWN_LABEL
    return int(image.labels[y, x])
