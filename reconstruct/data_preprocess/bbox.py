"""KITTI-360 3D dynamic bounding boxes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import PreprocessError

MAX_OBJECTS = 200_000
AABB_EPSILON = 1e-9
CUBE_EPSILON = 1e-12


@dataclass(frozen=True)
class DynamicBBox:
    label: str
    semantic_id: int
    timestamp: int
    start_frame: int
    end_frame: int
    dynamic: bool
    T_W_B: np.ndarray
    T_B_W: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def _parse_int(text: str | None, what: str) -> int:
    if text is None:
        raise PreprocessError(f"bbox object is missing <{what}>")
    try:
        return int(text.strip())
    except ValueError as exc:
        raise PreprocessError(f"bbox field <{what}> is not an integer") from exc


def _parse_bool(text: str | None, what: str) -> bool:
    value = _parse_int(text, what)
    if value not in (0, 1):
        raise PreprocessError(f"bbox field <{what}> must be 0 or 1")
    return value == 1


def _parse_transform(element: ET.Element) -> np.ndarray:
    data = element.findtext("data")
    if data is None:
        raise PreprocessError("bbox transform is missing <data>")
    values = data.split()
    if len(values) != 16:
        raise PreprocessError("bbox transform must contain exactly 16 values")
    try:
        return np.asarray([float(v) for v in values], dtype=np.float64).reshape(4, 4)
    except ValueError as exc:
        raise PreprocessError("bbox transform contains a non-numeric value") from exc


def _fill_derived(world_from_box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = []
    for x_sign in (-1, 1):
        for y_sign in (-1, 1):
            for z_sign in (-1, 1):
                corners.append([0.5 * x_sign, 0.5 * y_sign, 0.5 * z_sign, 1.0])
    world = (np.asarray(corners, dtype=np.float64) @ world_from_box.T)[:, :3]
    return world.min(axis=0), world.max(axis=0), np.linalg.inv(world_from_box)


def load_dynamic_bboxes(path: str | Path, max_objects: int = MAX_OBJECTS) -> list[DynamicBBox]:
    target = Path(path)
    if not target.is_file():
        raise PreprocessError(f"cannot read bbox XML {target}")
    text = target.read_text(encoding="utf-8", errors="replace")
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise PreprocessError(f"bbox XML {target} contains a document type or entity declaration")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise PreprocessError(f"cannot parse bbox XML {target}: {exc}") from exc

    boxes: list[DynamicBBox] = []
    for element in root:
        if not element.tag.startswith("object"):
            continue
        transform_element = element.find("transform")
        if transform_element is None:
            raise PreprocessError("bbox object is missing <transform>")
        transform = _parse_transform(transform_element)
        aabb_min, aabb_max, inverse = _fill_derived(transform)
        boxes.append(
            DynamicBBox(
                label=(element.findtext("label") or "").strip(),
                semantic_id=_parse_int(element.findtext("semanticId"), "semanticId"),
                timestamp=_parse_int(element.findtext("timestamp"), "timestamp"),
                start_frame=_parse_int(element.findtext("start_frame"), "start_frame"),
                end_frame=_parse_int(element.findtext("end_frame"), "end_frame"),
                dynamic=_parse_bool(element.findtext("dynamic"), "dynamic"),
                T_W_B=transform,
                T_B_W=inverse,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
            )
        )
        if len(boxes) > max_objects:
            raise PreprocessError(f"bbox XML {target} exceeds the object cap of {max_objects}")
    return boxes


def point_in_bbox(point_w: np.ndarray, box: DynamicBBox) -> bool:
    point = np.asarray(point_w, dtype=np.float64)
    if not np.isfinite(point).all():
        return False
    if np.any(point < box.aabb_min - AABB_EPSILON) or np.any(point > box.aabb_max + AABB_EPSILON):
        return False
    local = (box.T_B_W @ np.append(point, 1.0))[:3]
    return bool(np.all(np.abs(local) <= 0.5 + CUBE_EPSILON))


def applicable_boxes(
    boxes: list[DynamicBBox], frame: int, max_time_gap_frames: int
) -> list[DynamicBBox]:
    return [
        box
        for box in boxes
        if box.dynamic
        and box.start_frame <= frame <= box.end_frame
        and abs(box.timestamp - frame) <= max_time_gap_frames
    ]
