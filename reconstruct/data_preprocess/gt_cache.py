"""Frozen per-frame static GT point cache (``RLGT``).

Header 64 bytes: magic ``RLGT``, version u16, record size u16, count u32,
frame u32, reserved 48. Record 32 bytes: ``p_L f32[3]``, ``r_gt f32``,
``frame u32``, ``semantic_id u16``, ``camera u8``, ``valid u8``, pad[8].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import PreprocessError, ensure_dir, sha256_file, utc_now, write_json

MAGIC = b"RLGT"
VERSION = 1
RECORD_SIZE = 32
HEADER_SIZE = 64
NO_CAMERA = 255
UNKNOWN_LABEL = 65535


@dataclass(frozen=True)
class GtPoint:
    p_L: tuple[float, float, float]
    frame: int
    semantic_id: int
    camera: int
    valid: int
    r_gt: float


def write_gt_frame(path: str | Path, frame: int, points: list[GtPoint]) -> Path:
    if frame < 0:
        raise PreprocessError("GT frame id must be non-negative")
    target = Path(path)
    ensure_dir(target.parent)
    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    header[4:6] = np.uint16(VERSION).tobytes()
    header[6:8] = np.uint16(RECORD_SIZE).tobytes()
    header[8:12] = np.uint32(len(points)).tobytes()
    header[12:16] = np.uint32(frame).tobytes()

    body = bytearray(len(points) * RECORD_SIZE)
    for index, point in enumerate(points):
        base = index * RECORD_SIZE
        values = (float(point.p_L[0]), float(point.p_L[1]), float(point.p_L[2]), float(point.r_gt))
        if not np.isfinite(values).all():
            raise PreprocessError("refusing to write a non-finite GT point")
        body[base : base + 16] = np.asarray(values, dtype="<f4").tobytes()
        body[base + 16 : base + 20] = np.uint32(point.frame).tobytes()
        body[base + 20 : base + 22] = np.uint16(point.semantic_id).tobytes()
        body[base + 22] = point.camera
        body[base + 23] = point.valid
    target.write_bytes(bytes(header) + bytes(body))
    return target


def read_gt_frame(path: str | Path) -> tuple[int, list[GtPoint]]:
    target = Path(path)
    raw = target.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise PreprocessError(f"GT frame is shorter than its header: {target}")
    if raw[0:4] != MAGIC:
        raise PreprocessError(f"GT frame has a wrong magic: {target}")
    version = int(np.frombuffer(raw[4:6], dtype="<u2")[0])
    record_size = int(np.frombuffer(raw[6:8], dtype="<u2")[0])
    count = int(np.frombuffer(raw[8:12], dtype="<u4")[0])
    frame = int(np.frombuffer(raw[12:16], dtype="<u4")[0])
    if version != VERSION or record_size != RECORD_SIZE:
        raise PreprocessError(f"GT frame has an unsupported version or record size: {target}")
    if len(raw) != HEADER_SIZE + count * RECORD_SIZE:
        raise PreprocessError(f"GT frame length does not match its header count: {target}")
    points: list[GtPoint] = []
    for index in range(count):
        base = HEADER_SIZE + index * RECORD_SIZE
        record = raw[base : base + RECORD_SIZE]
        px, py, pz, r_gt = np.frombuffer(record[0:16], dtype="<f4")
        points.append(
            GtPoint(
                p_L=(float(px), float(py), float(pz)),
                frame=int(np.frombuffer(record[16:20], dtype="<u4")[0]),
                semantic_id=int(np.frombuffer(record[20:22], dtype="<u2")[0]),
                camera=int(record[22]),
                valid=int(record[23]),
                r_gt=float(r_gt),
            )
        )
    return frame, points


def build_gt_manifest(
    out_dir: str | Path,
    sequence: str,
    frames: list[dict],
    config_sha256: str,
    filename: str = "gt_manifest.json",
) -> dict:
    totals = {"frames": 0, "points": 0, "camera_00": 0, "camera_01": 0, "camera_unknown": 0}
    for entry in frames:
        totals["frames"] += 1
        totals["points"] += int(entry["count"])
        totals["camera_00"] += int(entry["camera_00"])
        totals["camera_01"] += int(entry["camera_01"])
        totals["camera_unknown"] += int(entry["camera_unknown"])
    document = {
        "schema_version": 1,
        "sequence": sequence,
        "generated_at_utc": utc_now(),
        "frames": frames,
        "totals": totals,
        "config_sha256": config_sha256,
    }
    write_json(Path(out_dir) / filename, document)
    return document


def frame_sha256(path: str | Path) -> str:
    return sha256_file(path)
