"""Binary readers/writers for the frozen simulation I/O contract.

The frozen layouts are little-endian, schema version 1:

* ``gt_rays.bin`` - magic ``RLGR``, 32-byte header, 96-byte records
  ``<Q9dIH10s>`` (ray_id, origin_W[3], dir_W[3], t_gt, t_min, t_max, frame,
  camera, reserved[10]).
* ``hits.bin`` - magic ``RLHT``, 32-byte header, 24-byte records
  ``<QdB3sI>`` (ray_id, t_pred, hit, reserved[3], prim_id).
* ``gt_rays_flags.bin`` - Phase-5 sidecar produced by IO-02, magic ``RLGF``,
  32-byte header, 16-byte records ``<QBBBBHH>`` (ray_id, bucket, occluded,
  camera, reserved, semantic_id, reserved).

A miss is always ``hit=0``, ``t_pred=+inf`` and ``prim_id=0xFFFFFFFF``; no
other miss convention is accepted.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path
from typing import Iterable, Sequence

GT_RAYS_HEADER = 32
GT_RAYS_RECORD = 96
HITS_HEADER = 32
HITS_RECORD = 24
FLAGS_HEADER = 32
FLAGS_RECORD = 16

GT_RAYS_MAGIC = b"RLGR"
HITS_MAGIC = b"RLHT"
FLAGS_MAGIC = b"RLGF"
VERSION = 1

PRIM_ID_UNAVAILABLE = 0xFFFFFFFF

_HEADER = struct.Struct("<4sHHI20s")
_GT_RAYS_RECORD = struct.Struct("<Q9dIH10s")
_HITS_RECORD = struct.Struct("<QdB3sI")
_FLAGS_RECORD = struct.Struct("<QBBBBHH")

_ZERO_20 = b"\x00" * 20
_ZERO_10 = b"\x00" * 10
_ZERO_3 = b"\x00" * 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: str | Path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def _read_header(
    handle,
    path: Path,
    *,
    magic: bytes,
    record_size: int,
) -> tuple[int, int]:
    raw = handle.read(_HEADER.size)
    if len(raw) != _HEADER.size:
        raise ValueError(f"{path}: truncated header")
    actual_magic, version, actual_record_size, count, _reserved = _HEADER.unpack(raw)
    if actual_magic != magic:
        raise ValueError(f"{path}: wrong magic {actual_magic!r}, expected {magic!r}")
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}, expected {VERSION}")
    if actual_record_size != record_size:
        raise ValueError(
            f"{path}: record_size {actual_record_size}, expected {record_size}"
        )
    return int(count), int(actual_record_size)


def _read_all_records(path: Path, *, magic: bytes, record_size: int) -> tuple[int, bytes]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise
    with path.open("rb") as handle:
        count, _ = _read_header(handle, path, magic=magic, record_size=record_size)
        expected = GT_RAYS_HEADER + count * record_size
        if size != expected:
            raise ValueError(
                f"{path}: file length {size} does not match 32 + {count} * {record_size}"
            )
        body = handle.read()
    if len(body) != count * record_size:
        raise ValueError(f"{path}: truncated record body")
    return count, body


def _require_int(value: object, *, name: str, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or value >= (1 << bits):
        raise ValueError(f"{name} does not fit in uint{bits}: {value!r}")
    return int(value)


# ---------------------------------------------------------------------------
# gt_rays.bin
# ---------------------------------------------------------------------------


def read_gt_rays(path: str | Path) -> list[dict]:
    """Read and validate ``gt_rays.bin``, preserving record order."""
    path = Path(path)
    count, body = _read_all_records(
        path, magic=GT_RAYS_MAGIC, record_size=GT_RAYS_RECORD
    )
    records: list[dict] = []
    seen: set[int] = set()
    unpack = _GT_RAYS_RECORD.unpack_from
    for index in range(count):
        values = unpack(body, index * GT_RAYS_RECORD)
        ray_id = int(values[0])
        origin = (float(values[1]), float(values[2]), float(values[3]))
        direction = (float(values[4]), float(values[5]), float(values[6]))
        t_gt = float(values[7])
        t_min = float(values[8])
        t_max = float(values[9])
        frame = int(values[10])
        camera = int(values[11])

        if ray_id in seen:
            raise ValueError(f"{path}: duplicate ray_id {ray_id} at record {index}")
        seen.add(ray_id)

        for value in (*origin, *direction, t_gt, t_min, t_max):
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite value at record {index}")

        norm = math.sqrt(
            direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2]
        )
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(
                f"{path}: non-unit dir_W at record {index} (|dir|={norm!r})"
            )
        if not t_min > 0.0:
            raise ValueError(f"{path}: t_min must be > 0 at record {index}")
        if not (t_min <= t_gt <= t_max):
            raise ValueError(
                f"{path}: t_gt {t_gt!r} is outside [{t_min!r}, {t_max!r}] "
                f"at record {index}"
            )
        if camera not in (0, 1):
            raise ValueError(f"{path}: unknown camera {camera} at record {index}")

        records.append(
            {
                "ray_id": ray_id,
                "origin_W": origin,
                "dir_W": direction,
                "t_gt": t_gt,
                "t_min": t_min,
                "t_max": t_max,
                "frame": frame,
                "camera": camera,
            }
        )
    return records


# ---------------------------------------------------------------------------
# gt_rays_flags.bin
# ---------------------------------------------------------------------------


def read_flags(path: str | Path) -> list[dict]:
    """Read and validate the Phase-5 ``RLGF`` flags sidecar."""
    path = Path(path)
    count, body = _read_all_records(
        path, magic=FLAGS_MAGIC, record_size=FLAGS_RECORD
    )
    records: list[dict] = []
    unpack = _FLAGS_RECORD.unpack_from
    for index in range(count):
        values = unpack(body, index * FLAGS_RECORD)
        records.append(
            {
                "ray_id": int(values[0]),
                "bucket": int(values[1]),
                "occluded": int(values[2]),
                "camera": int(values[3]),
                "semantic_id": int(values[5]),
            }
        )
    return records


# ---------------------------------------------------------------------------
# hits.bin
# ---------------------------------------------------------------------------


def _validate_hit_record(record: dict, index: int) -> tuple[int, float, int, int]:
    ray_id = _require_int(record.get("ray_id"), name="ray_id", bits=64)
    hit = record.get("hit")
    if isinstance(hit, bool):
        hit_value = int(hit)
    elif isinstance(hit, int) and hit in (0, 1):
        hit_value = hit
    else:
        raise ValueError(f"hits record {index}: hit must be 0 or 1")
    t_pred = float(record.get("t_pred"))
    prim_id = _require_int(record.get("prim_id"), name="prim_id", bits=32)

    if hit_value == 1:
        if not (math.isfinite(t_pred) and t_pred > 0.0):
            raise ValueError(
                f"hits record {index}: a hit needs a finite positive t_pred"
            )
    else:
        if not (math.isinf(t_pred) and t_pred > 0.0):
            raise ValueError(
                f"hits record {index}: a miss must use t_pred=+inf"
            )
        if prim_id != PRIM_ID_UNAVAILABLE:
            raise ValueError(
                f"hits record {index}: a miss must use prim_id=0xFFFFFFFF"
            )
    return ray_id, t_pred, hit_value, prim_id


def write_hits(path: str | Path, hits: Iterable[dict]) -> str:
    """Write ``hits.bin`` atomically and return its SHA-256."""
    records = list(hits)
    seen: set[int] = set()
    buffer = bytearray(HITS_HEADER + len(records) * HITS_RECORD)
    _HEADER.pack_into(buffer, 0, HITS_MAGIC, VERSION, HITS_RECORD, len(records), _ZERO_20)
    pack = _HITS_RECORD.pack_into
    for index, record in enumerate(records):
        ray_id, t_pred, hit_value, prim_id = _validate_hit_record(record, index)
        if ray_id in seen:
            raise ValueError(f"hits record {index}: duplicate ray_id {ray_id}")
        seen.add(ray_id)
        pack(
            buffer,
            HITS_HEADER + index * HITS_RECORD,
            ray_id,
            t_pred,
            hit_value,
            _ZERO_3,
            prim_id,
        )
    return _atomic_write_bytes(Path(path), bytes(buffer))


def read_hits(path: str | Path) -> list[dict]:
    """Read and validate ``hits.bin``, preserving record order."""
    path = Path(path)
    count, body = _read_all_records(
        path, magic=HITS_MAGIC, record_size=HITS_RECORD
    )
    records: list[dict] = []
    seen: set[int] = set()
    unpack = _HITS_RECORD.unpack_from
    for index in range(count):
        values = unpack(body, index * HITS_RECORD)
        hit_raw = int(values[2])
        if hit_raw not in (0, 1):
            raise ValueError(f"{path}: invalid hit value {hit_raw} at record {index}")
        record = {
            "ray_id": int(values[0]),
            "t_pred": float(values[1]),
            "hit": bool(hit_raw),
            "prim_id": int(values[4]),
        }
        ray_id, t_pred, hit_value, prim_id = _validate_hit_record(record, index)
        if ray_id in seen:
            raise ValueError(f"{path}: duplicate ray_id {ray_id} at record {index}")
        seen.add(ray_id)
        records.append(
            {
                "ray_id": ray_id,
                "t_pred": t_pred,
                "hit": bool(hit_value),
                "prim_id": prim_id,
            }
        )
    return records


# ---------------------------------------------------------------------------
# subset filters
# ---------------------------------------------------------------------------


def _filter_records(
    src: str | Path,
    dst: str | Path,
    keep_ray_ids: Iterable[int],
    *,
    magic: bytes,
    record_size: int,
) -> list[int]:
    src = Path(src)
    keep = {int(ray_id) for ray_id in keep_ray_ids}
    count, body = _read_all_records(src, magic=magic, record_size=record_size)
    unpack = struct.Struct("<Q").unpack_from
    kept: list[int] = []
    selected = bytearray()
    for index in range(count):
        offset = index * record_size
        ray_id = int(unpack(body, offset)[0])
        if ray_id in keep:
            kept.append(ray_id)
            selected.extend(body[offset : offset + record_size])
    missing = keep.difference(kept)
    if missing:
        raise ValueError(
            f"{src}: requested ray_ids not present: {sorted(missing)[:8]}"
        )
    header = _HEADER.pack(magic, VERSION, record_size, len(kept), _ZERO_20)
    _atomic_write_bytes(Path(dst), header + bytes(selected))
    return kept


def filter_gt_rays(
    src: str | Path, dst: str | Path, keep_ray_ids: Iterable[int]
) -> list[int]:
    """Copy only selected 96-byte records, preserving source order."""
    return _filter_records(
        src,
        dst,
        keep_ray_ids,
        magic=GT_RAYS_MAGIC,
        record_size=GT_RAYS_RECORD,
    )


def filter_hits(
    src: str | Path, dst: str | Path, keep_ray_ids: Iterable[int]
) -> list[int]:
    """Copy only selected 24-byte hit records, preserving source order."""
    return _filter_records(
        src, dst, keep_ray_ids, magic=HITS_MAGIC, record_size=HITS_RECORD
    )


__all__ = [
    "FLAGS_HEADER",
    "FLAGS_RECORD",
    "GT_RAYS_HEADER",
    "GT_RAYS_RECORD",
    "HITS_HEADER",
    "HITS_RECORD",
    "PRIM_ID_UNAVAILABLE",
    "filter_gt_rays",
    "filter_hits",
    "read_flags",
    "read_gt_rays",
    "read_hits",
    "sha256_file",
    "write_hits",
]
