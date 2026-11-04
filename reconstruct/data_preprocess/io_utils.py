"""Small IO helpers shared by the preprocessing modules.

All binary artifacts use little-endian layouts and are written atomically so a
failed stage never leaves a half-written file that the next stage would accept.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PreprocessError(RuntimeError):
    """A preprocessing precondition or artifact invariant failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    target = Path(path)
    if not target.is_file():
        raise PreprocessError(f"missing JSON file: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreprocessError(f"invalid JSON in {target}: {exc}") from exc


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    text = json.dumps(value, indent=indent, sort_keys=True) + "\n"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)
    return target


def write_bytes_atomic(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return target


def pack(fmt: str, *values: Any) -> bytes:
    return struct.pack("<" + fmt, *values)
