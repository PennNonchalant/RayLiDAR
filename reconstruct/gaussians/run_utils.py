"""Shared execution helpers for the Phase-5 Gaussian pipeline.

Plans 05 and 06 both train and render through this module so that the two
methods share one subprocess/logging/manifest path: a list argv (never
``shell=True``), one combined log per command, canonical JSON manifests, and a
single free-space guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MINIMUM_FREE_BYTES = 20_000_000_000


def utc_now() -> str:
    """Return the current UTC timestamp as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    """Read a JSON document from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_bytes(path: Path, data: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> str:
    """Write stable pretty JSON atomically and return its SHA-256."""
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return _atomic_write_bytes(Path(path), data)


def write_run_manifest(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write a canonical (compact) run manifest and return its SHA-256."""
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    return _atomic_write_bytes(Path(path), data)


def check_free_space(
    path: str | Path, minimum_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
) -> int:
    """Return the free bytes at ``path`` or raise when below the threshold."""
    usage = shutil.disk_usage(str(path))
    if usage.free < minimum_bytes:
        raise RuntimeError(
            f"insufficient free space at {path}: {usage.free} bytes available, "
            f"{minimum_bytes} required"
        )
    return int(usage.free)


def run_logged(
    argv: Sequence[str],
    cwd: str | Path | None,
    env: Mapping[str, str] | None,
    log_path: str | Path,
) -> dict[str, Any]:
    """Run ``argv`` without a shell, capture output, and write one combined log.

    ``argv`` must be a sequence of arguments; a shell string is rejected so no
    caller can accidentally reintroduce ``shell=True`` semantics.
    """
    if isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments, not a shell string")
    command = [str(argument) for argument in argv]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except OSError as exc:
        # Report a missing/unexecutable command like a shell would (127) so
        # callers can persist a failure manifest instead of crashing.
        returncode = 127
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
    duration_s = time.monotonic() - started

    rendered = " ".join(shlex.quote(argument) for argument in command)
    log_text = (
        f"$ {rendered}\n"
        f"# cwd: {cwd}\n"
        f"# returncode: {returncode}\n"
        f"# duration_s: {duration_s:.3f}\n"
        "--- stdout ---\n"
        f"{stdout}\n"
        "--- stderr ---\n"
        f"{stderr}\n"
    )
    log_sha256 = _atomic_write_bytes(log_path, log_text.encode("utf-8"))

    return {
        "argv": command,
        "cwd": str(cwd) if cwd is not None else None,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "duration_s": duration_s,
        "log_path": str(log_path),
        "log_sha256": log_sha256,
    }


def method_run_dir(cfg: dict, method: str, run_id: str) -> Path:
    """Return ``<artifacts_root>/runs/<method>-<run_id>``."""
    artifacts_root = Path(cfg["paths"]["artifacts_root"])
    return artifacts_root / "runs" / f"{method}-{run_id}"


def seed_launcher_source(seed: int, train_script: str = "train.py") -> str:
    """Return a ``runpy`` launcher that seeds Python/NumPy/Torch first.

    The caller writes this file next to the pinned upstream ``train.py`` and
    invokes it with the same CLI arguments the trainer expects.
    """
    return (
        '"""Seeded launcher generated by reconstruct.gaussians.run_utils."""\n'
        "\n"
        "import random\n"
        "import runpy\n"
        "import sys\n"
        "\n"
        "import numpy as np\n"
        "import torch\n"
        "\n"
        f"SEED = {int(seed)}\n"
        f"TRAIN_SCRIPT = {train_script!r}\n"
        "\n"
        "random.seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "torch.manual_seed(SEED)\n"
        "if torch.cuda.is_available():\n"
        "    torch.cuda.manual_seed_all(SEED)\n"
        "\n"
        "sys.argv = [TRAIN_SCRIPT] + sys.argv[1:]\n"
        'runpy.run_path(TRAIN_SCRIPT, run_name="__main__")\n'
    )


__all__ = [
    "DEFAULT_MINIMUM_FREE_BYTES",
    "check_free_space",
    "method_run_dir",
    "read_json",
    "run_logged",
    "seed_launcher_source",
    "sha256_file",
    "utc_now",
    "write_json_atomic",
    "write_run_manifest",
]
