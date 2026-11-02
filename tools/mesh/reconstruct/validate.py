"""Thin subprocess wrapper around the C++ mesh ingest gate.

The C++ ``raylidar_cli mesh validate`` command is the single structural
authority for Phase-4 mesh artifacts.  Python never re-implements that check;
it invokes the built binary with a fixed argument list (no shell) and raises a
loud error when the gate rejects an artifact.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRESET = "linux-debug"
CLI_TIMEOUT_S = 300


class MeshValidationError(RuntimeError):
    """The C++ gate rejected an artifact; ``exit_code`` is 4 or 6."""

    def __init__(self, exit_code: int, stderr: str, stdout: str = "") -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
        detail = (stderr or stdout or "").strip()
        super().__init__(f"mesh validation failed (exit {exit_code}): {detail}")


def cli_path() -> Path:
    """Resolve the built ``raylidar_cli`` and fail loudly when it is absent."""
    preset = os.environ.get("RAYLIDAR_PRESET", DEFAULT_PRESET)
    candidate = REPO_ROOT / "build" / preset / "raylidar_cli"
    if not candidate.is_file():
        raise RuntimeError(
            f"raylidar_cli is not built at {candidate}; run `bash scripts/build.sh` "
            f"(or set RAYLIDAR_PRESET, current preset: {preset})"
        )
    return candidate


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_S,
    )


def _parse_pairs(line: str) -> dict[str, Any]:
    """Parse ``OK key=value ...`` into a dict, splitting bbox triples."""
    if not line.startswith("OK"):
        raise MeshValidationError(4, f"unexpected CLI output: {line!r}")
    parsed: dict[str, Any] = {"ok": True, "raw": line}
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        if not key or not _:
            continue
        if key.startswith("bbox_"):
            try:
                parsed[key] = [float(part) for part in value.split(",")]
            except ValueError:
                parsed[key] = value
        elif key == "format":
            parsed[key] = value
        else:
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = value
    return parsed


def validate_run(
    run_dir: str | Path,
    manifest_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a run's manifest and its declared mesh files via the C++ gate."""
    root = Path(run_dir)
    manifest = Path(manifest_path) if manifest_path is not None else root / "mesh_manifest.json"
    argv = [
        str(cli_path()),
        "mesh",
        "validate",
        "--manifest",
        str(manifest),
        "--run-dir",
        str(root),
    ]
    if verify_hashes:
        argv.append("--verify-hashes")

    completed = _run(argv)
    if completed.returncode != 0:
        raise MeshValidationError(
            completed.returncode, completed.stderr, completed.stdout
        )
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    return _parse_pairs(line)


def validate_file(path: str | Path) -> dict[str, Any]:
    """Structurally validate one PLY/GLB/OBJ file via the C++ gate."""
    completed = _run([str(cli_path()), "mesh", "validate", "--file", str(path)])
    if completed.returncode != 0:
        raise MeshValidationError(
            completed.returncode, completed.stderr, completed.stdout
        )
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    return _parse_pairs(line)


__all__ = [
    "CLI_TIMEOUT_S",
    "MeshValidationError",
    "REPO_ROOT",
    "cli_path",
    "validate_file",
    "validate_run",
]
