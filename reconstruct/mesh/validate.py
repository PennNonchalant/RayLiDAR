"""Pure-Python structural validation of mesh artifacts.

The structural gate is implemented with the
package's own manifest and mesh readers. Structural errors raise
:class:`MeshValidationError`; nothing is silently repaired.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import manifest as manifest_module
from . import meshio

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRESET = "linux-debug"
CLI_TIMEOUT_S = 300


class MeshValidationError(RuntimeError):
    """The structural gate rejected an artifact (``exit_code`` 4 or 6)."""

    def __init__(self, exit_code: int, stderr: str, stdout: str = "") -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        self.stdout = stdout
        detail = (stderr or stdout or "").strip()
        super().__init__(f"mesh validation failed (exit {exit_code}): {detail}")


def cli_path() -> Path:
    """Return the module entry point (kept for API compatibility)."""
    return REPO_ROOT / "reconstruct" / "mesh" / "__main__.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(a: list[float], b: list[float], atol: float = 1e-6) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def validate_run(
    run_dir: str | Path,
    manifest_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a run's manifest and its declared mesh files."""
    root = Path(run_dir)
    manifest_file = Path(manifest_path) if manifest_path is not None else root / "mesh_manifest.json"
    if not manifest_file.is_file():
        raise MeshValidationError(4, f"missing mesh manifest: {manifest_file}")
    try:
        document = manifest_module.read_manifest(manifest_file)
    except Exception as exc:  # noqa: BLE001
        raise MeshValidationError(4, str(exc)) from exc

    errors = manifest_module.validate_manifest(document)
    if errors:
        raise MeshValidationError(4, "; ".join(errors))

    canonical = None
    for fmt, entry in document["files"].items():
        path = root / str(entry["path"])
        if not path.is_file():
            raise MeshValidationError(4, f"missing mesh file: {path}")
        try:
            vertices, triangles, bbox_min, bbox_max = meshio.read_back_counts(path, fmt)
        except Exception as exc:  # noqa: BLE001
            raise MeshValidationError(4, str(exc)) from exc
        if vertices != int(entry["vertex_count"]) or triangles != int(entry["triangle_count"]):
            raise MeshValidationError(4, f"mesh count mismatch: {path}")
        if not _close(bbox_min, document["bbox_min_m"]) or not _close(bbox_max, document["bbox_max_m"]):
            raise MeshValidationError(4, f"mesh bbox mismatch: {path}")
        if verify_hashes and _sha256(path) != entry["sha256"]:
            raise MeshValidationError(6, f"mesh SHA-256 mismatch: {path}")
        if fmt == "ply":
            canonical = (vertices, triangles, bbox_min, bbox_max)
    if canonical is None:
        raise MeshValidationError(4, "manifest has no PLY entry")
    vertices, triangles, bbox_min, bbox_max = canonical
    return {
        "ok": True,
        "format": "ply",
        "vertices": int(vertices),
        "triangles": int(triangles),
        "vertex_count": int(vertices),
        "triangle_count": int(triangles),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def validate_file(path: str | Path) -> dict[str, Any]:
    """Structurally validate one PLY/GLB/OBJ file."""
    destination = Path(path)
    if not destination.is_file():
        raise MeshValidationError(4, f"missing mesh file: {destination}")
    fmt = destination.suffix.lower().lstrip(".")
    if fmt not in ("ply", "glb", "obj"):
        raise MeshValidationError(4, f"unknown mesh extension: {destination.suffix}")
    try:
        vertices, triangles, bbox_min, bbox_max = meshio.read_back_counts(destination, fmt)
    except Exception as exc:  # noqa: BLE001
        raise MeshValidationError(4, str(exc)) from exc
    return {
        "ok": True,
        "format": fmt,
        "vertices": int(vertices),
        "triangles": int(triangles),
        "vertex_count": int(vertices),
        "triangle_count": int(triangles),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


__all__ = [
    "CLI_TIMEOUT_S",
    "MeshValidationError",
    "REPO_ROOT",
    "cli_path",
    "validate_file",
    "validate_run",
]


def main(argv: list[str] | None = None) -> int:
    """``python -m reconstruct.mesh.validate`` entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m reconstruct.mesh.validate")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--file", default=None)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.file:
            result = validate_file(args.file)
        else:
            result = validate_run(args.run_dir, args.manifest, verify_hashes=args.verify_hashes)
    except MeshValidationError as exc:
        print(str(exc))
        return exc.exit_code
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
