"""Pure-Python structural validation for exported mesh bundles.

This module is the single structural authority for a ``reconstruct`` export.
It never invokes a C++ binary or any other repository component: a manifest is
checked against its schema, and each declared PLY/GLB/OBJ is re-read from disk,
re-hashed, and re-counted before the bundle is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import manifest as manifest_module
from . import meshio

MANIFEST_FILE = "mesh_manifest.json"
SUMS_FILE = "SHA256SUMS"


class MeshValidationError(RuntimeError):
    """The exported bundle is malformed; ``errors`` lists every problem."""

    def __init__(self, errors: str | list[str]) -> None:
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MeshValidationError(f"manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MeshValidationError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MeshValidationError(f"manifest root must be a mapping: {path}")
    return payload


def _verify_sums(root: Path, manifest_path: Path, errors: list[str]) -> None:
    sums_path = root / SUMS_FILE
    if not sums_path.is_file():
        return
    for line_number, line in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            errors.append(f"{sums_path}:{line_number}: malformed checksum line")
            continue
        expected, name = parts[0], parts[1].lstrip("*")
        candidate = root / name
        if not candidate.is_file():
            errors.append(f"{sums_path}:{line_number}: missing file {name}")
            continue
        actual = _sha256_file(candidate)
        if actual != expected:
            errors.append(
                f"{sums_path}:{line_number}: {name} sha256 {actual} != {expected}"
            )
    recorded = {
        line.split(None, 1)[1].lstrip("*")
        for line in sums_path.read_text(encoding="utf-8").splitlines()
        if len(line.split()) == 2
    }
    if manifest_path.name not in recorded:
        errors.append(f"{sums_path}: does not record {manifest_path.name}")


def validate_run(
    run_dir: str | Path,
    manifest_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a run's manifest and every file it declares.

    A bundle is accepted only when the manifest schema is valid, all declared
    files exist, the recorded hashes and counts match the bytes on disk, and
    ``SHA256SUMS`` (when present) covers the manifest itself.
    """
    root = Path(run_dir)
    manifest_file = (
        Path(manifest_path) if manifest_path is not None else root / MANIFEST_FILE
    )
    payload = _read_manifest(manifest_file)

    errors = list(manifest_module.validate_manifest(payload))

    files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
    for fmt in ("ply", "glb", "obj"):
        entry = files.get(fmt)
        if not isinstance(entry, dict):
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            continue
        candidate = root / relative
        if not candidate.is_file():
            errors.append(f"files.{fmt}: missing file {candidate}")
            continue
        if verify_hashes:
            actual = _sha256_file(candidate)
            if actual != entry.get("sha256"):
                errors.append(
                    f"files.{fmt}: sha256 {actual} != {entry.get('sha256')}"
                )
        try:
            vertex_count, triangle_count, _, _ = meshio.read_back_counts(
                candidate, fmt
            )
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f"files.{fmt}: cannot re-read {candidate}: {exc}")
            continue
        if vertex_count != entry.get("vertex_count"):
            errors.append(
                f"files.{fmt}: vertex_count {vertex_count} != "
                f"{entry.get('vertex_count')}"
            )
        if triangle_count != entry.get("triangle_count"):
            errors.append(
                f"files.{fmt}: triangle_count {triangle_count} != "
                f"{entry.get('triangle_count')}"
            )

    viewer = payload.get("viewer")
    if isinstance(viewer, dict) and viewer.get("enabled"):
        relative = viewer.get("path")
        if isinstance(relative, str) and relative:
            candidate = root / relative
            if not candidate.is_file():
                errors.append(f"viewer: missing file {candidate}")
            elif verify_hashes:
                actual = _sha256_file(candidate)
                if actual != viewer.get("sha256"):
                    errors.append(
                        f"viewer: sha256 {actual} != {viewer.get('sha256')}"
                    )

    ply = files.get("ply") if isinstance(files, dict) else None
    if isinstance(ply, dict):
        if payload.get("vertex_count") != ply.get("vertex_count"):
            errors.append("vertex_count must equal files.ply.vertex_count")
        if payload.get("triangle_count") != ply.get("triangle_count"):
            errors.append("triangle_count must equal files.ply.triangle_count")

    if verify_hashes:
        _verify_sums(root, manifest_file, errors)

    if errors:
        raise MeshValidationError(errors)

    return {
        "ok": True,
        "manifest": str(manifest_file),
        "format": "ply",
        "vertex_count": int(payload.get("vertex_count") or 0),
        "triangle_count": int(payload.get("triangle_count") or 0),
    }


def validate_file(path: str | Path) -> dict[str, Any]:
    """Structurally validate one PLY/GLB/OBJ file without a manifest."""
    candidate = Path(path)
    fmt = candidate.suffix.lstrip(".").lower()
    if fmt not in ("ply", "glb", "obj"):
        raise MeshValidationError(f"unknown mesh format: {candidate.suffix!r}")
    if not candidate.is_file():
        raise MeshValidationError(f"mesh file is missing: {candidate}")
    try:
        vertex_count, triangle_count, bbox_min, bbox_max = meshio.read_back_counts(
            candidate, fmt
        )
    except (ValueError, KeyError, OSError) as exc:
        raise MeshValidationError(f"cannot read {candidate}: {exc}") from exc
    if vertex_count <= 0 or triangle_count <= 0:
        raise MeshValidationError(f"{candidate}: mesh has no vertices or triangles")
    return {
        "ok": True,
        "path": str(candidate),
        "format": fmt,
        "vertex_count": int(vertex_count),
        "triangle_count": int(triangle_count),
        "bbox_min_m": [float(value) for value in bbox_min],
        "bbox_max_m": [float(value) for value in bbox_max],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {__package__}.validate",
        description="Validate an exported mesh bundle or a single mesh file",
    )
    parser.add_argument("--manifest", help="path to mesh_manifest.json")
    parser.add_argument("--run-dir", help="directory holding the mesh bundle")
    parser.add_argument("--file", help="validate one PLY/GLB/OBJ file")
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        default=True,
        help="re-hash every declared file (default)",
    )
    parser.add_argument(
        "--no-verify-hashes",
        action="store_false",
        dest="verify_hashes",
        help="skip hash checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.file:
            result = validate_file(args.file)
        elif args.manifest:
            run_dir = Path(args.run_dir) if args.run_dir else Path(args.manifest).parent
            result = validate_run(
                run_dir, args.manifest, verify_hashes=args.verify_hashes
            )
        else:
            print("either --file or --manifest is required", file=sys.stderr)
            return 2
    except MeshValidationError as exc:
        for error in exc.errors:
            print(f"ERROR {error}", file=sys.stderr)
        return 1
    print(
        "OK "
        f"format={result.get('format')} "
        f"vertices={result.get('vertex_count')} "
        f"triangles={result.get('triangle_count')}"
    )
    return 0


__all__ = [
    "MANIFEST_FILE",
    "SUMS_FILE",
    "MeshValidationError",
    "main",
    "validate_file",
    "validate_run",
]

if __name__ == "__main__":
    raise SystemExit(main())
