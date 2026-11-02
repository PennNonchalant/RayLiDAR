"""Phase-4 ``mesh_manifest.json`` builder, writer, and schema validator.

The manifest is the handoff contract between reconstruction and the C++ ingest
gate.  Every hash, byte count, vertex/triangle count, and bounding box in it is
copied from the files that were actually written (``meshio.write_all`` already
re-reads and hashes each file), never from a caller's claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import meshio

SCHEMA_VERSION = 1
KIND = "mesh"
FRAME = "W"
UNITS = "m"

DEFAULT_CONFIG_PATH = (
    "tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_FORMATS = ("ply", "glb", "obj")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if path.is_absolute() or path.anchor:
        return False
    return all(part not in ("..", "") for part in path.parts)


def _normalise_file(fmt: str, entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"files.{fmt} must be a mapping")
    relative = entry.get("path")
    if relative is None and entry.get("absolute_path"):
        relative = Path(entry["absolute_path"]).name
    if not _is_safe_relative(relative):
        raise ValueError(f"files.{fmt}.path must be a safe relative path")
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_RE.match(sha256):
        raise ValueError(f"files.{fmt}.sha256 must be a lowercase 64-hex digest")
    vertex_count = entry.get("vertex_count")
    triangle_count = entry.get("triangle_count")
    if not isinstance(vertex_count, int) or vertex_count <= 0:
        raise ValueError(f"files.{fmt}.vertex_count must be positive")
    if not isinstance(triangle_count, int) or triangle_count <= 0:
        raise ValueError(f"files.{fmt}.triangle_count must be positive")
    size = entry.get("bytes")
    if not isinstance(size, int) or size <= 0:
        size = (Path(entry["absolute_path"]).stat().st_size if entry.get("absolute_path") else 0)
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"files.{fmt}.bytes must be positive")
    return {
        "path": relative,
        "sha256": sha256,
        "bytes": int(size),
        "format": fmt,
        "vertex_count": int(vertex_count),
        "triangle_count": int(triangle_count),
    }


def _normalise_viewer(run: dict[str, Any]) -> dict[str, Any]:
    viewer = run.get("viewer") or {}
    if not isinstance(viewer, dict):
        raise ValueError("viewer must be a mapping")
    enabled = bool(viewer.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    relative = viewer.get("path")
    absolute = viewer.get("absolute_path")
    if relative is None and absolute:
        relative = Path(absolute).name
    if not _is_safe_relative(relative):
        raise ValueError("viewer.path must be a safe relative path")

    result: dict[str, Any] = {"enabled": True, "path": relative}
    candidate: Path | None = None
    if absolute:
        candidate = Path(absolute)
    elif run.get("run_dir"):
        candidate = Path(run["run_dir"]) / relative
    if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".glb":
        vertex_count, triangle_count, _, _ = meshio.read_back_counts(candidate, "glb")
        result["vertex_count"] = int(vertex_count)
        result["triangle_count"] = int(triangle_count)
        result["sha256"] = _sha256_file(candidate)
    else:
        vertex_count = viewer.get("vertex_count")
        triangle_count = viewer.get("triangle_count")
        sha256 = viewer.get("sha256")
        if not isinstance(vertex_count, int) or vertex_count <= 0:
            raise ValueError("viewer.vertex_count must be positive")
        if not isinstance(triangle_count, int) or triangle_count <= 0:
            raise ValueError("viewer.triangle_count must be positive")
        if not isinstance(sha256, str) or not _SHA256_RE.match(sha256):
            raise ValueError("viewer.sha256 must be a lowercase 64-hex digest")
        result["vertex_count"] = int(vertex_count)
        result["triangle_count"] = int(triangle_count)
        result["sha256"] = sha256
    return result


def build_manifest(run: dict[str, Any]) -> dict[str, Any]:
    """Build the frozen Phase-4 manifest from the written artifacts in ``run``.

    Required inputs: ``route``, ``config_sha256``, and ``files`` (the
    ``write_all`` result).  Canonical counts and bbox always come from the PLY
    entry; GLB/OBJ are recorded as their own counts.
    """
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    route = run.get("route")
    if not isinstance(route, str) or not route:
        raise ValueError("route must be a non-empty string")
    files_in = run.get("files")
    if not isinstance(files_in, dict):
        raise ValueError("files is required (use meshio.write_all)")
    files = {}
    for fmt in _FILE_FORMATS:
        if fmt not in files_in:
            raise ValueError(f"files.{fmt} is required")
        files[fmt] = _normalise_file(fmt, files_in[fmt])

    config_path = run.get("config_path") or DEFAULT_CONFIG_PATH
    config_sha256 = run.get("config_sha256")
    if config_sha256 is None:
        candidate = Path(config_path)
        if candidate.is_file():
            config_sha256 = _sha256_file(candidate)
    if not isinstance(config_sha256, str) or not _SHA256_RE.match(config_sha256):
        raise ValueError("config_sha256 must be a lowercase 64-hex digest")

    ply = files["ply"]
    raw_ply = files_in["ply"]
    # write_all re-reads each file, so counts/bbox are already the bytes on
    # disk.  When the run directory is known, re-read the PLY once more to make
    # the manifest independent of any caller dict.
    vertex_count = ply["vertex_count"]
    triangle_count = ply["triangle_count"]
    bbox_min = raw_ply.get("bbox_min_m")
    bbox_max = raw_ply.get("bbox_max_m")
    candidate: Path | None = None
    if run.get("ply_path"):
        candidate = Path(run["ply_path"])
    elif run.get("run_dir"):
        candidate = Path(run["run_dir"]) / ply["path"]
    if candidate is not None and candidate.is_file():
        vertex_count, triangle_count, bbox_min, bbox_max = meshio.read_back_counts(
            candidate, "ply"
        )
    if (
        not isinstance(bbox_min, (list, tuple))
        or not isinstance(bbox_max, (list, tuple))
        or len(bbox_min) != 3
        or len(bbox_max) != 3
    ):
        raise ValueError(
            "files.ply must record bbox_min_m/bbox_max_m (use meshio.write_all)"
        )
    bbox_min = [float(value) for value in bbox_min]
    bbox_max = [float(value) for value in bbox_max]

    components = run.get("components")
    if components is None:
        components = {"count": 1, "largest_fraction": 1.0}
    degenerate = run.get("degenerate")
    if degenerate is None:
        degenerate = {"count": 0, "fraction": 0.0}
    decimation = run.get("decimation")
    if decimation is None:
        decimation = {"canonical_decimated": False, "method": "none"}

    inputs: dict[str, Any] = {
        "bundle_root": "",
        "frames": [],
        "image_camera": "image_00",
        "source_mesh_sha256": "",
        "dense_cloud_sha256": "",
    }
    inputs.update(run.get("inputs") or {})

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_at_utc": run.get("created_at_utc") or _utc_now(),
        "route": route,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "frame": FRAME,
        "units": UNITS,
        "files": files,
        "vertex_count": int(vertex_count),
        "triangle_count": int(triangle_count),
        "bbox_min_m": bbox_min,
        "bbox_max_m": bbox_max,
        "components": dict(components),
        "degenerate": dict(degenerate),
        "decimation": decimation,
        "viewer": _normalise_viewer(run),
        "inputs": inputs,
        "dense_backend": run.get("dense_backend", ""),
        "tool_versions": dict(run.get("tool_versions") or {}),
        "git_sha": run.get("git_sha", ""),
        "g3": run.get("g3") or {"pass": False, "checks": {}},
    }


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Check the manifest schema without reading or hashing any file."""
    errors: list[str] = []

    def fail(message: str) -> None:
        errors.append(message)

    if not isinstance(manifest, dict):
        return ["manifest must be a mapping"]

    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("schema_version must be 1")
    if manifest.get("kind") != KIND:
        fail("kind must be 'mesh'")
    if manifest.get("frame") != FRAME:
        fail("frame must be 'W'")
    if manifest.get("units") != UNITS:
        fail("units must be 'm'")
    if not isinstance(manifest.get("route"), str) or not manifest.get("route"):
        fail("route must be a non-empty string")
    if not isinstance(manifest.get("config_sha256"), str) or not _SHA256_RE.match(
        manifest["config_sha256"]
    ):
        fail("config_sha256 must be a lowercase 64-hex digest")

    files = manifest.get("files")
    if not isinstance(files, dict):
        fail("files must be a mapping")
    else:
        for fmt in _FILE_FORMATS:
            entry = files.get(fmt)
            if not isinstance(entry, dict):
                fail(f"files.{fmt} is required")
                continue
            if not _is_safe_relative(entry.get("path")):
                fail(f"files.{fmt}.path must be a safe relative path")
            if not isinstance(entry.get("sha256"), str) or not _SHA256_RE.match(
                entry["sha256"]
            ):
                fail(f"files.{fmt}.sha256 must be a lowercase 64-hex digest")
            if not isinstance(entry.get("bytes"), int) or entry.get("bytes") <= 0:
                fail(f"files.{fmt}.bytes must be positive")
            if (
                not isinstance(entry.get("vertex_count"), int)
                or entry.get("vertex_count") <= 0
            ):
                fail(f"files.{fmt}.vertex_count must be positive")
            if (
                not isinstance(entry.get("triangle_count"), int)
                or entry.get("triangle_count") <= 0
            ):
                fail(f"files.{fmt}.triangle_count must be positive")

    vertex_count = manifest.get("vertex_count")
    triangle_count = manifest.get("triangle_count")
    if not isinstance(vertex_count, int) or vertex_count <= 0:
        fail("vertex_count must be positive")
    if not isinstance(triangle_count, int) or triangle_count <= 0:
        fail("triangle_count must be positive")
    if isinstance(files, dict) and isinstance(files.get("ply"), dict):
        ply = files["ply"]
        if (
            isinstance(vertex_count, int)
            and isinstance(ply.get("vertex_count"), int)
            and ply["vertex_count"] != vertex_count
        ):
            fail("canonical vertex_count must equal files.ply.vertex_count")
        if (
            isinstance(triangle_count, int)
            and isinstance(ply.get("triangle_count"), int)
            and ply["triangle_count"] != triangle_count
        ):
            fail("canonical triangle_count must equal files.ply.triangle_count")

    for key in ("bbox_min_m", "bbox_max_m"):
        value = manifest.get(key)
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or not all(isinstance(item, (int, float)) for item in value)
        ):
            fail(f"{key} must hold three numbers")
    if not errors:
        for low, high in zip(manifest["bbox_min_m"], manifest["bbox_max_m"]):
            if low > high:
                fail("bbox_min_m must not exceed bbox_max_m")
                break

    components = manifest.get("components")
    if not isinstance(components, dict):
        fail("components must be a mapping")
    else:
        if not isinstance(components.get("count"), int) or components.get("count") < 1:
            fail("components.count must be at least 1")
        fraction = components.get("largest_fraction")
        if not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
            fail("components.largest_fraction must be in [0, 1]")

    degenerate = manifest.get("degenerate")
    if not isinstance(degenerate, dict):
        fail("degenerate must be a mapping")
    else:
        if not isinstance(degenerate.get("count"), int) or degenerate.get("count") < 0:
            fail("degenerate.count must be non-negative")
        fraction = degenerate.get("fraction")
        if not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
            fail("degenerate.fraction must be in [0, 1]")

    viewer = manifest.get("viewer")
    if not isinstance(viewer, dict):
        fail("viewer must be a mapping")
    else:
        enabled = viewer.get("enabled")
        if not isinstance(enabled, bool):
            fail("viewer.enabled must be a boolean")
        elif enabled:
            if not _is_safe_relative(viewer.get("path")):
                fail("viewer.path must be a safe relative path")
            if not isinstance(viewer.get("sha256"), str) or not _SHA256_RE.match(
                viewer["sha256"]
            ):
                fail("viewer.sha256 must be a lowercase 64-hex digest")
            for key in ("vertex_count", "triangle_count"):
                value = viewer.get(key)
                if not isinstance(value, int) or value <= 0:
                    fail(f"viewer.{key} must be positive")

    return errors


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``mesh_manifest.json`` and a ``sha256sum -c`` compatible sidecar."""
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid mesh manifest: " + "; ".join(errors))

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "mesh_manifest.json"
    text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    _atomic_write_text(manifest_path, text)

    recorded: dict[str, str] = {}
    for fmt in _FILE_FORMATS:
        entry = manifest["files"][fmt]
        recorded[entry["path"]] = entry["sha256"]
    if manifest.get("viewer", {}).get("enabled"):
        recorded[manifest["viewer"]["path"]] = manifest["viewer"]["sha256"]
    recorded[manifest_path.name] = _sha256_file(manifest_path)

    lines = [f"{recorded[name]}  {name}\n" for name in sorted(recorded)]
    _atomic_write_text(root / "SHA256SUMS", "".join(lines))
    return manifest_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "build_manifest",
    "write_manifest",
    "validate_manifest",
    "read_manifest",
]
