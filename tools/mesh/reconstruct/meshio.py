"""Frozen mesh writers for the Phase-4 handoff (IO-03).

The three containers written here are the only accepted export formats
(``docs/sim-io.md`` §5):

- binary little-endian PLY with ``comment frame W`` / ``comment units m``,
- glTF 2.0 GLB with W/metre ``asset.extras`` and identity node transforms,
- ASCII OBJ with the two required comments and triangle-only faces.

All positions are W-frame metres.  Nothing in this module rotates, scales, or
otherwise re-expresses the input arrays; the writers are byte-level encoders
plus read-back verification.

Only NumPy and the standard library are used, so exporting does not depend on
trimesh/Open3D being importable.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

FRAME = "W"
UNITS = "m"

# glTF component/primitive constants for the frozen subset.
_GLTF_FLOAT = 5126  # FLOAT
_GLTF_UNSIGNED_INT = 5125  # UNSIGNED_INT
_GLTF_ARRAY_BUFFER = 34962
_GLTF_ELEMENT_ARRAY_BUFFER = 34963
_GLTF_TRIANGLES = 4
_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK = 0x004E4942


class MeshFormatError(ValueError):
    """Raised when mesh arrays cannot be encoded without breaking the contract."""


def _as_vertices(vertices: Any) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise MeshFormatError("vertices must have shape (N, 3)")
    if array.shape[0] == 0:
        raise MeshFormatError("mesh must have at least one vertex")
    if not np.isfinite(array).all():
        raise MeshFormatError("vertex coordinates must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_faces(faces: Any) -> np.ndarray:
    array = np.asarray(faces)
    if array.ndim != 2 or array.shape[1] != 3:
        raise MeshFormatError("faces must be triangles with shape (M, 3)")
    if array.shape[0] == 0:
        raise MeshFormatError("mesh must have at least one triangle")
    if not np.issubdtype(array.dtype, np.integer):
        raise MeshFormatError("face indices must be integers")
    array = array.astype(np.int64, copy=False)
    if array.min() < 0:
        raise MeshFormatError("face indices must be non-negative")
    return np.ascontiguousarray(array, dtype=np.int64)


def mesh_stats(vertices: Any, faces: Any) -> dict[str, Any]:
    """Return counts and the W-frame metre bounding box of a triangle mesh."""
    vertex_array = _as_vertices(vertices)
    face_array = _as_faces(faces)
    if face_array.max() >= vertex_array.shape[0]:
        raise MeshFormatError("face index is out of range")
    bbox_min = [float(value) for value in vertex_array.min(axis=0)]
    bbox_max = [float(value) for value in vertex_array.max(axis=0)]
    return {
        "vertex_count": int(vertex_array.shape[0]),
        "triangle_count": int(face_array.shape[0]),
        "bbox_min_m": bbox_min,
        "bbox_max_m": bbox_max,
    }


def _ply_header(vertex_count: int, face_count: int) -> str:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment frame W\n"
        "comment units m\n"
        f"element vertex {vertex_count}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )


def write_ply(path: str | Path, vertices: Any, faces: Any) -> None:
    """Write a binary little-endian PLY with float64 W-frame positions."""
    vertex_array = _as_vertices(vertices)
    face_array = _as_faces(faces)
    if face_array.max() >= vertex_array.shape[0]:
        raise MeshFormatError("face index is out of range")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(_ply_header(vertex_array.shape[0], face_array.shape[0]).encode("ascii"))
        handle.write(np.ascontiguousarray(vertex_array, dtype="<f8").tobytes())
        for face in face_array:
            handle.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))


def _glb_chunk(chunk_type: int, payload: bytes, pad: bytes) -> bytes:
    padding = (-len(payload)) % 4
    padded = payload + pad * padding
    return struct.pack("<II", len(padded), chunk_type) + padded


def write_glb(path: str | Path, vertices: Any, faces: Any, *, extras: dict | None = None) -> None:
    """Write a minimal glTF 2.0 GLB with identity node transforms.

    Positions are stored as ``float32`` (componentType 5126) and indices as
    ``uint32`` (5125).  The accessor ``min``/``max`` arrays are computed from the
    stored float32 values, so the container's declared bounds are exact.
    """
    vertex_array = _as_vertices(vertices)
    face_array = _as_faces(faces)
    if face_array.max() >= vertex_array.shape[0]:
        raise MeshFormatError("face index is out of range")

    positions = np.ascontiguousarray(vertex_array, dtype="<f4")
    indices = np.ascontiguousarray(face_array.reshape(-1), dtype="<u4")
    bin_payload = positions.tobytes() + indices.tobytes()

    bbox_min = [float(value) for value in positions.min(axis=0)]
    bbox_max = [float(value) for value in positions.max(axis=0)]

    asset_extras: dict[str, Any] = {"frame": FRAME, "units": UNITS}
    if extras:
        for key in ("frame", "units"):
            if key in extras and extras[key] != asset_extras[key]:
                raise MeshFormatError(f"GLB extras.{key} must be {asset_extras[key]!r}")
        asset_extras.update(extras)

    document = {
        "asset": {
            "version": "2.0",
            "generator": "raylidar-meshio",
            "extras": asset_extras,
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # A node with no matrix/translation/rotation/scale is the identity.
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": _GLTF_TRIANGLES,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": _GLTF_FLOAT,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": bbox_min,
                "max": bbox_max,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": _GLTF_UNSIGNED_INT,
                "count": int(indices.shape[0]),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": int(positions.nbytes),
                "target": _GLTF_ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": int(positions.nbytes),
                "byteLength": int(indices.nbytes),
                "target": _GLTF_ELEMENT_ARRAY_BUFFER,
            },
        ],
        "buffers": [{"byteLength": len(bin_payload)}],
    }

    json_payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")
    json_chunk = _glb_chunk(_GLB_JSON_CHUNK, json_payload, b" ")
    bin_chunk = _glb_chunk(_GLB_BIN_CHUNK, bin_payload, b"\x00")
    total = 12 + len(json_chunk) + len(bin_chunk)
    container = struct.pack("<4sII", _GLB_MAGIC, _GLB_VERSION, total) + json_chunk + bin_chunk

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(container)


def write_obj(path: str | Path, vertices: Any, faces: Any) -> None:
    """Write an ASCII OBJ carrying only two comments, vertices, and triangles."""
    vertex_array = _as_vertices(vertices)
    face_array = _as_faces(faces)
    if face_array.max() >= vertex_array.shape[0]:
        raise MeshFormatError("face index is out of range")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# frame W\n")
        handle.write("# units m\n")
        for x, y, z in vertex_array:
            handle.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in face_array:
            handle.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def _read_ply(path: Path) -> tuple[int, int, list[float], list[float]]:
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        raise MeshFormatError(f"{path}: PLY header has no end_header")
    header_text = data[: end + len(b"end_header\n")].decode("ascii")
    header = header_text.splitlines()
    if not header or header[0].strip() != "ply":
        raise MeshFormatError(f"{path}: not a PLY file")
    if "format binary_little_endian 1.0" not in [line.strip() for line in header]:
        raise MeshFormatError(f"{path}: PLY is not binary_little_endian 1.0")

    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "element" and len(tokens) >= 3:
            in_vertex = tokens[1] == "vertex"
            if tokens[1] == "vertex":
                vertex_count = int(tokens[2])
            elif tokens[1] == "face":
                face_count = int(tokens[2])
        elif tokens[0] == "property" and in_vertex and len(tokens) >= 3:
            vertex_properties.append((tokens[2], tokens[1]))

    formats = {"double": "d", "float": "f"}
    try:
        offsets = [
            formats[property_type]
            for name, property_type in vertex_properties
            if name in ("x", "y", "z")
        ]
    except KeyError as error:  # pragma: no cover - defensive
        raise MeshFormatError(f"{path}: unsupported PLY vertex type") from error
    if len(offsets) != 3:
        raise MeshFormatError(f"{path}: PLY vertex element lacks x/y/z")

    body = data[len(header_text) :]
    vertex_struct = struct.Struct("<" + "".join(offsets))
    stride = vertex_struct.size
    if len(body) < vertex_count * stride:
        raise MeshFormatError(f"{path}: PLY vertex data is truncated")
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for index in range(vertex_count):
        x, y, z = vertex_struct.unpack_from(body, index * stride)
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
    return (
        vertex_count,
        face_count,
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    )


def _read_glb(path: Path) -> tuple[int, int, list[float], list[float]]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != _GLB_MAGIC:
        raise MeshFormatError(f"{path}: not a GLB container")
    version, total = struct.unpack_from("<II", data, 4)
    if version != _GLB_VERSION:
        raise MeshFormatError(f"{path}: GLB container version is not 2")
    if total != len(data):
        raise MeshFormatError(f"{path}: GLB length does not match the file size")

    offset = 12
    document = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + chunk_length]
        if len(payload) != chunk_length:
            raise MeshFormatError(f"{path}: GLB chunk is truncated")
        if chunk_type == _GLB_JSON_CHUNK and document is None:
            document = json.loads(payload.decode("utf-8"))
        offset += chunk_length
    if document is None:
        raise MeshFormatError(f"{path}: GLB has no JSON chunk")

    extras = document.get("asset", {}).get("extras", {})
    if extras.get("frame") != FRAME or extras.get("units") != UNITS:
        raise MeshFormatError(f"{path}: GLB asset extras are not W/metres")
    for node in document.get("nodes", []):
        for key in ("matrix", "translation", "rotation", "scale"):
            if key in node:
                raise MeshFormatError(f"{path}: GLB node carries a transform")

    accessors = document["accessors"]
    primitive = document["meshes"][0]["primitives"][0]
    position = accessors[primitive["attributes"]["POSITION"]]
    if position.get("componentType") != _GLTF_FLOAT or "min" not in position or "max" not in position:
        raise MeshFormatError(f"{path}: GLB POSITION accessor is not float32 with min/max")
    vertex_count = int(position["count"])
    index_accessor = accessors[primitive["indices"]]
    triangle_count = int(index_accessor["count"]) // 3
    return (
        vertex_count,
        triangle_count,
        [float(value) for value in position["min"]],
        [float(value) for value in position["max"]],
    )


def _read_obj(path: Path) -> tuple[int, int, list[float], list[float]]:
    vertex_count = 0
    triangle_count = 0
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    lines = path.read_text(encoding="ascii").splitlines()
    if lines[:2] != ["# frame W", "# units m"]:
        raise MeshFormatError(f"{path}: OBJ is missing the frame/units comments")
    for line in lines[2:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "v":
            if len(tokens) < 4:
                raise MeshFormatError(f"{path}: malformed OBJ vertex")
            x, y, z = (float(value) for value in tokens[1:4])
            xs.append(x)
            ys.append(y)
            zs.append(z)
            vertex_count += 1
        elif tokens[0] == "f":
            if len(tokens) != 4:
                raise MeshFormatError(f"{path}: OBJ face is not a triangle")
            for token in tokens[1:]:
                if int(token.split("/")[0]) <= 0:
                    raise MeshFormatError(f"{path}: OBJ face has a non-positive index")
            triangle_count += 1
    return (
        vertex_count,
        triangle_count,
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    )


def read_back_counts(path: str | Path, fmt: str) -> tuple[int, int, list[float], list[float]]:
    """Re-read a written mesh and return ``(vertices, triangles, bbox_min, bbox_max)``."""
    destination = Path(path)
    lowered = fmt.lower()
    if lowered == "ply":
        return _read_ply(destination)
    if lowered == "glb":
        return _read_glb(destination)
    if lowered == "obj":
        return _read_obj(destination)
    raise MeshFormatError(f"unknown mesh format: {fmt}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_all(
    out_dir: str | Path,
    basename: str,
    vertices: Any,
    faces: Any,
) -> dict[str, Any]:
    """Write the canonical PLY/GLB/OBJ triple and describe the written bytes.

    Returns ``{"files": {fmt: {...}}, "stats": {...}}``.  Each file entry records
    the path relative to ``out_dir`` (safe for a manifest), the absolute path,
    the SHA-256 of the bytes on disk, the byte size, and re-read counts.
    """
    vertex_array = _as_vertices(vertices)
    face_array = _as_faces(faces)
    stats = mesh_stats(vertex_array, face_array)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = basename or "mesh"
    writers = {
        "ply": write_ply,
        "glb": write_glb,
        "obj": write_obj,
    }

    files: dict[str, dict[str, Any]] = {}
    for fmt, writer in writers.items():
        relative = f"{prefix}.{fmt}"
        destination = root / relative
        writer(destination, vertex_array, face_array)
        vertex_count, triangle_count, bbox_min, bbox_max = read_back_counts(
            destination, fmt
        )
        files[fmt] = {
            "path": relative,
            "absolute_path": str(destination),
            "format": fmt,
            "sha256": _sha256(destination),
            "bytes": int(destination.stat().st_size),
            "vertex_count": int(vertex_count),
            "triangle_count": int(triangle_count),
            "bbox_min_m": bbox_min,
            "bbox_max_m": bbox_max,
        }

    return {"files": files, "stats": stats}


__all__ = [
    "MeshFormatError",
    "mesh_stats",
    "write_ply",
    "write_glb",
    "write_obj",
    "read_back_counts",
    "write_all",
]
