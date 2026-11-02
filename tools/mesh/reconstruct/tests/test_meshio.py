"""Round-trip tests for the frozen PLY/GLB/OBJ writers (IO-03)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tools.mesh.reconstruct import meshio

VERTICES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]
FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    version, total = struct.unpack_from("<II", data, 4)
    assert version == 2
    assert total == len(data)
    offset = 12
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            return json.loads(payload.decode("utf-8"))
    raise AssertionError("GLB has no JSON chunk")


def test_mesh_stats_returns_counts_and_bbox() -> None:
    stats = meshio.mesh_stats(VERTICES, FACES)

    assert stats == {
        "vertex_count": 4,
        "triangle_count": 4,
        "bbox_min_m": [0.0, 0.0, 0.0],
        "bbox_max_m": [1.0, 1.0, 1.0],
    }


def test_tetrahedron_round_trips_through_all_three_formats(tmp_path: Path) -> None:
    result = meshio.write_all(tmp_path, "mesh", VERTICES, FACES)

    expected = meshio.mesh_stats(VERTICES, FACES)
    assert set(result["files"]) == {"ply", "glb", "obj"}
    re_read = {}
    for fmt, entry in result["files"].items():
        assert entry["path"] == f"mesh.{fmt}"
        assert Path(entry["absolute_path"]).is_file()
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        counts = meshio.read_back_counts(entry["absolute_path"], fmt)
        re_read[fmt] = counts
        assert counts[0] == expected["vertex_count"]
        assert counts[1] == expected["triangle_count"]
        assert counts[2] == pytest.approx(expected["bbox_min_m"])
        assert counts[3] == pytest.approx(expected["bbox_max_m"])

    for counts in re_read.values():
        assert counts == re_read["ply"]
    assert result["stats"] == expected


def test_ply_is_binary_little_endian_with_frozen_comments(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    meshio.write_ply(path, VERTICES, FACES)

    data = path.read_bytes()
    assert data.startswith(b"ply\nformat binary_little_endian 1.0")
    assert b"comment frame W" in data.split(b"end_header")[0]
    assert b"comment units m" in data.split(b"end_header")[0]
    header = data[: data.index(b"end_header\n") + len(b"end_header\n")]
    body = data[len(header) :]
    assert len(body) == 4 * 3 * 8 + 4 * (1 + 4 * 3)


def test_glb_has_identity_nodes_and_w_metre_extras(tmp_path: Path) -> None:
    path = tmp_path / "mesh.glb"
    meshio.write_glb(path, VERTICES, FACES)

    document = _read_glb_json(path)
    assert document["asset"]["version"] == "2.0"
    assert document["asset"]["extras"]["frame"] == "W"
    assert document["asset"]["extras"]["units"] == "m"
    for node in document["nodes"]:
        for key in ("matrix", "translation", "rotation", "scale"):
            assert key not in node

    primitive = document["meshes"][0]["primitives"][0]
    assert primitive["mode"] == 4
    position = document["accessors"][primitive["attributes"]["POSITION"]]
    assert position["componentType"] == 5126
    assert position["type"] == "VEC3"
    assert position["min"] == [0.0, 0.0, 0.0]
    assert position["max"] == [1.0, 1.0, 1.0]

    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteLength"] % 4 == 0


def test_glb_rejects_extras_that_override_frame_or_units(tmp_path: Path) -> None:
    with pytest.raises(meshio.MeshFormatError):
        meshio.write_glb(
            tmp_path / "bad.glb", VERTICES, FACES, extras={"frame": "C"}
        )


def test_obj_starts_with_comments_and_has_only_triangles(tmp_path: Path) -> None:
    path = tmp_path / "mesh.obj"
    meshio.write_obj(path, VERTICES, FACES)

    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[:2] == ["# frame W", "# units m"]
    face_lines = [line for line in lines if line.startswith("f ")]
    assert len(face_lines) == len(FACES)
    for line in face_lines:
        indices = line.split()[1:]
        assert len(indices) == 3
        assert all(int(value) >= 1 for value in indices)
    assert not any(line.startswith(("o ", "g ", "vp ", "l ", "p ")) for line in lines)


def test_non_finite_vertex_is_rejected(tmp_path: Path) -> None:
    bad = list(VERTICES)
    bad[0] = (float("nan"), 0.0, 0.0)
    with pytest.raises(meshio.MeshFormatError):
        meshio.write_ply(tmp_path / "nan.ply", bad, FACES)


def test_quad_face_array_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(meshio.MeshFormatError):
        meshio.write_obj(tmp_path / "quad.obj", VERTICES, [(0, 1, 2, 3)])


def test_out_of_range_face_index_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(meshio.MeshFormatError):
        meshio.write_glb(tmp_path / "range.glb", VERTICES, [(0, 1, 4)])
