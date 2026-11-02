"""Canonical ingest tests: W/metre envelope, exact artifacts, loud failure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.mesh.reconstruct import manifest, meshio, validate
from tools.mesh.reconstruct.ingest import IngestError, ingest

TETRA_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
TETRA_FACES = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]
)
FRAMES = (4615, 4616)


def _require_cli() -> None:
    try:
        validate.cli_path()
    except RuntimeError as error:  # pragma: no cover - environment dependent
        pytest.skip(str(error))


def _write_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    geometry = root / "geometry"
    geometry.mkdir(parents=True)
    entries = []
    for index, frame in enumerate(FRAMES):
        entries.append(
            {
                "frame": frame,
                "T_W_image_00_rect": [
                    [1.0, 0.0, 0.0, float(index) * 10.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (geometry / "cameras.json").write_text(
        json.dumps({"frames": entries}), encoding="utf-8"
    )
    return root


def _cfg(bundle: Path, *, max_triangles: int = 1_000_000) -> dict:
    return {
        "route": "colmap_openmvs",
        "bundle_root": str(bundle),
        "out_root": str(bundle),
        "params": {
            "image_camera": "image_00",
            "cleanup": {"min_component_triangles": 1},
            "viewer": {"max_triangles": max_triangles},
        },
    }


def _route_result(tmp_path: Path, vertices: np.ndarray, faces: np.ndarray) -> dict:
    mesh_path = tmp_path / "route.ply"
    meshio.write_ply(mesh_path, vertices, faces)
    return {
        "route": "colmap_openmvs",
        "mesh_path": str(mesh_path),
        "dense_cloud_path": None,
        "config_sha256": "a" * 64,
        "descriptor": {
            "route": "colmap_openmvs",
            "config_sha256": "a" * 64,
            "tool_versions": {"openmvs": "2.4.0"},
            "dense_backend": "colmap_cuda",
        },
        "report": {},
    }


def _run(
    tmp_path: Path,
    bundle: Path,
    route: dict,
    *,
    max_triangles: int = 1_000_000,
) -> dict:
    return ingest(
        route,
        str(bundle),
        "smoke",
        list(FRAMES),
        _cfg(bundle, max_triangles=max_triangles),
        str(tmp_path / "work"),
        {},
    )


def _grid_mesh(size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.arange(size, dtype=np.float64)
    grid = np.array(
        [(x, y, 0.0) for y in coordinates for x in coordinates], dtype=np.float64
    )
    faces = []
    for row in range(size - 1):
        for column in range(size - 1):
            lower = row * size + column
            upper = lower + size
            faces.append([lower, lower + 1, upper + 1])
            faces.append([lower, upper + 1, upper])
    return grid, np.asarray(faces, dtype=np.int64)


def _ply_header_length(data: bytes) -> int:
    marker = b"end_header\n"
    return data.index(marker) + len(marker)


def _write_empty_ply(path: Path, vertices: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment frame W\n"
        "comment units m\n"
        f"element vertex {vertices.shape[0]}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    path.write_bytes(
        header.encode("ascii")
        + np.ascontiguousarray(vertices, dtype="<f8").tobytes()
    )


def test_tetrahedron_ingests_into_a_validated_canonical_artifact(
    tmp_path: Path,
) -> None:
    _require_cli()
    bundle = _write_bundle(tmp_path)

    result = _run(tmp_path, bundle, _route_result(tmp_path, TETRA_VERTICES, TETRA_FACES))

    artifact = result["artifact"]
    canonical = Path(artifact["path"]).parent
    assert artifact["frame"] == "W"
    assert artifact["units"] == "m"
    assert artifact["fmt"] == "ply"
    assert artifact["vertex_count"] == 4
    assert artifact["triangle_count"] == 4
    assert len(artifact["sha256"]) == 64

    for name in ("mesh.ply", "mesh.glb", "mesh.obj", "mesh_viewer.glb",
                 "mesh_manifest.json", "SHA256SUMS", "ingest_report.json"):
        assert (canonical / name).is_file(), name

    built = manifest.read_manifest(result["manifest_path"])
    ply_vertices, ply_triangles, _, _ = meshio.read_back_counts(
        canonical / "mesh.ply", "ply"
    )
    assert built["frame"] == "W"
    assert built["units"] == "m"
    assert built["vertex_count"] == ply_vertices == 4
    assert built["triangle_count"] == ply_triangles == 4
    assert built["decimation"]["canonical_decimated"] is False
    assert built["decimation"]["viewer"]["applied"] is False
    assert built["viewer"]["enabled"] is True
    assert built["viewer"]["triangle_count"] <= built["triangle_count"]
    assert built["viewer"]["path"] == "mesh_viewer.glb"

    validation = validate.validate_run(canonical, verify_hashes=True)
    assert validation["ok"] is True
    assert validation["vertices"] == 4
    assert validation["triangles"] == 4


def test_mesh_far_outside_the_camera_envelope_fails_loudly(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    shifted = TETRA_VERTICES + 100_000.0

    with pytest.raises(IngestError, match="metric W-frame mesh"):
        _run(tmp_path, bundle, _route_result(tmp_path, shifted, TETRA_FACES))


def test_rotated_mesh_passes_without_alignment(tmp_path: Path) -> None:
    _require_cli()
    bundle = _write_bundle(tmp_path)
    # Shift inside the envelope, then rotate 90 degrees about x.  If ingest
    # aligned the mesh to the camera centres, the resulting bbox would move.
    shifted = TETRA_VERTICES + np.array([50.0, 0.0, 0.0])
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rotated = shifted @ rotation.T
    expected_min = rotated.min(axis=0)
    expected_max = rotated.max(axis=0)

    result = _run(tmp_path, bundle, _route_result(tmp_path, rotated, TETRA_FACES))

    canonical = Path(result["artifact"]["path"]).parent
    _, _, bbox_min, bbox_max = meshio.read_back_counts(canonical / "mesh.ply", "ply")
    assert bbox_min == pytest.approx(expected_min.tolist())
    assert bbox_max == pytest.approx(expected_max.tolist())


def test_empty_mesh_fails(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    empty_path = tmp_path / "empty.ply"
    _write_empty_ply(empty_path, TETRA_VERTICES)
    route = _route_result(tmp_path, TETRA_VERTICES, TETRA_FACES)
    route["mesh_path"] = str(empty_path)

    with pytest.raises(IngestError):
        _run(tmp_path, bundle, route)


def test_viewer_glb_is_decimated_while_the_canonical_is_not(tmp_path: Path) -> None:
    _require_cli()
    bundle = _write_bundle(tmp_path)
    vertices, faces = _grid_mesh(size=5)

    result = _run(
        tmp_path,
        bundle,
        _route_result(tmp_path, vertices, faces),
        max_triangles=10,
    )

    built = manifest.read_manifest(result["manifest_path"])
    assert built["triangle_count"] == 32
    assert built["decimation"]["canonical_decimated"] is False
    assert built["decimation"]["viewer"]["applied"] is True
    assert built["decimation"]["viewer"]["target_triangles"] == 10
    assert built["viewer"]["triangle_count"] <= built["triangle_count"]
    assert built["viewer"]["triangle_count"] <= 10
    canonical = Path(result["artifact"]["path"]).parent
    _, ply_triangles, _, _ = meshio.read_back_counts(canonical / "mesh.ply", "ply")
    assert ply_triangles == 32


def test_a_one_byte_mutation_after_ingest_is_exit_six(tmp_path: Path) -> None:
    _require_cli()
    bundle = _write_bundle(tmp_path)
    result = _run(tmp_path, bundle, _route_result(tmp_path, TETRA_VERTICES, TETRA_FACES))
    canonical = Path(result["artifact"]["path"]).parent
    ply_path = canonical / "mesh.ply"

    data = bytearray(ply_path.read_bytes())
    offset = _ply_header_length(bytes(data)) + 4 * 3 * 8 + 1
    assert data[offset] == 0
    data[offset] = 1
    ply_path.write_bytes(data)

    with pytest.raises(validate.MeshValidationError) as excinfo:
        validate.validate_run(canonical, verify_hashes=True)
    assert excinfo.value.exit_code == 6


def test_no_alignment_or_scale_helpers_are_reachable() -> None:
    source = Path(__file__).resolve().parents[1] / "ingest.py"
    text = source.read_text(encoding="utf-8").lower()

    for forbidden in ("umeyama", "apply_transform", "icp", "scale("):
        assert forbidden not in text
