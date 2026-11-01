"""Integration tests for the Python -> C++ mesh ingest gate (IO-03)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.mesh.reconstruct import manifest, meshio, validate

VERTICES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]
FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _require_cli() -> None:
    try:
        validate.cli_path()
    except RuntimeError as error:  # pragma: no cover - environment dependent
        pytest.skip(str(error))


def _make_run(tmp_path: Path) -> dict:
    written = meshio.write_all(tmp_path, "mesh", VERTICES, FACES)
    built = manifest.build_manifest(
        {
            "route": "colmap_openmvs",
            "config_sha256": "a" * 64,
            "files": written["files"],
            "run_dir": str(tmp_path),
            "viewer": {"enabled": False},
        }
    )
    manifest.write_manifest(tmp_path, built)
    return built


def _ply_header_length(data: bytes) -> int:
    marker = b"end_header\n"
    return data.index(marker) + len(marker)


def test_validate_run_accepts_a_complete_artifact(tmp_path: Path) -> None:
    _require_cli()
    built = _make_run(tmp_path)

    result = validate.validate_run(tmp_path, verify_hashes=True)

    assert result["ok"] is True
    assert result["vertices"] == built["vertex_count"]
    assert result["triangles"] == built["triangle_count"]
    assert result["bbox_min"] == pytest.approx(built["bbox_min_m"])
    assert result["bbox_max"] == pytest.approx(built["bbox_max_m"])


def test_a_one_byte_ply_mutation_is_exit_6(tmp_path: Path) -> None:
    _require_cli()
    _make_run(tmp_path)
    path = tmp_path / "mesh.ply"
    data = bytearray(path.read_bytes())
    # First face index: 0 -> 1. Still a valid, in-range triangle; the file
    # remains structurally valid with identical counts and bbox, so only the
    # recorded hash can catch it.
    offset = _ply_header_length(bytes(data)) + 4 * 3 * 8 + 1
    assert data[offset] == 0
    data[offset] = 1
    path.write_bytes(data)

    with pytest.raises(validate.MeshValidationError) as excinfo:
        validate.validate_run(tmp_path, verify_hashes=True)
    assert excinfo.value.exit_code == 6
    assert "SHA-256" in excinfo.value.stderr


def test_a_truncated_obj_is_exit_4(tmp_path: Path) -> None:
    _require_cli()
    _make_run(tmp_path)
    (tmp_path / "mesh.obj").write_text("# frame W\n# units m\n", encoding="ascii")

    with pytest.raises(validate.MeshValidationError) as excinfo:
        validate.validate_run(tmp_path, verify_hashes=True)
    assert excinfo.value.exit_code == 4


def test_validate_file_reports_the_format(tmp_path: Path) -> None:
    _require_cli()
    _make_run(tmp_path)

    result = validate.validate_file(tmp_path / "mesh.glb")

    assert result["ok"] is True
    assert result["format"] == "glb"
    assert result["vertices"] == 4
    assert result["triangles"] == 4
