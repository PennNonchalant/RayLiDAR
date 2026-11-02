"""Schema and byte-provenance tests for the Phase-4 mesh manifest."""

from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from pathlib import Path

from tools.mesh.reconstruct import manifest, meshio

VERTICES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]
FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
CONFIG_SHA = "a" * 64


def _run(tmp_path: Path, *, viewer: bool = True) -> dict:
    written = meshio.write_all(tmp_path, "mesh", VERTICES, FACES)
    run = {
        "route": "colmap_openmvs",
        "config_sha256": CONFIG_SHA,
        "files": written["files"],
        "run_dir": str(tmp_path),
        "inputs": {"bundle_root": "/data/bundle", "frames": [4615]},
        "dense_backend": "colmap_cuda",
        "tool_versions": {"openmvs": "2.4.0"},
        "git_sha": "b" * 40,
    }
    if viewer:
        viewer_path = tmp_path / "mesh_viewer.glb"
        meshio.write_glb(viewer_path, VERTICES, FACES)
        run["viewer"] = {
            "enabled": True,
            "path": viewer_path.name,
            "absolute_path": str(viewer_path),
        }
    return run


def test_build_manifest_covers_every_required_key(tmp_path: Path) -> None:
    built = manifest.build_manifest(_run(tmp_path))

    required = {
        "schema_version",
        "kind",
        "created_at_utc",
        "route",
        "config_path",
        "config_sha256",
        "frame",
        "units",
        "files",
        "vertex_count",
        "triangle_count",
        "bbox_min_m",
        "bbox_max_m",
        "components",
        "degenerate",
        "decimation",
        "viewer",
        "inputs",
        "dense_backend",
        "tool_versions",
        "git_sha",
        "g3",
    }
    assert required <= set(built)
    assert built["schema_version"] == 1
    assert built["kind"] == "mesh"
    assert built["frame"] == "W"
    assert built["units"] == "m"
    assert built["viewer"]["enabled"] is True
    assert built["viewer"]["vertex_count"] == 4
    assert built["viewer"]["sha256"] == hashlib.sha256(
        (tmp_path / "mesh_viewer.glb").read_bytes()
    ).hexdigest()
    assert manifest.validate_manifest(built) == []


def test_manifest_counts_come_from_the_written_ply(tmp_path: Path) -> None:
    built = manifest.build_manifest(_run(tmp_path))
    ply_count = meshio.read_back_counts(tmp_path / "mesh.ply", "ply")

    assert built["vertex_count"] == ply_count[0]
    assert built["triangle_count"] == ply_count[1]
    assert built["bbox_min_m"] == ply_count[2]
    assert built["bbox_max_m"] == ply_count[3]
    assert built["files"]["ply"]["vertex_count"] == built["vertex_count"]
    assert built["files"]["ply"]["triangle_count"] == built["triangle_count"]


def test_removing_a_required_section_is_reported(tmp_path: Path) -> None:
    built = manifest.build_manifest(_run(tmp_path))

    for key in (
        "schema_version",
        "kind",
        "frame",
        "units",
        "files",
        "components",
        "degenerate",
    ):
        broken = copy.deepcopy(built)
        del broken[key]
        errors = manifest.validate_manifest(broken)
        assert errors, f"removing {key} must be reported"


def test_ply_count_drift_is_reported(tmp_path: Path) -> None:
    built = manifest.build_manifest(_run(tmp_path))
    built["files"]["ply"]["vertex_count"] += 1

    errors = manifest.validate_manifest(built)
    assert any("canonical vertex_count" in error for error in errors)


def test_write_manifest_emits_a_sha256sum_compatible_sidecar(tmp_path: Path) -> None:
    built = manifest.build_manifest(_run(tmp_path))
    written = manifest.write_manifest(tmp_path, built)

    assert written == tmp_path / "mesh_manifest.json"
    assert manifest.read_manifest(written) == built
    sums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    names = {line.split("  ", 1)[1] for line in sums}
    assert names == {
        "mesh.ply",
        "mesh.glb",
        "mesh.obj",
        "mesh_viewer.glb",
        "mesh_manifest.json",
    }
    if shutil.which("sha256sum"):
        completed = subprocess.run(
            ["sha256sum", "-c", "SHA256SUMS"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_viewer_can_be_explicitly_disabled(tmp_path: Path) -> None:
    built = manifest.build_manifest(_run(tmp_path, viewer=False))

    assert built["viewer"] == {"enabled": False}
    assert manifest.validate_manifest(built) == []
    manifest.write_manifest(tmp_path, built)
    names = {
        line.split("  ", 1)[1]
        for line in (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert "mesh_viewer.glb" not in names


def test_build_manifest_rejects_absolute_artifact_paths(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run["files"]["ply"]["path"] = str(tmp_path / "mesh.ply")

    try:
        manifest.build_manifest(run)
    except ValueError as error:
        assert "relative" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("absolute path must be rejected")
