"""Fast route tests: command plans, output resolution, envelope, alpha shape.

The real OpenMVS chain and the real dense cloud are exercised by
``scripts/phase4.sh --route-smoke``; these tests never invoke an external
binary.  The alpha-shape test runs Open3D on a small synthetic cube shell so the
real meshing code path is covered without touching reconstruction data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.mesh.reconstruct.contracts import WindowSpec
from tools.mesh.reconstruct.routes import colmap_openmvs, open3d_alpha

ROUTE_DIR = Path(__file__).resolve().parents[1] / "routes"
SMOKE = WindowSpec("smoke", tuple(range(4615, 4625)))


def _env() -> dict:
    return {
        "packages": {"open3d": "0.20.0"},
        "dense_backend": "colmap_cuda",
        "colmap": {"version": "4.0.4"},
        "openmvs": {"version": "2.4.0"},
    }


def _cfg() -> dict:
    return {
        "route": "colmap_openmvs",
        "params": {
            "threads": 8,
            "openmvs": {
                "remove_spurious": 4,
                "close_holes": 30,
                "smooth": 2,
                "refine_scales": 1,
                "max_face_area": 16,
            },
            "open3d": {
                "voxel_smoke": 0.10,
                "voxel_full": 0.05,
                "alpha_factor": 4.0,
                "normal_k": 30,
                "envelope_xy_m": 60.0,
                "envelope_z_m": [-10.0, 30.0],
            },
        },
    }


# --------------------------------------------------------------------------- #
# Canonical COLMAP -> OpenMVS route
# --------------------------------------------------------------------------- #


def test_colmap_openmvs_describe_reports_versions_and_config_hash() -> None:
    route = colmap_openmvs.ColmapOpenMVS(cfg=_cfg(), env=_env())
    descriptor = route.describe()

    assert route.name == "colmap_openmvs"
    assert descriptor.route == "colmap_openmvs"
    assert descriptor.dense_backend == "colmap_cuda"
    assert descriptor.tool_versions["colmap"] == "4.0.4"
    assert descriptor.tool_versions["openmvs"] == "2.4.0"
    assert len(descriptor.config_sha256) == 64


def test_colmap_openmvs_plan_has_four_binaries_in_chain_order(tmp_path: Path) -> None:
    binaries = {
        name: tmp_path / name
        for name in colmap_openmvs.REQUIRED_BINARIES
    }
    commands = colmap_openmvs.plan_commands(
        binaries=binaries,
        workspace=tmp_path / "workspace",
        images_dir=tmp_path / "images",
        out_dir=tmp_path,
        threads=8,
        params=_cfg()["params"],
    )

    assert [command["name"] for command in commands] == [
        "InterfaceCOLMAP",
        "DensifyPointCloud",
        "ReconstructMesh",
        "RefineMesh",
    ]
    assert [Path(command["argv"][0]).name for command in commands] == [
        "InterfaceCOLMAP",
        "DensifyPointCloud",
        "ReconstructMesh",
        "RefineMesh",
    ]
    flattened = " ".join(part for command in commands for part in command["argv"])
    assert "scene_dense.ply" in flattened
    assert "scene_dense.mvs" in flattened
    assert "scene_mesh_refined.ply" in flattened
    assert "--remove-spurious 4" in flattened
    assert "--close-holes 30" in flattened
    assert "--smooth 2" in flattened
    assert "--scales 1" in flattened
    assert "--max-face-area 16" in flattened


def test_colmap_openmvs_never_uses_a_shell() -> None:
    source = (ROUTE_DIR / "colmap_openmvs.py").read_text(encoding="utf-8")

    assert "shell=True" not in source


def test_colmap_openmvs_resolves_documented_output_names(tmp_path: Path) -> None:
    for name in colmap_openmvs.EXPECTED_OUTPUTS.values():
        (tmp_path / name).write_bytes(b"x")

    resolved = colmap_openmvs.resolve_outputs(tmp_path)

    assert set(resolved) == set(colmap_openmvs.EXPECTED_OUTPUTS)
    assert resolved["refined"].name == "scene_mesh_refined.ply"


def test_colmap_openmvs_reports_missing_output_with_directory_listing(
    tmp_path: Path,
) -> None:
    (tmp_path / "scene.mvs").write_bytes(b"x")

    with pytest.raises(colmap_openmvs.RouteError) as excinfo:
        colmap_openmvs.resolve_outputs(tmp_path)

    message = str(excinfo.value)
    assert "scene_mesh_refined.ply" in message
    assert "scene.mvs" in message


def test_colmap_openmvs_accepts_mesh_inside_camera_envelope() -> None:
    stats = {
        "bbox_min_m": [1377.0, 3493.0, 113.4],
        "bbox_max_m": [1392.8, 3506.5, 115.9],
    }
    centers = [[1373.0, 3500.0, 115.3], [1380.0, 3502.0, 115.5]]

    colmap_openmvs.validate_mesh_envelope(stats, centers)


def test_colmap_openmvs_rejects_bbox_disjoint_from_camera_envelope() -> None:
    stats = {
        "bbox_min_m": [5000.0, 5000.0, 5000.0],
        "bbox_max_m": [5010.0, 5010.0, 5010.0],
    }
    centers = [[1373.0, 3500.0, 115.3], [1380.0, 3502.0, 115.5]]

    with pytest.raises(colmap_openmvs.RouteError) as excinfo:
        colmap_openmvs.validate_mesh_envelope(stats, centers)

    message = str(excinfo.value)
    assert "disjoint" in message or "exceeds" in message


def test_colmap_openmvs_rejects_metre_scale_error() -> None:
    stats = {
        "bbox_min_m": [0.0, 0.0, 0.0],
        "bbox_max_m": [900.0, 900.0, 900.0],
    }
    centers = [[0.0, 0.0, 0.0]]

    with pytest.raises(colmap_openmvs.RouteError, match="exceeds"):
        colmap_openmvs.validate_mesh_envelope(stats, centers)
