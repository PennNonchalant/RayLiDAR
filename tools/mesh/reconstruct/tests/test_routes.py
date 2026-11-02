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


# --------------------------------------------------------------------------- #
# Open3D alpha-shape route
# --------------------------------------------------------------------------- #


def _write_ascii_points(path: Path, points: np.ndarray) -> Path:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cube_shell(half: float = 5.0, spacing: float = 0.1) -> np.ndarray:
    axis = np.arange(-half, half + spacing / 2.0, spacing)
    grid = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    faces = []
    for value in (-half, half):
        faces.append(np.column_stack([np.full(grid.shape[0], value), grid]))
        faces.append(np.column_stack([grid[:, 0], np.full(grid.shape[0], value), grid[:, 1]]))
        faces.append(np.column_stack([grid[:, 0], grid[:, 1], np.full(grid.shape[0], value)]))
    return np.concatenate(faces, axis=0)


def test_open3d_envelope_excludes_points_outside_camera_aabb() -> None:
    centers = [[0.0, 0.0, 0.0], [10.0, 2.0, 0.0]]
    low, high = open3d_alpha.camera_envelope(
        centers, xy_m=60.0, z_range=(-10.0, 30.0)
    )

    points = np.asarray(
        [
            [5.0, 0.0, 0.0],  # inside
            [100.0, 0.0, 0.0],  # outside x
            [5.0, 0.0, 100.0],  # outside z
        ]
    )
    keep = open3d_alpha.crop_mask(points, low, high)

    assert keep.tolist() == [True, False, False]
    assert low[0] == -60.0 and high[0] == 70.0
    assert low[2] == -10.0 and high[2] == 30.0


def test_open3d_route_requires_a_dense_cloud(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["route"] = "open3d_alpha"
    route = open3d_alpha.Open3DAlpha(
        cfg=cfg, env=_env(), camera_centers=[[0.0, 0.0, 0.0]]
    )

    with pytest.raises(open3d_alpha.RouteError, match="dense cloud"):
        route.build(SMOKE, tmp_path, tmp_path / "missing.ply")


def test_open3d_route_builds_nonempty_mesh_on_synthetic_cloud(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["route"] = "open3d_alpha"
    cfg["params"]["open3d"]["voxel_smoke"] = 0.25
    cfg["params"]["open3d"]["alpha_factor"] = 2.0

    dense = _write_ascii_points(tmp_path / "dense.ply", _cube_shell())
    route = open3d_alpha.Open3DAlpha(
        cfg=cfg,
        env=_env(),
        camera_centers=[[0.0, 0.0, 0.0]] * len(SMOKE),
        dense_result={"dense_backend": "colmap_cuda"},
    )

    output = route.build(SMOKE, tmp_path, dense)

    assert output.mesh_path.is_file()
    report = json.loads((tmp_path / "open3d" / "open3d_report.json").read_text())
    assert report["gt_used_for_construction"] is False
    assert report["mesh"]["triangle_count"] > 0
    assert report["point_counts"]["input"] == 6 * 101 * 101
    assert report["dense_cloud_source"] == "colmap_cuda"
    assert report["params"]["alpha"] == pytest.approx(0.5)


def test_open3d_route_records_the_openmvs_fallback_source(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["route"] = "open3d_alpha"
    cfg["params"]["open3d"]["voxel_smoke"] = 0.25
    cfg["params"]["open3d"]["alpha_factor"] = 2.0

    dense = _write_ascii_points(tmp_path / "scene_dense.ply", _cube_shell())
    route = open3d_alpha.Open3DAlpha(
        cfg=cfg,
        env=_env(),
        camera_centers=[[0.0, 0.0, 0.0]] * len(SMOKE),
        dense_result={
            "dense_backend": "openmvs_densify",
            "dense_cloud_source": "openmvs_densify",
        },
    )

    route.build(SMOKE, tmp_path, dense)

    report = json.loads((tmp_path / "open3d" / "open3d_report.json").read_text())
    assert report["dense_cloud_source"] == "openmvs_densify"
    assert report["gt_used_for_construction"] is False
