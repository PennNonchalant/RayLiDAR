"""Deterministic G3 unit tests on synthetic meshes and synthetic frozen artifacts.

No real dataset is touched: the tests write tiny ``RLGT``/``RLRY`` files with the
documented 64/32-byte and 32/64-byte layouts and small PLY meshes, so the
decoders, the transform math, and every threshold are exercised in isolation.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from synthetic import FRAMES, make_bundle
from tools.mesh.reconstruct import cameras, cleanup, g3

SMOKE = list(range(4615, 4625))

# Independent mirrors of the frozen layouts (not the module's private dtypes).
GT_DTYPE = np.dtype(
    {
        "names": ["p_L", "r_gt", "frame", "semantic_id", "camera", "valid", "pad"],
        "formats": [("<f4", (3,)), "<f4", "<u4", "<u2", "u1", "u1", "V8"],
        "offsets": [0, 12, 16, 20, 22, 23, 24],
        "itemsize": 32,
    }
)
RAY_DTYPE = np.dtype(
    {
        "names": [
            "ray_id",
            "dir_C",
            "t_gt",
            "p_L",
            "p_C",
            "bucket",
            "occluded",
            "camera",
            "semantic_id",
            "pad",
        ],
        "formats": [
            "<u8",
            ("<f4", (3,)),
            "<f4",
            ("<f4", (3,)),
            ("<f4", (3,)),
            "u1",
            "u1",
            "u1",
            "<u2",
            "V11",
        ],
        "offsets": [0, 8, 20, 24, 36, 48, 49, 50, 51, 53],
        "itemsize": 64,
    }
)


# --------------------------------------------------------------------------- #
# Synthetic frozen-artifact writers
# --------------------------------------------------------------------------- #


def write_gt_frame(
    path: Path,
    frame: int,
    points: np.ndarray,
    *,
    magic: bytes = b"RLGT",
    record_size: int = 32,
    length_override: int | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    header = bytearray(64)
    header[0:4] = magic
    struct.pack_into("<HHII", header, 4, 1, record_size, points.shape[0], frame)
    records = np.zeros(points.shape[0], dtype=GT_DTYPE)
    records["p_L"] = points
    records["frame"] = np.uint32(frame)
    records["valid"] = 1
    payload = bytes(header) + records.tobytes()
    if length_override is not None:
        payload = payload[:length_override]
    path.write_bytes(payload)
    return path


def write_ray_frame(
    path: Path,
    frame: int,
    directions: np.ndarray,
    t_gt: np.ndarray,
    *,
    camera: int = 0,
    magic: bytes = b"RLRY",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
    t_gt = np.asarray(t_gt, dtype=np.float32).reshape(-1)
    header = bytearray(32)
    header[0:4] = magic
    struct.pack_into("<HHII", header, 4, 1, 64, directions.shape[0], frame)
    records = np.zeros(directions.shape[0], dtype=RAY_DTYPE)
    records["ray_id"] = np.arange(directions.shape[0], dtype=np.uint64) | (
        np.uint64(frame) << np.uint64(32)
    )
    records["dir_C"] = directions
    records["t_gt"] = t_gt
    records["bucket"] = 0
    records["camera"] = camera
    path.write_bytes(bytes(header) + records.tobytes())
    return path


def write_plane(
    path: Path,
    *,
    z: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> Path:
    vertices = np.asarray(
        [
            [x_range[0], y_range[0], z],
            [x_range[1], y_range[0], z],
            [x_range[1], y_range[1], z],
            [x_range[0], y_range[1], z],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    write_mesh(path, vertices, faces)
    return path


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
        str(path), file_type="ply"
    )
    return path


def tetrahedra(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    vertices = np.vstack([base, base + np.asarray([offset, 0.0, 0.0])])
    return vertices, np.vstack([faces, faces + 4])


def cube_with_duplicate_face() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.int64,
    )
    # One repeated-index triangle: 1/13 of the pre-clean triangles (> 0.001).
    return vertices, np.vstack([faces, np.asarray([[0, 0, 1]], dtype=np.int64)])


# --------------------------------------------------------------------------- #
# bbox ratio
# --------------------------------------------------------------------------- #


def test_bbox_check_passes_on_matching_extent() -> None:
    check = g3.bbox_check([0, 0, 0], [10, 20, 30], [0, 0, 0], [10, 20, 30])

    assert check["pass"] is True
    assert check["measured"]["ratio"] == [1.0, 1.0, 1.0]


def test_bbox_check_fails_when_mesh_is_1_5x_too_large() -> None:
    check = g3.bbox_check([0, 0, 0], [15, 30, 45], [0, 0, 0], [10, 20, 30])

    assert check["pass"] is False
    assert check["measured"]["ratio"] == [1.5, 1.5, 1.5]
    assert "outside" in check["reason"]


# --------------------------------------------------------------------------- #
# GT decoding and the T_W_V extent
# --------------------------------------------------------------------------- #


def test_gt_extent_transforms_p_L_with_t_W_V(tmp_path: Path) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    frame = 4615
    samples = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [-5.0, 7.0, 2.0]], dtype=np.float64
    )
    points = np.tile(samples, (200_000, 1))  # 600k points per frame
    artifacts = tmp_path / "artifacts"
    for target in (4615, 4616):
        write_gt_frame(
            artifacts / "02" / "gt" / f"frame_{target:010d}.bin", target, points
        )

    minimum, maximum, count = g3.gt_extent(
        bundle, [4615, 4616], artifacts_root=artifacts
    )

    expected = []
    for target in (4615, 4616):
        transform = cameras.T_W_V(bundle, target)
        expected.append(samples @ transform[:3, :3].T + transform[:3, 3])
    expected_points = np.vstack(expected)
    assert count == 1_200_000
    np.testing.assert_allclose(minimum, expected_points.min(axis=0), atol=1e-4)
    np.testing.assert_allclose(maximum, expected_points.max(axis=0), atol=1e-4)


def test_gt_extent_rejects_too_few_points(tmp_path: Path) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    artifacts = tmp_path / "artifacts"
    write_gt_frame(
        artifacts / "02" / "gt" / "frame_0000004615.bin",
        4615,
        np.zeros((4, 3)),
    )

    with pytest.raises(g3.G3Error, match="points"):
        g3.gt_extent(bundle, [4615], artifacts_root=artifacts)


def test_gt_decoder_rejects_bad_magic_and_length(tmp_path: Path) -> None:
    path = write_gt_frame(
        tmp_path / "gt" / "frame_0000004615.bin",
        4615,
        np.zeros((3, 3)),
        magic=b"NOPE",
    )
    with pytest.raises(g3.G3Error, match="magic"):
        g3._read_gt_frame(path, 4615)

    path = write_gt_frame(
        tmp_path / "gt" / "frame_0000004615.bin", 4615, np.zeros((3, 3))
    )
    truncated = path.with_name("truncated.bin")
    truncated.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(g3.G3Error, match="length"):
        g3._read_gt_frame(truncated, 4615)


# --------------------------------------------------------------------------- #
# Camera-outside hits
# --------------------------------------------------------------------------- #


def test_camera_outside_forward_hit_rate_is_one_on_a_covering_plane(
    tmp_path: Path,
) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    plane = write_plane(
        tmp_path / "plane.ply",
        z=12.0,
        x_range=(4600.0, 4640.0),
        y_range=(-40.0, 40.0),
    )

    result = g3.camera_outside(plane, bundle, SMOKE)

    assert result["frames"] == len(SMOKE)
    assert result["rays"] == len(SMOKE) * 5
    assert result["forward_hit_rate"] == pytest.approx(1.0)
    assert result["near_hits"] == []
    check = g3.camera_outside_check(result)
    assert check["pass"] is False
    assert "forward hit rate" in check["reason"]


def test_camera_outside_near_hit_fails(tmp_path: Path) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    # Camera 4615 sits at z = 2.0 m; a plane at z = 2.5 m is 0.5 m ahead.
    plane = write_plane(
        tmp_path / "near.ply",
        z=2.5,
        x_range=(4550.0, 4700.0),
        y_range=(-50.0, 50.0),
    )

    result = g3.camera_outside(plane, bundle, [4615])

    assert result["near_hits"], result
    assert all(hit["t_m"] < g3.NEAR_HIT_MAX_M for hit in result["near_hits"])
    assert g3.camera_outside_check(result)["pass"] is False


def test_camera_outside_partial_coverage_passes_the_hit_rate_band(
    tmp_path: Path,
) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    # Cameras sit at x = 4615..4624; the plane stops at x = 4620 so the
    # forward rate is a strict fraction inside [0.3, 0.9], and no ray is near.
    plane = write_plane(
        tmp_path / "partial.ply",
        z=12.0,
        x_range=(4600.0, 4620.0),
        y_range=(-40.0, 40.0),
    )

    result = g3.camera_outside(plane, bundle, SMOKE)
    check = g3.camera_outside_check(result)

    assert g3.FORWARD_HIT_RATE_MIN <= result["forward_hit_rate"] <= g3.FORWARD_HIT_RATE_MAX
    assert result["near_hits"] == []
    assert check["pass"] is True


# --------------------------------------------------------------------------- #
# Components and degenerates
# --------------------------------------------------------------------------- #


def test_two_components_50_50_fail_the_95_percent_check() -> None:
    vertices, faces = tetrahedra(offset=10.0)
    stats = cleanup.clean_mesh(vertices, faces, min_component_triangles=1)["stats"]

    assert stats["components_count"] == 2
    assert stats["largest_component_fraction"] == pytest.approx(0.5)
    check = g3.components_check(stats)
    assert check["pass"] is False
    assert "largest component" in check["reason"]


def test_duplicate_index_face_exceeds_the_degenerate_threshold() -> None:
    vertices, faces = cube_with_duplicate_face()
    stats = cleanup.clean_mesh(vertices, faces, min_component_triangles=1)["stats"]

    assert stats["degenerate_removed_count"] == 1
    assert stats["degenerate_fraction"] == pytest.approx(1.0 / 13.0)
    assert stats["degenerate_fraction"] > g3.DEGENERATE_FRACTION_MAX
    check = g3.degenerate_check(stats)
    assert check["pass"] is False


# --------------------------------------------------------------------------- #
# Smoke range
# --------------------------------------------------------------------------- #


def _smoke_rays(tmp_path: Path, *, t_gt_values: np.ndarray | None = None) -> Path:
    artifacts = tmp_path / "artifacts"
    if t_gt_values is None:
        t_gt_values = np.full(1500, 3.0)
    directions = np.tile(np.asarray([0.0, 0.0, 1.0]), (t_gt_values.size, 1))
    write_ray_frame(
        artifacts / "02" / "rays" / "frame_0000004615.bin",
        4615,
        directions,
        t_gt_values,
    )
    return artifacts


def test_smoke_range_median_equals_plane_offset(tmp_path: Path) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    artifacts = _smoke_rays(tmp_path)
    plane = write_plane(
        tmp_path / "plane.ply",
        z=5.5,  # camera 4615 origin is z = 2.0, so t_pred = 3.5 and t_gt = 3.0
        x_range=(4550.0, 4700.0),
        y_range=(-50.0, 50.0),
    )

    result = g3.smoke_range(plane, bundle, [4615], artifacts_root=artifacts)

    assert result["hits"] == 1500
    assert result["median_abs_error_m"] == pytest.approx(0.5, abs=1e-9)
    assert g3.smoke_range_check(result)["pass"] is True


def test_smoke_range_reports_insufficient_hits(tmp_path: Path) -> None:
    bundle = cameras.load_bundle(make_bundle(tmp_path / "bundle"))
    artifacts = tmp_path / "artifacts"
    write_ray_frame(
        artifacts / "02" / "rays" / "frame_0000004615.bin",
        4615,
        np.tile(np.asarray([0.0, 0.0, 1.0]), (4, 1)),
        np.full(4, 3.0),
    )
    plane = write_plane(
        tmp_path / "plane.ply",
        z=5.0,
        x_range=(4550.0, 4700.0),
        y_range=(-50.0, 50.0),
    )

    result = g3.smoke_range(plane, bundle, [4615], artifacts_root=artifacts)

    assert result["hits"] == 4
    assert result["median_abs_error_m"] is None
    assert result["reason"] == "insufficient hits"
    assert g3.smoke_range_check(result)["pass"] is False


def test_ray_decoder_rejects_non_unit_direction(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    path = write_ray_frame(
        artifacts / "02" / "rays" / "frame_0000004615.bin",
        4615,
        np.tile(np.asarray([0.0, 0.0, 2.0]), (4, 1)),
        np.full(4, 3.0),
    )

    with pytest.raises(g3.G3Error, match="unit"):
        g3._read_ray_frame(path, 4615)


def test_camera_check_thresholds_are_frozen_constants() -> None:
    assert (g3.BBOX_RATIO_MIN, g3.BBOX_RATIO_MAX) == (0.8, 1.2)
    assert (g3.FORWARD_HIT_RATE_MIN, g3.FORWARD_HIT_RATE_MAX) == (0.3, 0.9)
    assert g3.NEAR_HIT_MAX_M == 1.5
    assert g3.LARGEST_COMPONENT_MIN == 0.95
    assert g3.DEGENERATE_FRACTION_MAX == 0.001
    assert g3.SMOKE_MEDIAN_ERROR_MAX_M == 1.0
