"""Unit tests for dense-backend resolution and PLY inspection (no GPU)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tools.mesh.reconstruct import dense


def _ascii_ply(path: Path, points: list[tuple[float, float, float]]) -> Path:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "end_header",
    ]
    lines += [f"{x} {y} {z} 7" for x, y, z in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _binary_ply(path: Path, points: list[tuple[float, float, float]]) -> Path:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "end_header",
            "",
        ]
    ).encode("ascii")
    body = b"".join(struct.pack("<fffB", x, y, z, 9) for x, y, z in points)
    path.write_bytes(header + body)
    return path


def test_auto_without_cuda_raises() -> None:
    with pytest.raises(dense.DenseBackendUnavailable):
        dense.resolve_dense_backend(
            {"params": {"dense_backend": "auto"}},
            {"colmap_cuda": {"has_cuda": False}},
        )


def test_auto_with_cuda_resolves() -> None:
    backend = dense.resolve_dense_backend(
        {"params": {"dense_backend": "auto"}},
        {"colmap_cuda": {"has_cuda": True, "path": "/bin/true"}},
    )
    assert backend == "colmap_cuda"


def test_explicit_colmap_cuda_requires_cuda() -> None:
    with pytest.raises(dense.DenseBackendUnavailable):
        dense.resolve_dense_backend(
            {"params": {"dense_backend": "colmap_cuda"}},
            {"colmap_cuda": {"has_cuda": False}},
        )


def test_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown dense_backend"):
        dense.resolve_dense_backend(
            {"params": {"dense_backend": "openmvs"}},
            {"colmap_cuda": {"has_cuda": True}},
        )


def test_produce_writes_blocked_json(tmp_path: Path) -> None:
    cfg = {"params": {"dense_backend": "auto", "threads": 8, "colmap": {}}}
    with pytest.raises(SystemExit) as caught:
        dense.produce(
            "/nonexistent",
            "smoke",
            list(range(4615, 4625)),
            cfg,
            tmp_path,
            {"colmap_cuda": {"has_cuda": False}},
            None,
        )
    assert caught.value.code == 3
    blocked = json.loads(
        (tmp_path / "DENSE_BACKEND_BLOCKED.json").read_text(encoding="utf-8")
    )
    assert "no CUDA patch_match_stereo" in blocked["reason"]
    assert [option["id"] for option in blocked["options"]] == [
        "openmvs_densify",
        "install_cuda_colmap",
        "drop_open3d",
    ]
    assert blocked["frames"] == list(range(4615, 4625))


def test_produce_requires_model_result(tmp_path: Path) -> None:
    cfg = {"params": {"dense_backend": "auto", "threads": 8, "colmap": {}}}
    with pytest.raises(ValueError, match="colmap_result"):
        dense.produce(
            "/nonexistent",
            "smoke",
            list(range(4615, 4625)),
            cfg,
            tmp_path,
            {"colmap_cuda": {"has_cuda": True, "path": "/bin/true"}},
            None,
        )


def test_read_ply_ascii(tmp_path: Path) -> None:
    path = _ascii_ply(tmp_path / "cloud.ply", [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    count, points = dense.read_ply_points(path)
    assert count == 2
    assert points.shape == (2, 3)
    assert points[1].tolist() == [4.0, 5.0, 6.0]


def test_read_ply_binary(tmp_path: Path) -> None:
    path = _binary_ply(tmp_path / "cloud.ply", [(-1.0, 0.0, 2.0), (3.0, 4.0, 5.0)])
    count, points = dense.read_ply_points(path)
    assert count == 2
    assert points.tolist() == [[-1.0, 0.0, 2.0], [3.0, 4.0, 5.0]]


def test_assert_fused_cloud_rejects_small_cloud(tmp_path: Path) -> None:
    path = _ascii_ply(tmp_path / "cloud.ply", [(0.0, 0.0, 0.0)])
    with pytest.raises(RuntimeError, match="fewer than"):
        dense.assert_fused_cloud(path, minimum=10_000)


def test_assert_fused_cloud_rejects_non_finite(tmp_path: Path) -> None:
    path = _ascii_ply(
        tmp_path / "cloud.ply", [(0.0, 0.0, 0.0), (float("nan"), 1.0, 2.0)]
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        dense.assert_fused_cloud(path, minimum=2)


def test_assert_fused_cloud_accepts_valid_cloud(tmp_path: Path) -> None:
    points = [(float(index), 0.0, 0.0) for index in range(5)]
    path = _ascii_ply(tmp_path / "cloud.ply", points)
    metadata = dense.assert_fused_cloud(path, minimum=5)
    assert metadata["point_count"] == 5
    assert metadata["finite_sample"] is True
