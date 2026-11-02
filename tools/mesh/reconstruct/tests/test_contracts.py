from __future__ import annotations

from pathlib import Path

import pytest

from tools.mesh.reconstruct.contracts import (
    ContractError,
    MeshArtifact,
    MeshReconstructor,
    RawRouteOutput,
    RouteDescriptor,
    WindowSpec,
)


def test_window_spec_accepts_contiguous_frames() -> None:
    window = WindowSpec("smoke", (4615, 4616, 4617))

    assert window.name == "smoke"
    assert window.frames == (4615, 4616, 4617)
    assert len(window) == 3


def test_window_spec_rejects_empty_frames() -> None:
    with pytest.raises(ContractError, match="non-empty"):
        WindowSpec("smoke", ())


def test_window_spec_rejects_non_increasing_frames() -> None:
    with pytest.raises(ContractError, match="strictly increasing"):
        WindowSpec("smoke", (4615, 4615))
    with pytest.raises(ContractError, match="strictly increasing"):
        WindowSpec("smoke", (4616, 4615))


def test_window_spec_rejects_non_contiguous_frames() -> None:
    with pytest.raises(ContractError, match="contiguous"):
        WindowSpec("smoke", (4615, 4617))


def test_mesh_artifact_defaults_are_world_metres(tmp_path: Path) -> None:
    artifact = MeshArtifact(
        path=tmp_path / "mesh.ply",
        fmt="ply",
        vertex_count=4,
        triangle_count=2,
        bbox_min_m=(-1.0, -1.0, 0.0),
        bbox_max_m=(1.0, 1.0, 2.0),
        sha256="0" * 64,
        route="colmap_openmvs",
        config_sha256="1" * 64,
        components={"count": 1},
        degenerate={"count": 0, "fraction": 0.0},
    )

    assert artifact.frame == "W"
    assert artifact.units == "m"
    assert artifact.viewer_glb is None


def test_route_descriptor_and_raw_output_are_plain_records(tmp_path: Path) -> None:
    descriptor = RouteDescriptor(
        route="open3d_alpha",
        config_sha256="a" * 64,
        tool_versions={"open3d": "0.20.0"},
        dense_backend="colmap_cuda",
    )
    raw = RawRouteOutput(mesh_path=tmp_path / "raw.ply")

    assert descriptor.tool_versions["open3d"] == "0.20.0"
    assert raw.mesh_path.name == "raw.ply"
    assert raw.dense_cloud_path is None
    assert raw.logs_path is None


def test_mesh_reconstructor_is_a_runtime_protocol() -> None:
    assert getattr(MeshReconstructor, "_is_protocol", False)
    assert getattr(MeshReconstructor, "_is_runtime_protocol", False)

    class Dummy:
        name = "dummy"

        def describe(self) -> RouteDescriptor:
            return RouteDescriptor(route="dummy", config_sha256="0" * 64)

        def build(self, window, workdir, dense_cloud=None):  # noqa: ANN001
            return Path(workdir) / "dummy.ply"

    assert isinstance(Dummy(), MeshReconstructor)
