"""Config-only route switching, unavailability reasons, and A/B wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.mesh.reconstruct import routes
from tools.mesh.reconstruct.config import load_config
from tools.mesh.reconstruct.contracts import RawRouteOutput, WindowSpec

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SMOKE = WindowSpec("smoke", tuple(range(4615, 4625)))
ENV = {"packages": {"open3d": "0.20.0"}, "dense_backend": "colmap_cuda"}


@pytest.mark.parametrize(
    ("config_name", "route", "class_name"),
    [
        ("canonical_colmap_openmvs", "colmap_openmvs", "ColmapOpenMVS"),
        ("open3d_alpha", "open3d_alpha", "Open3DAlpha"),
        ("openmvs_direct", "openmvs_direct", "OpenMVSDirect"),
    ],
)
def test_each_config_selects_its_route_class(
    config_name: str, route: str, class_name: str
) -> None:
    cfg = load_config(CONFIG_DIR / f"{config_name}.yaml")

    assert cfg["route"] == route
    assert routes.resolve_route(cfg["route"]).__name__ == class_name


def test_unknown_route_is_unavailable() -> None:
    with pytest.raises(routes.RouteUnavailable, match="unknown route"):
        routes.resolve_route("poisson")


def test_openmvs_direct_reports_create_structure_reason(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_DIR / "openmvs_direct.yaml")
    route = routes.resolve_route("openmvs_direct")(cfg=cfg, env=ENV)

    with pytest.raises(routes.RouteUnavailable, match="CreateStructure"):
        route.build(SMOKE, tmp_path, None)


def test_build_passes_the_same_dense_cloud_to_both_runnable_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path | None] = []

    class FakeRoute:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        def describe(self):
            from tools.mesh.reconstruct.contracts import RouteDescriptor

            return RouteDescriptor(
                route="fake", config_sha256="0" * 64, dense_backend="colmap_cuda"
            )

        def build(self, window, workdir, dense_cloud=None):  # noqa: ANN001
            seen.append(dense_cloud)
            mesh = Path(workdir) / "mesh.ply"
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_bytes(b"mesh")
            self.report = {
                "mesh": {
                    "path": str(mesh),
                    "sha256": "1" * 64,
                    "vertex_count": 3,
                    "triangle_count": 1,
                    "bbox_min_m": [0.0, 0.0, 0.0],
                    "bbox_max_m": [1.0, 1.0, 1.0],
                },
                "wall_time_s": 1.0,
                "peak_rss_mb": 1.0,
                "dense_backend": "colmap_cuda",
            }
            self.report_path = Path(workdir) / "report.json"
            self.report_path.write_text("{}")
            return RawRouteOutput(mesh_path=mesh, dense_cloud_path=dense_cloud)

    monkeypatch.setattr(routes, "resolve_route", lambda name: FakeRoute)
    dense = tmp_path / "fused.ply"
    dense.write_bytes(b"ply")
    cfg = {"route": "colmap_openmvs", "params": {}}
    frames = list(SMOKE.frames)

    for route_name in ("colmap_openmvs", "open3d_alpha"):
        payload = routes.build(
            route_name,
            bundle_root=str(tmp_path),
            window_name="smoke",
            frames=frames,
            cfg=cfg,
            workdir=str(tmp_path / route_name),
            env=ENV,
            colmap_result={},
            dense_result={"fused_ply": str(dense), "dense_backend": "colmap_cuda"},
        )
        assert payload["route"] == route_name
        assert payload["route_result"]["mesh_path"] == payload["mesh_path"]

    assert seen == [dense, dense]


def test_route_stage_sequences() -> None:
    assert routes.route_stages("colmap_openmvs") == ("model", "build")
    assert routes.route_stages("open3d_alpha") == ("model", "dense", "build")
    assert routes.route_stages("openmvs_direct") == ()
    with pytest.raises(routes.RouteUnavailable):
        routes.route_stages("nope")


def test_preflight_reports_openmvs_direct_as_exit_2() -> None:
    code, message = routes.route_smoke_preflight(
        "openmvs_direct", CONFIG_DIR / "openmvs_direct.yaml"
    )

    assert code == 2
    assert "CreateStructure" in message


def test_preflight_allows_canonical_route() -> None:
    code, message = routes.route_smoke_preflight(
        "colmap_openmvs",
        CONFIG_DIR / "canonical_colmap_openmvs.yaml",
        out_root="/nonexistent/artifacts",
    )

    assert code == 0
    assert message == ""


def test_fallback_state_is_absent_without_the_decision_file(tmp_path: Path) -> None:
    state = routes.fallback_state(tmp_path)

    assert state["recorded"] is False
    assert state["backend"] is None


def test_preflight_requires_canonical_run_when_fallback_is_recorded(
    tmp_path: Path,
) -> None:
    decisions = tmp_path / "route_decisions"
    decisions.mkdir()
    (decisions / "dense-backend.json").write_text(
        json.dumps({"backend": "openmvs_densify"})
    )

    code, message = routes.route_smoke_preflight(
        "open3d_alpha", CONFIG_DIR / "open3d_alpha.yaml", out_root=tmp_path
    )

    assert code == 3
    assert "colmap_openmvs" in message
    assert "scene_dense.ply" in message


def test_seed_open3d_dense_writes_the_dense_stage_record(tmp_path: Path) -> None:
    dense = tmp_path / "scene_dense.ply"
    dense.write_bytes(b"ply")

    path = routes.seed_open3d_dense(tmp_path / "run", dense)

    payload = json.loads(path.read_text())
    assert payload["dense_result"]["fused_ply"] == str(dense)
    assert payload["dense_result"]["dense_cloud_source"] == "openmvs_densify"


def test_seed_open3d_dense_rejects_a_missing_cloud(tmp_path: Path) -> None:
    with pytest.raises(routes.RouteUnavailable, match="missing"):
        routes.seed_open3d_dense(tmp_path, tmp_path / "missing.ply")


def test_write_ab_from_stages_records_hashes_and_provenance(tmp_path: Path) -> None:
    run_ids = {"colmap_openmvs": "canon-smoke", "open3d_alpha": "o3d-smoke"}
    for index, (route_name, run_id) in enumerate(run_ids.items()):
        stage = tmp_path / "runs" / run_id / "smoke" / "stages" / "build.json"
        stage.parent.mkdir(parents=True)
        stage.write_text(
            json.dumps(
                {
                    "route": route_name,
                    "mesh_path": f"/m/{route_name}.ply",
                    "descriptor": {
                        "config_sha256": f"{index}" * 64,
                        "dense_backend": "colmap_cuda",
                    },
                    "report": {
                        "mesh": {
                            "path": f"/m/{route_name}.ply",
                            "sha256": "a" * 64,
                            "vertex_count": 3,
                            "triangle_count": 1,
                            "bbox_min_m": [0.0, 0.0, 0.0],
                            "bbox_max_m": [1.0, 1.0, 1.0],
                        },
                        "wall_time_s": 2.0,
                        "peak_rss_mb": 3.0,
                    },
                }
            )
        )

    path = routes.write_ab_from_stages(
        tmp_path, run_ids, {"shared_dense_cloud": False, "note": "test"}
    )

    record = json.loads(path.read_text())
    assert [row["route"] for row in record["routes"]] == [
        "colmap_openmvs",
        "open3d_alpha",
    ]
    assert all(row["status"] == "ok" for row in record["routes"])
    assert all(row["mesh_sha256"] == "a" * 64 for row in record["routes"])
    assert record["provenance"]["shared_dense_cloud"] is False


def test_phase4_entrypoint_exposes_route_smoke() -> None:
    script = (Path(__file__).resolve().parents[4] / "scripts" / "phase4.sh").read_text()

    assert "--route-smoke" in script
