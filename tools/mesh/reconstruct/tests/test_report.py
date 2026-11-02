"""A/B report fields, promotion hashing, and the idempotent local index."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.mesh.reconstruct import report

SMOKE = list(range(4615, 4625))
FULL = list(range(4615, 4675))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checks() -> dict:
    return {
        "bbox_ratio_ok": {
            "pass": True,
            "measured": {
                "ratio": [1.0, 1.0],
                "axes": ["x", "y"],
                "height_band": {
                    "mesh_z_min_m": 0.0,
                    "mesh_z_max_m": 10.0,
                    "gt_z_min_m": 0.0,
                    "gt_z_max_m": 10.0,
                    "z_min_margin_m": 5.0,
                    "z_max_margin_m": 20.0,
                    "z_min_ok": True,
                    "z_max_ok": True,
                    "overlap": True,
                },
            },
            "threshold": {
                "min": 0.8,
                "max": 1.2,
                "axes": ["x", "y"],
                "z_min_margin_m": 5.0,
                "z_max_margin_m": 20.0,
            },
            "reason": "",
        },
        "camera_outside_ok": {
            "pass": True,
            "measured": {"forward_hit_rate": 0.6, "near_hit_count": 0},
            "threshold": {"forward_hit_rate_min": 0.3, "near_hit_max_m": 1.5},
            "reason": "",
        },
        "components_ok": {
            "pass": True,
            "measured": {"component_count": 1, "largest_component_fraction": 0.99},
            "threshold": {"largest_component_fraction_min": 0.90},
            "reason": "",
        },
        "degenerate_ok": {
            "pass": True,
            "measured": {"degenerate_count": 0, "degenerate_fraction": 0.0},
            "threshold": {"degenerate_fraction_max": 0.001},
            "reason": "",
        },
        "smoke_range_ok": {
            "pass": True,
            "measured": {"median_abs_error_m": 0.2, "p90_abs_error_m": 0.8},
            "threshold": {"median_abs_error_m_max": 1.0},
            "reason": "",
        },
    }


def _measurements() -> dict:
    return {
        name: {"measured": check["measured"], "reference": check["threshold"]}
        for name, check in _checks().items()
    }


def _g3_report(passed: bool = True) -> dict:
    return {
        "status": "measured",
        "mode": "diagnostics",
        "window": "full",
        "frames": FULL,
        "gt_frames": FULL,
        "smoke_range_frames": SMOKE,
        "measurements": _measurements(),
        "amendment": {
            "id": "G3-AMEND-01",
            "approved_at_utc": "2026-11-01",
            "camera_visible_gt": True,
        },
        "reference_thresholds": {
            "bbox_ratio_min": 0.8,
            "bbox_ratio_max": 1.2,
            "bbox_axes": ["x", "y"],
            "mesh_z_min_margin_m": 5.0,
            "mesh_z_max_margin_m": 20.0,
            "forward_hit_rate_min": 0.3,
            "near_hit_max_m": 1.5,
            "largest_component_fraction_min": 0.90,
            "degenerate_fraction_max": 0.001,
            "smoke_median_abs_error_m_max": 1.0,
            "ray_t_min_m": 1e-4,
            "camera_outside_t_max_m": 200.0,
            "gt_visible_camera_max": 2,
        },
        "metrics": {
            "components": {"count": 1, "largest_component_fraction": 0.99},
            "degenerate": {"count": 0, "fraction": 0.0},
            "camera_outside": {
                "forward_hit_rate": 0.6,
                "near_hits": [],
                "directions": {
                    "forward": [0.0, 0.0, 1.0],
                    "yaw_plus_30": [0.5, 0.0, 0.866],
                },
            },
        },
        "mesh": {"path": "/tmp/mesh.ply", "sha256": "c" * 64},
        "alignment_applied": False,
        "evaluation_frames_used_for_tuning": False,
    }


def _route(route_name: str, backend: str) -> dict:
    return {
        "route": route_name,
        "status": "ok",
        "config_sha256": "d" * 64,
        "mesh_sha256": "e" * 64,
        "vertex_count": 22882,
        "triangle_count": 44852,
        "bbox_min_m": [1376.9, 3465.5, 112.99],
        "bbox_max_m": [1459.7, 3509.9, 128.97],
        "components_count": 1,
        "largest_component_fraction": 0.99,
        "degenerate_fraction": 0.0,
        "build_wall_time_s": 17.2,
        "peak_rss_mb": 384.5,
        "dense_backend": backend,
        "g3_status": "measured",
        "g3_measurements": _measurements(),
    }


def _artifact(tmp_path: Path) -> dict:
    return {
        "path": str(tmp_path / "mesh.ply"),
        "sha256": "c" * 64,
        "vertex_count": 22882,
        "triangle_count": 44852,
        "bbox_min_m": [0.0, 0.0, 0.0],
        "bbox_max_m": [10.0, 10.0, 10.0],
    }


def test_write_reports_every_d12_field_and_the_fallback_caveat(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "artifacts" / "04"
    cfg = {
        "route": "colmap_openmvs",
        "out_root": str(out_root),
        "windows": {"smoke": SMOKE, "full": FULL},
    }
    routes = [
        _route("colmap_openmvs", "openmvs_densify"),
        _route("open3d_alpha", "openmvs_densify"),
    ]

    written = report.write(
        _artifact(tmp_path),
        _g3_report(),
        routes,
        cfg,
        str(tmp_path / "run"),
        {},
        run_id="run-1",
        asset_dir=tmp_path / "assets" / "run-1",
    )

    text = Path(written["ab_report_path"]).read_text(encoding="utf-8")
    for token in (
        "route",
        "status",
        "config sha256",
        "mesh sha256",
        "vertices",
        "triangles",
        "bbox",
        "largest fraction",
        "degenerate fraction",
        "build wall",
        "peak RSS",
        "dense backend",
        "forward hit rate",
        "near hits",
        "smoke median",
        "z band",
    ):
        assert token in text, token
    assert "G3-AMEND-01" in text
    assert "camera-visible static subset" in text
    assert report.MESHER_COMPARISON_SENTENCE in text
    assert "What this does not show" in text
    assert written["handoff_path"] is not None
    handoff = Path(written["handoff_path"]).read_text(encoding="utf-8")
    assert "raylidar_cli mesh validate" in handoff
    assert "metres" in handoff


def test_write_omits_the_fallback_caveat_for_the_colmap_backend(
    tmp_path: Path,
) -> None:
    cfg = {
        "route": "colmap_openmvs",
        "out_root": str(tmp_path / "artifacts" / "04"),
        "windows": {"smoke": SMOKE, "full": FULL},
    }
    written = report.write(
        _artifact(tmp_path),
        _g3_report(),
        [_route("colmap_openmvs", "colmap_cuda")],
        cfg,
        str(tmp_path / "run"),
        {},
        run_id="run-1",
    )

    text = Path(written["ab_report_path"]).read_text(encoding="utf-8")
    assert report.MESHER_COMPARISON_SENTENCE not in text


def _canonical_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run" / "canonical"
    run_dir.mkdir(parents=True)
    blobs = {
        "mesh.ply": b"ply-bytes",
        "mesh.glb": b"glb-bytes",
        "mesh.obj": b"obj-bytes",
        "mesh_viewer.glb": b"viewer-bytes",
    }
    for name, payload in blobs.items():
        (run_dir / name).write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "kind": "mesh",
        "route": "colmap_openmvs",
        "config_sha256": "d" * 64,
        "created_at_utc": "2026-11-01T09:00:00Z",
        "files": {
            "ply": {"path": "mesh.ply", "sha256": _sha(run_dir / "mesh.ply")},
            "glb": {"path": "mesh.glb", "sha256": _sha(run_dir / "mesh.glb")},
            "obj": {"path": "mesh.obj", "sha256": _sha(run_dir / "mesh.obj")},
        },
        "viewer": {
            "path": "mesh_viewer.glb",
            "sha256": _sha(run_dir / "mesh_viewer.glb"),
        },
        "vertex_count": 8,
        "triangle_count": 8,
        "bbox_min_m": [0.0, 0.0, 0.0],
        "bbox_max_m": [1.0, 1.0, 1.0],
        "inputs": {"frames": SMOKE},
        "dense_backend": "colmap_cuda",
        "g3": {"pass": False, "checks": {}},
    }
    manifest_path = run_dir / "mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_dir, manifest_path


def test_promotion_accepts_a_diagnostic_report(tmp_path: Path) -> None:
    run_dir, manifest_path = _canonical_run(tmp_path)

    promoted = report.promote(
        run_dir,
        "run-1",
        _g3_report(),
        manifest_path,
        assets=tmp_path / "assets",
    )

    assert (Path(promoted["asset_dir"]) / "mesh.ply").is_file()
    delivered = json.loads(
        (Path(promoted["asset_dir"]) / "mesh_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert delivered["g3"]["status"] == "measured"
    assert "pass" not in delivered["g3"]


def test_promotion_writes_verifiable_checksums(tmp_path: Path) -> None:
    run_dir, manifest_path = _canonical_run(tmp_path)

    promoted = report.promote(
        run_dir,
        "run-1",
        _g3_report(),
        manifest_path,
        assets=tmp_path / "assets",
    )

    asset_dir = Path(promoted["asset_dir"])
    assert asset_dir == tmp_path / "assets" / "run-1"
    for name in (
        "mesh.ply",
        "mesh.glb",
        "mesh.obj",
        "mesh_viewer.glb",
        "mesh_manifest.json",
        "SHA256SUMS",
    ):
        assert (asset_dir / name).is_file(), name
    assert promoted["files"]["mesh.ply"] == _sha(run_dir / "mesh.ply")
    completed = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=asset_dir,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    delivered = json.loads((asset_dir / "mesh_manifest.json").read_text(encoding="utf-8"))
    assert delivered["g3"]["status"] == "measured"
    assert "pass" not in delivered["g3"]


def test_promotion_marks_the_destination_invalid_on_a_mutation(
    tmp_path: Path,
) -> None:
    run_dir, manifest_path = _canonical_run(tmp_path)
    # Mutate a source file after the manifest was written.
    (run_dir / "mesh.glb").write_bytes(b"tampered-bytes")

    with pytest.raises(RuntimeError, match="disagree with the manifest"):
        report.promote(
            run_dir,
            "run-1",
            _g3_report(),
            manifest_path,
            assets=tmp_path / "assets",
        )

    assert (tmp_path / "assets" / "run-1" / ".invalid").is_file()


def test_index_merge_preserves_other_entries_and_is_idempotent(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runs": [
                    {
                        "run_id": "unrelated",
                        "route": "open3d_alpha",
                        "asset_dir": "/elsewhere",
                    }
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir, manifest_path = _canonical_run(tmp_path)
    promoted = report.promote(
        run_dir, "run-1", _g3_report(), manifest_path, assets=assets
    )

    first = report.update_index(
        promoted["asset_dir"],
        "run-1",
        promoted["manifest"],
        _g3_report(),
        assets=assets,
    )
    first_text = first.read_text(encoding="utf-8")
    second = report.update_index(
        promoted["asset_dir"],
        "run-1",
        promoted["manifest"],
        _g3_report(),
        assets=assets,
    )
    second_text = second.read_text(encoding="utf-8")

    assert first_text == second_text
    index = json.loads(second_text)
    run_ids = [run["run_id"] for run in index["runs"]]
    assert run_ids == ["run-1", "unrelated"]
    entry = index["runs"][0]
    assert entry["mesh_sha256"] == promoted["files"]["mesh.ply"]
    assert entry["manifest_sha256"] == promoted["manifest_sha256"]
    assert entry["url"] is None
    assert entry["download_sha256"] is None


def test_summarise_run_reads_the_persisted_stage_records(tmp_path: Path) -> None:
    workdir = tmp_path / "runs" / "colmap_openmvs-x" / "full"
    (workdir / "stages").mkdir(parents=True)
    (workdir / "stages" / "build.json").write_text(
        json.dumps(
            {
                "route": "colmap_openmvs",
                "window": "full",
                "report": {
                    "mesh": {
                        "path": "/tmp/mesh.ply",
                        "sha256": "f" * 64,
                        "vertex_count": 10,
                        "triangle_count": 12,
                        "bbox_min_m": [0.0, 0.0, 0.0],
                        "bbox_max_m": [1.0, 1.0, 1.0],
                    },
                    "wall_time_s": 5.0,
                    "peak_rss_mb": 100.0,
                    "dense_backend": "colmap_cuda",
                },
                "descriptor": {"config_sha256": "a" * 64, "dense_backend": "colmap_cuda"},
            }
        ),
        encoding="utf-8",
    )
    (workdir / "stages" / "ingest.json").write_text(
        json.dumps(
            {
                "artifact": {
                    "components": {"count": 3, "largest_fraction": 0.5},
                    "degenerate": {"count": 0, "fraction": 0.0},
                }
            }
        ),
        encoding="utf-8",
    )
    (workdir / "g3_report.json").write_text(
        json.dumps(_g3_report()), encoding="utf-8"
    )

    row = report.summarise_run(workdir)

    assert row["run_id"] == "colmap_openmvs-x"
    assert row["mesh_sha256"] == "f" * 64
    assert row["components_count"] == 3
    assert row["largest_component_fraction"] == 0.5
    assert row["g3_status"] == "measured"
    assert row["g3_measurements"]["components_ok"]["measured"][
        "largest_component_fraction"
    ] == 0.99
    assert row["build_wall_time_s"] == 5.0


def test_write_states_a_mixed_dense_cloud_provenance_honestly(
    tmp_path: Path,
) -> None:
    cfg = {
        "route": "colmap_openmvs",
        "out_root": str(tmp_path / "artifacts" / "04"),
        "windows": {"smoke": SMOKE, "full": FULL},
    }
    canonical = _route("colmap_openmvs", "colmap_cuda")
    canonical["dense_cloud_source"] = "openmvs_densify"
    ab = _route("open3d_alpha", "colmap_cuda")
    ab["dense_cloud_source"] = "colmap_cuda"

    written = report.write(
        _artifact(tmp_path),
        _g3_report(),
        [canonical, ab],
        cfg,
        str(tmp_path / "run"),
        {},
        run_id="run-1",
    )

    text = Path(written["ab_report_path"]).read_text(encoding="utf-8")
    assert "do not share a dense cloud" in text
    assert "DensifyPointCloud" in text
    assert report.MESHER_COMPARISON_SENTENCE not in text
