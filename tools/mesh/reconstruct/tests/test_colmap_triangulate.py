"""Slow real-bundle test: fixed-pose triangulation must not move poses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.mesh.reconstruct.colmap_model import run
from tools.mesh.reconstruct.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[4]
BUNDLE_ROOT = Path("/data-20T/hybrid-lidar-sim/exports/p2-60f-step10-4615-4674")
ENV_REPORT = Path("/data-20T/hybrid-lidar-sim/artifacts/04/env_report.json")
CONFIG = REPO_ROOT / "tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml"


@pytest.mark.slow
def test_smoke_triangulation_keeps_poses(tmp_path: Path) -> None:
    if not BUNDLE_ROOT.is_dir():
        pytest.skip(f"real P2 bundle is not available: {BUNDLE_ROOT}")

    cfg = load_config(CONFIG)
    env = json.loads(ENV_REPORT.read_text(encoding="utf-8")) if ENV_REPORT.is_file() else {}
    result = run(
        BUNDLE_ROOT,
        "smoke",
        cfg["windows"]["smoke"],
        cfg,
        tmp_path,
        env,
    )
    report = result["report"]

    assert report["image_count"] == 10
    assert report["pose_max_abs_delta"] <= 1e-9
    assert report["camera_center_max_error_m"] <= 1e-6
    assert report["points3D_count"] > 0
    assert report["observation_count"] > 0

    before = Path(report["images_before_path"]).read_text(encoding="utf-8")
    after = Path(report["images_after_path"]).read_text(encoding="utf-8")
    assert before == after
    assert report["images_before_sha256"] == report["images_after_sha256"]

    assert Path(result["colmap_result"]["sparse_tri_txt_dir"], "images.txt").is_file()
    assert Path(result["colmap_result"]["report_path"]).is_file()
