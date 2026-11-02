from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tools.mesh.reconstruct.config import config_sha256, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CANONICAL = CONFIG_DIR / "canonical_colmap_openmvs.yaml"


def _base_config() -> dict:
    return {
        "schema_version": 1,
        "route": "colmap_openmvs",
        "bundle_root": "/data-20T/hybrid-lidar-sim/exports/p2-60f-step10-4615-4674",
        "out_root": "/data-20T/hybrid-lidar-sim/artifacts/04",
        "windows": {"smoke": [1, 2, 3], "full": [1, 2, 3, 4]},
        "params": {"image_camera": "image_00"},
    }


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_config_loads_and_hashes_deterministically() -> None:
    first = load_config(CANONICAL)
    second = load_config(CANONICAL)

    digest = config_sha256(first)

    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert digest == config_sha256(second)
    assert first["route"] == "colmap_openmvs"
    assert first["windows"]["smoke"][0] == 4615


def test_route_change_changes_hash() -> None:
    base = _base_config()
    other = copy.deepcopy(base)
    other["route"] = "open3d_alpha"

    assert config_sha256(base) != config_sha256(other)


def test_window_frame_change_changes_hash() -> None:
    base = _base_config()
    other = copy.deepcopy(base)
    other["windows"]["smoke"] = [1, 2, 4]

    assert config_sha256(base) != config_sha256(other)


def test_forbidden_gt_key_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["params"]["gt_path"] = "/tmp/gt.bin"
    path = _write_config(tmp_path / "gt.yaml", payload)

    with pytest.raises(ValueError, match="forbidden config key: gt_path"):
        load_config(path)


def test_forbidden_lidar_path_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["params"]["input_cloud"] = (
        "/data-20T/hybrid-lidar-sim/artifacts/02/rays/frame_4615.bin"
    )
    path = _write_config(tmp_path / "rays.yaml", payload)

    with pytest.raises(
        ValueError, match="GT/LiDAR input is not a reconstruction input"
    ):
        load_config(path)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["surprise"] = True
    path = _write_config(tmp_path / "unknown.yaml", payload)

    with pytest.raises(ValueError, match="unknown top-level config key"):
        load_config(path)


def test_unknown_route_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["route"] = "poisson"
    path = _write_config(tmp_path / "route.yaml", payload)

    with pytest.raises(ValueError, match="unknown route"):
        load_config(path)


def test_non_contiguous_window_is_rejected(tmp_path: Path) -> None:
    payload = _base_config()
    payload["windows"]["smoke"] = [1, 3]
    path = _write_config(tmp_path / "window.yaml", payload)

    with pytest.raises(ValueError, match="contiguous"):
        load_config(path)
