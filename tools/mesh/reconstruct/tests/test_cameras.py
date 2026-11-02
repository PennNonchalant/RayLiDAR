"""Unit tests for the P2 bundle loader and camera transforms (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic import FRAMES, camera, make_bundle

from tools.mesh.reconstruct import cameras as cam


def test_valid_bundle_loads(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    assert len(bundle.frames) == 60
    assert bundle.frames[0] == 4615
    assert bundle.frames[-1] == 4674
    assert bundle.cameras["image_00"]["model"] == "PINHOLE"
    assert sorted(bundle.frame_rows) == list(FRAMES)


def test_missing_image_raises(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    (root / "raw" / "images" / "image_00" / "0000004615.png").unlink()
    with pytest.raises(cam.BundleError, match="missing rectified image"):
        cam.load_bundle(root)


def test_missing_pose_row_raises(tmp_path: Path) -> None:
    root = make_bundle(tmp_path / "bundle")
    frames_path = root / "geometry" / "frames.csv"
    lines = frames_path.read_text(encoding="utf-8").splitlines()
    frames_path.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")
    with pytest.raises(cam.BundleError, match="pose rows do not match"):
        cam.load_bundle(root)


def test_non_pinhole_camera_raises(tmp_path: Path) -> None:
    doc = camera()
    doc["model"] = "SIMPLE_RADIAL"
    with pytest.raises(cam.BundleError, match="PINHOLE"):
        cam.load_bundle(make_bundle(tmp_path / "bundle", cameras=doc))


def test_nonzero_distortion_raises(tmp_path: Path) -> None:
    doc = camera()
    doc["distortion"] = [0.01, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(cam.BundleError, match="distortion"):
        cam.load_bundle(make_bundle(tmp_path / "bundle", cameras=doc))


def test_reflection_matrix_raises(tmp_path: Path) -> None:
    reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
    root = make_bundle(
        tmp_path / "bundle",
        frame_overrides={4615: {"T_W_image_00_rect": reflection.tolist()}},
    )
    bundle = cam.load_bundle(root)
    with pytest.raises(cam.BundleError, match="reflection"):
        cam.T_W_C(bundle, 4615)


def test_world_camera_round_trip(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    for frame in (4615, 4640, 4674):
        T_W_C = cam.T_W_C(bundle, frame)
        T_C_W = cam.T_C_W(bundle, frame)
        assert np.max(np.abs(T_W_C @ T_C_W - np.eye(4))) <= 1e-12


def test_smoke_window_is_exact(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "smoke")
    assert window.name == "smoke"
    assert window.frames == tuple(range(4615, 4625))
    assert cam.window_frames(bundle, list(range(4615, 4625))).frames == window.frames


def test_full_window_is_60_frames(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "full")
    assert len(window.frames) == 60
    assert window.frames[0] == 4615
    assert window.frames[-1] == 4674


def test_arbitrary_window_is_rejected(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    with pytest.raises(cam.BundleError, match="window must equal"):
        cam.window_frames(bundle, list(range(4620, 4630)))


def test_window_with_frame_outside_bundle_is_rejected(tmp_path: Path) -> None:
    other_frames = tuple(range(5000, 5060))
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle", frames=other_frames))
    with pytest.raises(cam.BundleError, match="not in bundle"):
        cam.window_frames(bundle, "smoke")


def test_pose_delta_must_be_zero(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle", pose_delta={4615: 1}))
    with pytest.raises(cam.BundleError, match="pose_delta"):
        cam.window_frames(bundle, "smoke")


def test_T_W_V_matches_pose_chain(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    transform = cam.T_W_V(bundle, 4615)
    assert np.allclose(transform, bundle.frame_rows[4615]["T_W_I"], atol=1e-12)
    rotation = transform[:3, :3]
    assert abs(np.linalg.det(rotation) - 1.0) <= 1e-12
    assert np.max(np.abs(rotation @ rotation.T - np.eye(3))) <= 1e-12
