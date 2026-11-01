"""Unit tests for the P2 bundle loader and camera transforms (synthetic data)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools.mesh.reconstruct import cameras as cam

FRAMES = tuple(range(4615, 4675))
FX = 552.554261
FY = 552.554261
CX = 682.049453
CY = 238.769549


def _yaw(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _transform(rotation: np.ndarray, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _camera() -> dict:
    return {
        "model": "PINHOLE",
        "width": 1408,
        "height": 376,
        "K": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
        "P_rect": [[FX, 0.0, CX, 0.0], [0.0, FY, CY, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _pose_rows(frames=FRAMES, *, pose_delta: dict[int, int] | None = None) -> list[dict]:
    rows = []
    for frame in frames:
        T_W_I = _transform(_yaw(0.01 * (frame - frames[0])), [float(frame), 1.0, 2.0])
        delta = 0 if pose_delta is None else int(pose_delta.get(frame, 0))
        rows.append(
            {
                "frame": frame,
                "pose_frame": frame - delta,
                "pose_delta": delta,
                "T_W_I": T_W_I,
                "image_00_rect": T_W_I,
                "image_01_rect": T_W_I,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = ["frame", "pose_frame", "pose_delta"]
    for prefix in ("T_W_I", "image_00_rect", "image_01_rect"):
        columns += [f"{prefix}_{name}" for name in cam._ROW_ORDER]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            record = {
                "frame": row["frame"],
                "pose_frame": row["pose_frame"],
                "pose_delta": row["pose_delta"],
            }
            for prefix in ("T_W_I", "image_00_rect", "image_01_rect"):
                values = row[prefix][:3, :4].reshape(-1)
                for name, value in zip(cam._ROW_ORDER, values):
                    record[f"{prefix}_{name}"] = repr(float(value))
            writer.writerow(record)


def make_bundle(
    root: Path,
    *,
    frames=FRAMES,
    pose_delta: dict[int, int] | None = None,
    cameras: dict | None = None,
    frame_overrides: dict[int, dict] | None = None,
) -> Path:
    """Write a synthetic but structurally valid P2 bundle into ``root``."""
    geometry = root / "geometry"
    geometry.mkdir(parents=True, exist_ok=True)
    images = root / "raw" / "images" / "image_00"
    images.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (images / f"{frame:010d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    rows = _pose_rows(frames, pose_delta=pose_delta)
    _write_csv(geometry / "frames.csv", rows)

    camera_doc = _camera() if cameras is None else cameras
    frame_entries = []
    for row in rows:
        entry = {
            "frame": row["frame"],
            "T_W_image_00_rect": row["image_00_rect"].tolist(),
            "T_image_00_rect_W": np.linalg.inv(row["image_00_rect"]).tolist(),
            "T_W_image_01_rect": row["image_01_rect"].tolist(),
            "T_image_01_rect_W": np.linalg.inv(row["image_01_rect"]).tolist(),
        }
        if frame_overrides and row["frame"] in frame_overrides:
            entry.update(frame_overrides[row["frame"]])
        frame_entries.append(entry)

    cameras_doc = {
        "schema_version": 1,
        "sequence": "synthetic",
        "convention": "p_A = T_A_B * p_B",
        "world_frame": "synthetic W",
        "units": "metre",
        "max_pose_delta_frames": 0,
        "cameras": {"image_00": camera_doc, "image_01": camera_doc},
        "frames": frame_entries,
    }
    (geometry / "cameras.json").write_text(
        json.dumps(cameras_doc, indent=2) + "\n", encoding="utf-8"
    )
    calib = {
        "schema_version": 1,
        "T_I_image_00": np.eye(4).tolist(),
        "T_I_image_01": np.eye(4).tolist(),
        "T_V_C0": np.eye(4).tolist(),
    }
    (geometry / "calib.json").write_text(
        json.dumps(calib, indent=2) + "\n", encoding="utf-8"
    )
    return root


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
    camera = _camera()
    camera["model"] = "SIMPLE_RADIAL"
    with pytest.raises(cam.BundleError, match="PINHOLE"):
        cam.load_bundle(make_bundle(tmp_path / "bundle", cameras=camera))


def test_nonzero_distortion_raises(tmp_path: Path) -> None:
    camera = _camera()
    camera["distortion"] = [0.01, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(cam.BundleError, match="distortion"):
        cam.load_bundle(make_bundle(tmp_path / "bundle", cameras=camera))


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
    bundle = cam.load_bundle(
        make_bundle(tmp_path / "bundle", pose_delta={4615: 1})
    )
    with pytest.raises(cam.BundleError, match="pose_delta"):
        cam.window_frames(bundle, "smoke")


def test_T_W_V_matches_pose_chain(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    transform = cam.T_W_V(bundle, 4615)
    assert np.allclose(transform, bundle.frame_rows[4615]["T_W_I"], atol=1e-12)
    rotation = transform[:3, :3]
    assert abs(np.linalg.det(rotation) - 1.0) <= 1e-12
    assert np.max(np.abs(rotation @ rotation.T - np.eye(3))) <= 1e-12
