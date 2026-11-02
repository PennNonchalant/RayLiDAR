"""Fast unit tests for the known-pose COLMAP model writer."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
from synthetic import CX, CY, FX, FY, camera, make_bundle

from tools.mesh.reconstruct import cameras as cam
from tools.mesh.reconstruct import colmap_model


def _write_database(path: Path, window, camera_row: dict) -> None:
    """Create a minimal COLMAP-like database with one camera per image."""
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY, model TEXT, "
            "width INTEGER, height INTEGER, params BLOB)"
        )
        for index, frame in enumerate(window.frames, start=1):
            connection.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (index, f"{frame:010d}.png", index),
            )
            blob = struct.pack(
                "<4d", FX, FY, CX, CY
            )
            connection.execute(
                "INSERT INTO cameras VALUES (?, ?, ?, ?, ?)",
                (
                    index,
                    camera_row["model"],
                    camera_row["width"],
                    camera_row["height"],
                    blob,
                ),
            )
        connection.commit()
    finally:
        connection.close()


def test_quaternion_identity() -> None:
    quaternion = colmap_model.quat_wxyz_from_rotation(np.eye(3))
    assert quaternion == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-15)


def test_quaternion_ninety_degree_yaw() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    w, x, y, z = colmap_model.quat_wxyz_from_rotation(rotation)
    assert w == pytest.approx(np.sqrt(0.5), abs=1e-12)
    assert x == pytest.approx(0.0, abs=1e-12)
    assert y == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(np.sqrt(0.5), abs=1e-12)


def test_quaternion_round_trip_and_positive_w() -> None:
    rotation = np.array(
        [
            [-0.9912, 0.1234, 0.0456],
            [-0.1234, -0.9877, 0.0912],
            [0.0563, 0.0910, 0.9943],
        ]
    )
    # Orthonormalise so the round trip is compared against a real rotation.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    quaternion = colmap_model.quat_wxyz_from_rotation(rotation)
    assert quaternion[0] >= 0.0
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-15)
    recovered = colmap_model.rotation_from_quat_wxyz(quaternion)
    assert np.max(np.abs(recovered - rotation)) <= 1e-12


def test_writer_emits_world_to_camera(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "smoke")
    workdir = tmp_path / "work"
    workdir.mkdir()

    database = tmp_path / "database.db"
    _write_database(database, window, camera())
    image_rows, camera_rows = colmap_model.read_database(database)
    sparse_dir = colmap_model.write_model_files(
        bundle, window, workdir, image_rows, camera_rows
    )

    poses = colmap_model.parse_images_txt(sparse_dir / "images.txt")
    assert len(poses) == 10
    point_in_world = np.array([1375.0, 3500.0, 116.0, 1.0])
    for frame in window.frames:
        name = f"{frame:010d}.png"
        values = poses[name]
        rotation = colmap_model.rotation_from_quat_wxyz(values[:4])
        T_C_W = np.eye(4)
        T_C_W[:3, :3] = rotation
        T_C_W[:3, 3] = values[4:7]
        expected = cam.T_C_W(bundle, frame)
        assert np.max(np.abs(T_C_W[:3, :3] - expected[:3, :3])) <= 1e-12
        assert np.max(np.abs(T_C_W[:3, 3] - expected[:3, 3])) <= 1e-9
        assert np.max(np.abs(T_C_W @ point_in_world - expected @ point_in_world)) <= 1e-9


def test_camera_params_match_manifest(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "smoke")
    workdir = tmp_path / "work"
    workdir.mkdir()
    database = tmp_path / "database.db"
    _write_database(database, window, camera())
    image_rows, camera_rows = colmap_model.read_database(database)
    sparse_dir = colmap_model.write_model_files(
        bundle, window, workdir, image_rows, camera_rows
    )

    lines = [
        line
        for line in (sparse_dir / "cameras.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(lines) == 10
    for line in lines:
        tokens = line.split()
        assert tokens[1] == "PINHOLE"
        assert int(tokens[2]) == 1408
        assert int(tokens[3]) == 376
        params = [float(value) for value in tokens[4:8]]
        assert params == pytest.approx([FX, FY, CX, CY], abs=1e-6)


def test_points3d_is_empty(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "smoke")
    workdir = tmp_path / "work"
    workdir.mkdir()
    database = tmp_path / "database.db"
    _write_database(database, window, camera())
    image_rows, camera_rows = colmap_model.read_database(database)
    sparse_dir = colmap_model.write_model_files(
        bundle, window, workdir, image_rows, camera_rows
    )
    assert colmap_model.parse_points3d(sparse_dir / "points3D.txt") == (0, 0)


def test_prepare_images_symlinks(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "smoke")
    images_dir = colmap_model.prepare_images(bundle, window, tmp_path / "work")
    for frame in window.frames:
        target = images_dir / f"{frame:010d}.png"
        assert target.is_symlink()
        assert target.resolve() == cam.image_path(bundle, frame).resolve()


def test_normalized_pose_file_is_sorted(tmp_path: Path) -> None:
    bundle = cam.load_bundle(make_bundle(tmp_path / "bundle"))
    window = cam.window_frames(bundle, "smoke")
    workdir = tmp_path / "work"
    workdir.mkdir()
    database = tmp_path / "database.db"
    _write_database(database, window, camera())
    image_rows, camera_rows = colmap_model.read_database(database)
    sparse_dir = colmap_model.write_model_files(
        bundle, window, workdir, image_rows, camera_rows
    )
    out = tmp_path / "images_before.txt"
    poses = colmap_model.write_normalized_poses(sparse_dir / "images.txt", out)
    names = [line.split()[0] for line in out.read_text(encoding="utf-8").splitlines()]
    assert names == sorted(poses)
    assert names[0] == "0000004615.png"
