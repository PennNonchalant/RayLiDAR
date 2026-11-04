"""Fabricate a known-pose COLMAP model and triangulate with frozen poses.

The model is built from KITTI-360 metric poses and the rectified ``image_00``
frames of the P2 bundle.  No Structure-from-Motion, mapper, bundle adjustment
with pose freedom, Umeyama, ICP, LiDAR, or evaluation data is used (D-14, D-15).

Pipeline (all commands are argument lists, never shell strings):

1. ``feature_extractor`` writes ``database.db`` (PINHOLE, frozen intrinsics).
2. ``exhaustive_matcher`` matches features on the CPU.
3. The COLMAP text model is fabricated from ``geometry/cameras.json`` and the
   database ids: ``images.txt`` stores **world→camera** poses.
4. ``point_triangulator`` triangulates points with every intrinsic refinement
   disabled; the poses must come back numerically unchanged.
5. ``model_converter`` exports the result to text for the invariance diff.

Pose invariance is the RECON-01 gate: a metric mesh built on a drifted or
re-based model is worthless downstream.  Each assertion is therefore hard.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shlex
import shutil
import sqlite3
import struct
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .cameras import (
    Bundle,
    T_W_C,
    image_path,
    load_bundle,
    window_frames,
)

DEFAULT_COLMAP = Path(
    os.environ.get("RAYLIDAR_COLMAP", "/data-20T/conda_envs/colmap/bin/colmap")
)
POSE_MAX_ABS_DELTA_ATOL = 1e-9
CAMERA_CENTER_ERROR_ATOL = 1e-6
CAMERA_PARAMS_ATOL = 1e-6
COLMAP_TIMEOUT_S = 3600.0

# COLMAP stores the camera model as an integer enum in ``database.db`` (the
# text model files use the name).  The ids are COLMAP's ``CameraModelId`` order.
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


class ColmapError(RuntimeError):
    """A COLMAP stage or a reconstruction invariant failed."""

    exit_code = 4


# --------------------------------------------------------------------------- #
# Quaternions
# --------------------------------------------------------------------------- #


def quat_wxyz_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to ``(w, x, y, z)`` with ``w >= 0``.

    Uses the trace/Shepperd branch selection so the result stays stable for
    rotations near 180 degrees, then normalises and fixes the double-cover sign.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ColmapError(f"quat_wxyz_from_rotation: expected 3x3, got {matrix.shape}")

    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0 or not np.isfinite(norm):
        raise ColmapError("quat_wxyz_from_rotation: degenerate quaternion")
    quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)


def rotation_from_quat_wxyz(
    quat: Sequence[float],
) -> np.ndarray:
    """Convert ``(w, x, y, z)`` back to a 3x3 rotation matrix."""
    w, x, y, z = (float(value) for value in quat)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm == 0.0:
        raise ColmapError("rotation_from_quat_wxyz: zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Tooling helpers
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _colmap_binary(env: Mapping, key: str = "colmap") -> Path:
    configured = (env.get(key) or {}).get("path") if isinstance(env, Mapping) else None
    candidate = Path(configured) if configured else DEFAULT_COLMAP
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ColmapError(f"COLMAP binary is not executable: {candidate}")
    return candidate


def _colmap_info(env: Mapping) -> dict:
    info = (env.get("colmap") or {}) if isinstance(env, Mapping) else {}
    banner = info.get("banner") or ""
    if not banner:
        code, output = _capture([_colmap_binary(env), "-h"])
        banner = output if code is not None else ""
    return {
        "path": str(_colmap_binary(env)),
        "banner": banner,
        "version": info.get("version") or "",
    }


def _capture(command: Sequence) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            capture_output=True,
            text=True,
            timeout=COLMAP_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def _run_tool(command: Sequence, log_path: Path | None = None) -> str:
    """Run one COLMAP command and return its combined output."""
    parts = [str(part) for part in command]
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("$ " + shlex.join(parts) + "\n")
    try:
        completed = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=COLMAP_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ColmapError(f"failed to run {parts[0]}: {exc}") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(output)
            if output and not output.endswith("\n"):
                log.write("\n")
    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-20:])
        raise ColmapError(
            f"{Path(parts[0]).name} {parts[1] if len(parts) > 1 else ''} "
            f"exited {completed.returncode}:\n{tail}"
        )
    return output


# --------------------------------------------------------------------------- #
# Database + model files
# --------------------------------------------------------------------------- #


def _camera_model_name(value: object) -> str:
    """Map COLMAP's integer camera-model enum to its text name."""
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        return value
    try:
        model_id = int(value)
    except (TypeError, ValueError):
        return str(value)
    return CAMERA_MODEL_NAMES.get(model_id, f"UNKNOWN_{model_id}")


def read_database(path: Path) -> tuple[list[tuple[int, int, str]], list[dict]]:
    """Return ``(image rows, camera rows)`` from a COLMAP ``database.db``.

    Image rows are ``(image_id, camera_id, name)``; camera rows carry the full
    parameter vector decoded from COLMAP's double blob.
    """
    path = Path(path)
    if not path.is_file():
        raise ColmapError(f"missing COLMAP database: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        images = connection.execute(
            "SELECT image_id, camera_id, name FROM images ORDER BY image_id"
        ).fetchall()
        cameras = connection.execute(
            "SELECT camera_id, model, width, height, params FROM cameras "
            "ORDER BY camera_id"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ColmapError(f"cannot read {path}: {exc}") from exc
    finally:
        connection.close()

    camera_rows = []
    for camera_id, model, width, height, blob in cameras:
        blob = bytes(blob or b"")
        count = len(blob) // 8
        params = struct.unpack(f"<{count}d", blob[: count * 8]) if count else ()
        camera_rows.append(
            {
                "camera_id": int(camera_id),
                "model": _camera_model_name(model),
                "width": int(width),
                "height": int(height),
                "params": tuple(float(value) for value in params),
            }
        )
    return [(int(i), int(c), str(n)) for i, c, n in images], camera_rows


def world_to_camera_pose(T_W_C: np.ndarray) -> tuple[tuple[float, ...], np.ndarray]:
    """Return the COLMAP ``(quaternion, translation)`` for a ``T_W_C`` pose.

    A quaternion can only encode a proper rotation, while the bundle's
    ``T_W_C`` rotation carries the official ``~1e-6`` ``R_rect`` defect.  Over a
    ~4 km world baseline that defect moves a naive ``t = T_C_W[:3, 3]`` camera
    centre by millimetres.  The camera centre is the physically meaningful
    quantity, so the translation is derived from it under the quaternion's
    rotation: ``t = -R(q) * C_W``.  The result is exactly world→camera
    (``p_C = R * (p_W - C_W)``) and the centre round-trips to float precision.
    """
    center = np.asarray(T_W_C[:3, 3], dtype=np.float64)
    # COLMAP stores world→camera: invert the bundle's camera→world rotation.
    camera_from_world = np.asarray(T_W_C[:3, :3], dtype=np.float64).T
    quaternion = quat_wxyz_from_rotation(camera_from_world)
    rotation = rotation_from_quat_wxyz(quaternion)
    translation = -rotation @ center
    return quaternion, translation


def _pose_line(
    image_id: int,
    camera_id: int,
    name: str,
    T_W_C: np.ndarray,
) -> str:
    (qw, qx, qy, qz), translation = world_to_camera_pose(T_W_C)
    tx, ty, tz = (float(value) for value in translation)
    return (
        f"{image_id} {qw:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
        f"{tx:.17g} {ty:.17g} {tz:.17g} {camera_id} {name}"
    )


def write_model_files(
    bundle: Bundle,
    window,
    workdir: Path,
    image_rows: Sequence[tuple[int, int, str]],
    camera_rows: Sequence[dict],
) -> Path:
    """Write ``sparse/{cameras,images,points3D}.txt`` from validated rows."""
    workdir = Path(workdir)
    sparse_dir = workdir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    expected = {f"{frame:010d}.png": frame for frame in window.frames}
    seen = {}
    for image_id, camera_id, name in image_rows:
        if name not in expected:
            raise ColmapError(f"database has unexpected image {name!r}")
        if name in seen:
            raise ColmapError(f"database has duplicate image {name!r}")
        seen[name] = (image_id, camera_id)
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ColmapError(f"database is missing images: {missing}")

    camera_ids = {row["camera_id"] for row in camera_rows}
    for name, (_, camera_id) in seen.items():
        if camera_id not in camera_ids:
            raise ColmapError(f"image {name} references unknown camera {camera_id}")

    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(camera_rows)}\n")
        for row in camera_rows:
            params = " ".join(f"{value:.17g}" for value in row["params"])
            handle.write(
                f"{row['camera_id']} {row['model']} {row['width']} "
                f"{row['height']} {params}\n"
            )

    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write(
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(image_rows)}\n")
        for image_id, camera_id, name in image_rows:
            frame = expected[name]
            handle.write(
                _pose_line(image_id, camera_id, name, T_W_C(bundle, frame)) + "\n"
            )
            handle.write("\n")

    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        handle.write("# Number of points: 0, mean track length: 0\n")

    return sparse_dir


def write_normalized_poses(images_txt: Path, out_path: Path) -> dict[str, list[float]]:
    """Write and return ``name qw qx qy qz tx ty tz`` sorted by name."""
    poses = parse_images_txt(images_txt)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for name in sorted(poses):
            values = poses[name]
            handle.write(
                " ".join([name, *(f"{value:.12f}" for value in values)]) + "\n"
            )
    return poses


def parse_images_txt(path: Path) -> dict[str, list[float]]:
    """Parse COLMAP ``images.txt`` into ``name -> [qw,qx,qy,qz,tx,ty,tz]``."""
    path = Path(path)
    if not path.is_file():
        raise ColmapError(f"missing COLMAP images.txt: {path}")
    poses: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        # A pose line has exactly ten fields and ends with the image name.
        if len(tokens) != 10:
            continue
        try:
            values = [float(token) for token in tokens[1:8]]
            int(tokens[0])
            int(tokens[8])
        except ValueError:
            continue
        name = tokens[9]
        if name in poses:
            raise ColmapError(f"{path}: duplicate image {name!r}")
        poses[name] = values
    if not poses:
        raise ColmapError(f"{path}: no image poses found")
    return poses


def parse_points3d(path: Path) -> tuple[int, int]:
    """Return ``(point_count, observation_count)`` for a COLMAP text model."""
    path = Path(path)
    if not path.is_file():
        raise ColmapError(f"missing COLMAP points3D.txt: {path}")
    points = 0
    observations = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 8:
            continue
        points += 1
        observations += max(0, (len(tokens) - 8) // 2)
    return points, observations


def camera_centers(poses: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    """Camera centres ``C_W = -R_C_W^T t_C_W`` from world→camera poses."""
    centers = {}
    for name, values in poses.items():
        rotation = rotation_from_quat_wxyz(values[:4])
        translation = np.asarray(values[4:7], dtype=np.float64)
        centers[name] = -rotation.T @ translation
    return centers


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def prepare_images(bundle: Bundle, window, workdir: Path) -> Path:
    """Symlink (never copy) the window's rectified PNGs into ``workdir/images``."""
    images_dir = Path(workdir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for frame in window.frames:
        source = image_path(bundle, frame)
        target = images_dir / source.name
        if target.is_symlink() or target.exists():
            if target.is_symlink():
                target.unlink()
            else:
                raise ColmapError(f"refusing to overwrite real file {target}")
        target.symlink_to(source.resolve())
    return images_dir


def _assert_database(
    images: Sequence[tuple[int, int, str]],
    cameras: Sequence[dict],
    bundle: Bundle,
    window,
    camera_name: str,
    expected_params: Sequence[float],
    log_path: Path | None,
) -> None:
    expected_names = [f"{frame:010d}.png" for frame in window.frames]
    names = [name for _, _, name in images]
    if sorted(names) != sorted(expected_names):
        raise ColmapError(
            f"database images differ from the window: {sorted(set(names) ^ set(expected_names))}"
        )
    if len(cameras) != len(expected_names):
        raise ColmapError(
            f"expected {len(expected_names)} camera rows, got {len(cameras)}"
        )
    camera = bundle.cameras[camera_name]
    for row in cameras:
        if row["model"] != "PINHOLE":
            raise ColmapError(f"camera {row['camera_id']} is {row['model']}, not PINHOLE")
        if row["width"] != int(camera["width"]) or row["height"] != int(camera["height"]):
            raise ColmapError(f"camera {row['camera_id']} has unexpected size")
        if len(row["params"]) < 4:
            raise ColmapError(f"camera {row['camera_id']} has no PINHOLE params")
        delta = float(np.max(np.abs(np.asarray(row["params"][:4]) - np.asarray(expected_params))))
        if delta > CAMERA_PARAMS_ATOL:
            raise ColmapError(
                f"camera {row['camera_id']} params drift by {delta:.3e}"
            )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def run(
    bundle_root: str | Path,
    window_name: str,
    frames: Iterable[int],
    cfg: dict,
    workdir: str | Path,
    env: Mapping,
) -> dict:
    """Run the model stage: known-pose model + fixed-pose triangulation."""
    started = time.time()
    workdir = Path(workdir)
    colmap_dir = workdir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(bundle_root)
    window = window_frames(bundle, frames, allow_arbitrary=True)
    if tuple(int(frame) for frame in frames) != window.frames:
        raise ColmapError(
            f"config frames {tuple(frames)} do not match the validated {window.name} window"
        )

    params = cfg.get("params") or {}
    colmap_cfg = params.get("colmap") or {}
    threads = int(params.get("threads", 8))
    camera_name = str(params.get("image_camera", "image_00"))
    camera = bundle.cameras[camera_name]
    K = np.asarray(camera["K"], dtype=np.float64)
    expected_params = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    camera_params = ",".join(f"{value:.17g}" for value in expected_params)

    binary = _colmap_binary(env, "colmap")
    info = _colmap_info(env)
    log_path = colmap_dir / "model_log.txt"
    log_path.write_text("", encoding="utf-8")

    database_path = colmap_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    for directory in ("sparse", "sparse_tri", "sparse_tri_txt"):
        target = colmap_dir / directory
        if target.exists():
            shutil.rmtree(target)

    images_dir = prepare_images(bundle, window, colmap_dir)

    _run_tool(
        [
            binary,
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            images_dir,
            "--ImageReader.camera_model",
            str(colmap_cfg.get("camera_model", "PINHOLE")),
            "--ImageReader.single_camera",
            str(int(colmap_cfg.get("single_camera", 0))),
            "--ImageReader.camera_params",
            camera_params,
            "--FeatureExtraction.use_gpu",
            str(int(colmap_cfg.get("feature_use_gpu", 0))),
            "--FeatureExtraction.num_threads",
            str(threads),
        ],
        log_path,
    )

    image_rows, camera_rows = read_database(database_path)
    _assert_database(
        image_rows, camera_rows, bundle, window, camera_name, expected_params, log_path
    )

    sparse_dir = write_model_files(bundle, window, colmap_dir, image_rows, camera_rows)
    before_path = colmap_dir / "images_before.txt"
    before_poses = write_normalized_poses(sparse_dir / "images.txt", before_path)

    pair_count = len(window.frames) * (len(window.frames) - 1) // 2
    matching_gpu = str(int(colmap_cfg.get("matching_use_gpu", 0)))
    if colmap_cfg.get("matcher") == "sequential" and pair_count > 2000:
        _run_tool(
            [
                binary,
                "sequential_matcher",
                "--database_path",
                database_path,
                "--SequentialMatching.overlap",
                "10",
                "--FeatureMatching.use_gpu",
                matching_gpu,
            ],
            log_path,
        )
    else:
        _run_tool(
            [
                binary,
                "exhaustive_matcher",
                "--database_path",
                database_path,
                "--FeatureMatching.use_gpu",
                matching_gpu,
            ],
            log_path,
        )

    (colmap_dir / "sparse_tri").mkdir(parents=True, exist_ok=True)

    _run_tool(
        [
            binary,
            "point_triangulator",
            "--database_path",
            database_path,
            "--image_path",
            images_dir,
            "--input_path",
            sparse_dir,
            "--output_path",
            colmap_dir / "sparse_tri",
            "--refine_intrinsics",
            "0",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.ba_refine_sensor_from_rig",
            "0",
        ],
        log_path,
    )

    triangulated_txt = colmap_dir / "sparse_tri_txt"
    triangulated_txt.mkdir(parents=True, exist_ok=True)
    _run_tool(
        [
            binary,
            "model_converter",
            "--input_path",
            colmap_dir / "sparse_tri",
            "--output_path",
            triangulated_txt,
            "--output_type",
            "TXT",
        ],
        log_path,
    )

    after_path = colmap_dir / "images_after.txt"
    after_poses = write_normalized_poses(triangulated_txt / "images.txt", after_path)

    if sorted(before_poses) != sorted(after_poses):
        raise ColmapError("triangulation changed the image set")
    pose_delta = 0.0
    for name, before in before_poses.items():
        after = after_poses[name]
        pose_delta = max(pose_delta, max(abs(a - b) for a, b in zip(before, after)))

    centers = camera_centers(after_poses)
    center_error = 0.0
    for frame in window.frames:
        name = f"{frame:010d}.png"
        expected = T_W_C(bundle, frame, camera_name)[:3, 3]
        center_error = max(center_error, float(np.max(np.abs(centers[name] - expected))))

    points3d_count, observation_count = parse_points3d(triangulated_txt / "points3D.txt")

    if pose_delta > POSE_MAX_ABS_DELTA_ATOL:
        raise ColmapError(
            f"triangulation moved poses by {pose_delta:.3e} (limit {POSE_MAX_ABS_DELTA_ATOL:.0e})"
        )
    if center_error > CAMERA_CENTER_ERROR_ATOL:
        raise ColmapError(
            f"camera centres differ from the bundle by {center_error:.3e} m "
            f"(limit {CAMERA_CENTER_ERROR_ATOL:.0e} m)"
        )
    if points3d_count <= 0:
        raise ColmapError("triangulation produced no 3D points")

    report = {
        "window": window.name,
        "frames": list(window.frames),
        "image_count": len(window.frames),
        "points3D_count": points3d_count,
        "observation_count": observation_count,
        "pose_max_abs_delta": pose_delta,
        "camera_center_max_error_m": center_error,
        "images_before_sha256": _sha256_file(before_path),
        "images_after_sha256": _sha256_file(after_path),
        "colmap": info,
        "threads": threads,
        "wall_time_s": round(time.time() - started, 3),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 1
        ),
        "bundle_root": str(bundle.root),
        "database_path": str(database_path),
        "images_dir": str(images_dir),
        "sparse_dir": str(sparse_dir),
        "sparse_tri_dir": str(colmap_dir / "sparse_tri"),
        "sparse_tri_txt_dir": str(triangulated_txt),
        "images_before_path": str(before_path),
        "images_after_path": str(after_path),
    }
    report_path = colmap_dir / "triangulation_report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    return {
        **report,
        "report_path": str(report_path),
        "report": report,
        "colmap_result": {
            "bundle_root": str(bundle.root),
            "window": window.name,
            "frames": list(window.frames),
            "database_path": str(database_path),
            "images_dir": str(images_dir),
            "sparse_dir": str(sparse_dir),
            "sparse_tri_dir": str(colmap_dir / "sparse_tri"),
            "sparse_tri_txt_dir": str(triangulated_txt),
            "report_path": str(report_path),
            "report": report,
        },
    }


__all__ = [
    "CAMERA_CENTER_ERROR_ATOL",
    "ColmapError",
    "POSE_MAX_ABS_DELTA_ATOL",
    "camera_centers",
    "parse_images_txt",
    "parse_points3d",
    "prepare_images",
    "quat_wxyz_from_rotation",
    "read_database",
    "rotation_from_quat_wxyz",
    "run",
    "world_to_camera_pose",
    "write_model_files",
    "write_normalized_poses",
]
