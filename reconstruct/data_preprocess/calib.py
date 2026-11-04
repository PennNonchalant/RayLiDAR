"""KITTI-360 calibration loading and frame-transform composition.

Naming follows ``p_A = T_A_B * p_B``: ``T_W_V`` maps LiDAR (``V``) points into the
KITTI-360 world frame ``W``, and ``T_Ck_rect_V`` maps ``V`` points into the
rectified camera frame ``Ck_rect``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import PreprocessError


def invert_rigid(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)


def rect4(rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    return matrix


def assert_rotation(rotation: np.ndarray, atol: float = 2.0e-6) -> None:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise PreprocessError(f"expected a 3x3 rotation, got {rotation.shape}")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > atol:
        raise PreprocessError(f"rotation determinant is not +1 (det={determinant:.6g})")
    defect = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    if defect > atol:
        raise PreprocessError(f"rotation is not orthonormal (max|RR^T-I|={defect:.3e})")


def _lines(path: Path) -> list[str]:
    if not path.is_file():
        raise PreprocessError(f"missing calibration file: {path}")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(stripped)
    return result


def _from_row_major_12(values: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return matrix


def load_cam_to_pose(path: str | Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for line in _lines(Path(path)):
        if ":" not in line:
            raise PreprocessError(f"{path}: expected 'camera: 12 values'")
        name, rest = line.split(":", 1)
        name = name.strip()
        tokens = rest.split()
        if len(tokens) != 12:
            raise PreprocessError(f"{path}: expected 12 values for {name}, got {len(tokens)}")
        if name in result:
            raise PreprocessError(f"{path}: duplicate camera entry {name!r}")
        result[name] = _from_row_major_12([float(v) for v in tokens])
    if not result:
        raise PreprocessError(f"{path}: no camera entries found")
    return result


def load_cam_to_velo(path: str | Path) -> np.ndarray:
    values: list[float] = []
    for line in _lines(Path(path)):
        values.extend(float(v) for v in line.split())
    if len(values) != 12:
        raise PreprocessError(f"{path}: expected exactly 12 values, got {len(values)}")
    return _from_row_major_12(values)


def _camera_name(suffix: str) -> str:
    return suffix if suffix.startswith("image_") else "image_" + suffix


def load_perspective(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    projections: dict[str, np.ndarray] = {}
    rectifications: dict[str, np.ndarray] = {}
    for line in _lines(Path(path)):
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        tokens = rest.split()
        if key.startswith("P_rect_"):
            if len(tokens) != 12:
                raise PreprocessError(f"{path}: expected 12 values for {key}, got {len(tokens)}")
            projections[_camera_name(key[len("P_rect_"):])] = np.asarray(
                [float(v) for v in tokens], dtype=np.float64
            ).reshape(3, 4)
        elif key.startswith("R_rect_"):
            if len(tokens) != 9:
                raise PreprocessError(f"{path}: expected 9 values for {key}, got {len(tokens)}")
            rectifications[_camera_name(key[len("R_rect_"):])] = np.asarray(
                [float(v) for v in tokens], dtype=np.float64
            ).reshape(3, 3)
    if not projections:
        raise PreprocessError(f"{path}: no P_rect entries found")
    if not rectifications:
        raise PreprocessError(f"{path}: no R_rect entries found")
    return projections, rectifications


@dataclass(frozen=True)
class Kitti360Calib:
    T_V_C0: np.ndarray
    T_I_Ck: dict[str, np.ndarray]
    P_rect: dict[str, np.ndarray]
    R_rect: dict[str, np.ndarray]
    K_rect: dict[str, np.ndarray]


def load_kitti360_calib(calibration_dir: str | Path) -> Kitti360Calib:
    root = Path(calibration_dir)
    t_v_c0 = load_cam_to_velo(root / "calib_cam_to_velo.txt")
    t_i_ck = load_cam_to_pose(root / "calib_cam_to_pose.txt")
    p_rect, r_rect = load_perspective(root / "perspective.txt")

    k_rect: dict[str, np.ndarray] = {}
    for camera, projection in p_rect.items():
        fx = float(projection[0, 0])
        fy = float(projection[1, 1])
        cx = float(projection[0, 2])
        cy = float(projection[1, 2])
        if not (fx > 0.0 and fy > 0.0):
            raise PreprocessError(f"{root}: {camera} has a non-positive focal length")
        if not (0.0 < cx < 1408.0 and 0.0 < cy < 376.0):
            raise PreprocessError(f"{root}: {camera} principal point is outside 1408x376")
        k_rect[camera] = projection[:, :3].copy()

    for camera, rotation in r_rect.items():
        assert_rotation(rotation)
    for camera in p_rect:
        if camera not in t_i_ck:
            raise PreprocessError(f"{root}: {camera} has P_rect but no calib_cam_to_pose entry")
        if camera not in r_rect:
            raise PreprocessError(f"{root}: {camera} has P_rect but no R_rect")
    if "image_00" not in t_i_ck:
        raise PreprocessError(f"{root}: calib_cam_to_pose.txt has no image_00")
    return Kitti360Calib(T_V_C0=t_v_c0, T_I_Ck=t_i_ck, P_rect=p_rect, R_rect=r_rect, K_rect=k_rect)


def compose_T_W_V(calib: Kitti360Calib, T_W_I: np.ndarray) -> np.ndarray:
    camera = calib.T_I_Ck.get("image_00")
    if camera is None:
        raise PreprocessError("calibration is missing image_00 in calib_cam_to_pose.txt")
    return compose(compose(T_W_I, camera), invert_rigid(calib.T_V_C0))


def compose_T_W_Ck_rect(calib: Kitti360Calib, T_W_I: np.ndarray, camera: str) -> np.ndarray:
    pose = calib.T_I_Ck.get(camera)
    if pose is None:
        raise PreprocessError(f"calibration is missing {camera} in calib_cam_to_pose.txt")
    rectification = calib.R_rect.get(camera)
    if rectification is None:
        raise PreprocessError(f"calibration is missing R_rect for {camera}")
    return compose(compose(T_W_I, pose), invert_rigid(rect4(rectification)))


def compose_T_Ck_rect_V(calib: Kitti360Calib, T_W_I: np.ndarray, camera: str) -> np.ndarray:
    return compose(invert_rigid(compose_T_W_Ck_rect(calib, T_W_I, camera)), compose_T_W_V(calib, T_W_I))
