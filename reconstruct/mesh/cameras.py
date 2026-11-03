"""P2 bundle loading plus frame/window and camera-transform helpers.

The P2 export bundle is the only construction input for the canonical route:
rectified ``image_00`` PNGs and ``geometry/*`` (cameras, calibration, poses).
Evaluation data (GT points, frozen rays, LiDAR) is never reachable from this
module (D-14, T-04-02-02).

All transforms follow the frozen KITTI-360 frame contract: ``T_AB`` maps ``p_B`` into frame A
(``p_A = T_AB * p_B``), the world frame is KITTI-360 ``W`` in metres, and there
is no second convention, scale, or alignment step.

Tolerances are file-precision-aware.  ``geometry/cameras.json`` composes
``T_W_Ck_rect`` through ``R_rect``, whose orthonormality defect measures
``9.93e-7``; the official rotation gate is therefore set at
``2e-6``.  ``T_W_V`` is composed from the pose chain and measures below
``2e-10``, so it keeps the strict ``1e-9`` gate.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import WindowSpec

# Gate for official KITTI-360 rectified/world rotations.
ROTATION_ATOL = 2e-6
# Gate for transforms composed from the pose chain (T_W_I / T_I_C0 / T_V_C0).
STRICT_ROTATION_ATOL = 1e-9
# Bottom-row and shape checks are exact in the bundle; allow only float noise.
BOTTOM_ROW_ATOL = 1e-12

BUNDLE_FRAME_COUNT = 60
IMAGE_CAMERAS: tuple[str, ...] = ("image_00", "image_01")
DEFAULT_CAMERA = "image_00"

_FRAME_CSV_KEYS: tuple[str, ...] = ("T_W_I", "image_00_rect", "image_01_rect")
_ROW_ORDER: tuple[str, ...] = (
    "r11",
    "r12",
    "r13",
    "tx",
    "r21",
    "r22",
    "r23",
    "ty",
    "r31",
    "r32",
    "r33",
    "tz",
)


class BundleError(ValueError):
    """The P2 bundle is missing, malformed, or violates the frame contract."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise BundleError(f"missing bundle file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"invalid JSON in {path}: {exc}") from exc


def _as_matrix(values: Any, *, where: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape == (3, 4):
        lifted = np.eye(4, dtype=np.float64)
        lifted[:3, :4] = matrix
        matrix = lifted
    if matrix.shape != (4, 4):
        raise BundleError(f"{where}: expected a 4x4 transform, got {matrix.shape}")
    return matrix


def _check_bottom_row(matrix: np.ndarray, *, where: str) -> None:
    bottom = np.asarray([0.0, 0.0, 0.0, 1.0])
    if np.max(np.abs(matrix[3] - bottom)) > BOTTOM_ROW_ATOL:
        raise BundleError(f"{where}: bottom row must be [0, 0, 0, 1]")


def _check_rotation(
    rotation: np.ndarray, *, where: str, atol: float = ROTATION_ATOL
) -> None:
    if rotation.shape != (3, 3):
        raise BundleError(f"{where}: expected a 3x3 rotation, got {rotation.shape}")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > atol:
        if determinant < 0.0:
            raise BundleError(f"{where}: reflection matrix (det={determinant:.6g})")
        raise BundleError(f"{where}: rotation determinant is not +1 (det={determinant:.6g})")
    defect = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    if defect > atol:
        raise BundleError(f"{where}: rotation is not orthonormal (max|RR^T-I|={defect:.3e})")


def _csv_matrix(row: Mapping[str, str], prefix: str, *, where: str) -> np.ndarray:
    try:
        values = [float(row[f"{prefix}_{name}"]) for name in _ROW_ORDER]
    except KeyError as exc:
        raise BundleError(f"{where}: frames.csv is missing {exc.args[0]}") from exc
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    _check_bottom_row(matrix, where=f"{where} ({prefix})")
    return matrix


@dataclass(frozen=True)
class Bundle:
    """Validated view of a P2 export bundle.

    ``frames`` is the ordered list of camera frames present in the bundle;
    ``frame_rows`` carries the parsed ``frames.csv`` matrices; ``frames_by_id``
    carries the per-frame ``cameras.json`` records.
    """

    root: Path
    frames: tuple[int, ...]
    cameras: dict[str, dict]
    calib: dict[str, Any]
    frame_rows: dict[int, dict[str, np.ndarray]] = field(repr=False)
    frames_by_id: dict[int, dict] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise BundleError("Bundle.root must be a pathlib.Path")
        if not self.frames:
            raise BundleError("Bundle.frames must be non-empty")


def load_frame_rows(bundle: Bundle) -> dict[int, dict[str, np.ndarray]]:
    """Parse ``geometry/frames.csv`` into per-frame 4x4 transforms.

    Every requested frame must carry the exact pose (``pose_delta == 0``); there
    is no nearest-pose fallback in construction (D-15, T-04-02-01).
    """
    path = bundle.root / "geometry" / "frames.csv"
    if not path.is_file():
        raise BundleError(f"missing bundle file: {path}")
    rows: dict[int, dict[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            where = f"{path}:{line_number}"
            try:
                frame = int(row["frame"])
                pose_frame = int(row["pose_frame"])
                pose_delta = int(row["pose_delta"])
            except (KeyError, TypeError, ValueError) as exc:
                raise BundleError(f"{where}: invalid frame columns ({exc})") from exc
            if frame in rows:
                raise BundleError(f"{where}: duplicate frame {frame}")
            matrices = {
                prefix: _csv_matrix(row, prefix, where=where) for prefix in _FRAME_CSV_KEYS
            }
            rows[frame] = {
                "pose_frame": np.asarray([pose_frame], dtype=np.int64),
                "pose_delta": np.asarray([pose_delta], dtype=np.int64),
                **matrices,
            }
    if not rows:
        raise BundleError(f"{path}: no pose rows")
    return rows


def _validate_cameras(cameras: Any, *, where: str) -> dict[str, dict]:
    if not isinstance(cameras, dict) or not cameras:
        raise BundleError(f"{where}: 'cameras' must be a non-empty mapping")
    for name, camera in cameras.items():
        if not isinstance(camera, dict):
            raise BundleError(f"{where}: camera {name} must be a mapping")
        if camera.get("model") != "PINHOLE":
            raise BundleError(f"{where}: camera {name} model must be PINHOLE")
        distortion = camera.get("distortion")
        if not isinstance(distortion, list) or not distortion:
            raise BundleError(f"{where}: camera {name} has no distortion list")
        if any(float(value) != 0.0 for value in distortion):
            raise BundleError(f"{where}: camera {name} distortion must be all zero")
        matrix = np.asarray(camera.get("K"), dtype=np.float64)
        if matrix.shape != (3, 3):
            raise BundleError(f"{where}: camera {name} K must be 3x3")
        if not np.all(np.isfinite(matrix)):
            raise BundleError(f"{where}: camera {name} K is not finite")
    return cameras


def load_bundle(root: str | Path) -> Bundle:
    """Load and validate a P2 export bundle.

    Requires ``geometry/cameras.json``, ``geometry/frames.csv``,
    ``geometry/calib.json`` and one rectified ``raw/images/image_00`` PNG per
    frame.  No evaluation-only input is read.
    """
    root = Path(root)
    cameras_path = root / "geometry" / "cameras.json"
    frames_path = root / "geometry" / "frames.csv"
    calib_path = root / "geometry" / "calib.json"
    for path in (cameras_path, frames_path, calib_path):
        if not path.is_file():
            raise BundleError(f"missing bundle file: {path}")

    cameras_doc = _read_json(cameras_path)
    if not isinstance(cameras_doc, dict):
        raise BundleError(f"{cameras_path}: root must be a mapping")
    if cameras_doc.get("schema_version") != 1:
        raise BundleError(f"{cameras_path}: unsupported schema_version")
    if cameras_doc.get("units") != "metre":
        raise BundleError(f"{cameras_path}: units must be 'metre'")
    if int(cameras_doc.get("max_pose_delta_frames", 0)) != 0:
        raise BundleError(f"{cameras_path}: max_pose_delta_frames must be 0")
    cameras = _validate_cameras(cameras_doc.get("cameras"), where=str(cameras_path))

    frame_entries = cameras_doc.get("frames")
    if not isinstance(frame_entries, list) or not frame_entries:
        raise BundleError(f"{cameras_path}: 'frames' must be a non-empty list")
    frames: list[int] = []
    frames_by_id: dict[int, dict] = {}
    for entry in frame_entries:
        if not isinstance(entry, dict) or "frame" not in entry:
            raise BundleError(f"{cameras_path}: malformed frame entry")
        frame = int(entry["frame"])
        if frame in frames_by_id:
            raise BundleError(f"{cameras_path}: duplicate frame {frame}")
        frames.append(frame)
        frames_by_id[frame] = entry
    if frames != sorted(frames):
        raise BundleError(f"{cameras_path}: frame list must be strictly increasing")
    if len(set(frames)) != len(frames):
        raise BundleError(f"{cameras_path}: duplicate frame ids")
    if len(frames) != BUNDLE_FRAME_COUNT:
        raise BundleError(
            f"{cameras_path}: expected {BUNDLE_FRAME_COUNT} frames, got {len(frames)}"
        )

    calib = _read_json(calib_path)
    if not isinstance(calib, dict):
        raise BundleError(f"{calib_path}: root must be a mapping")
    for key in ("T_I_image_00", "T_V_C0"):
        if key not in calib:
            raise BundleError(f"{calib_path}: missing {key}")
        _as_matrix(calib[key], where=f"{calib_path}:{key}")

    bundle = Bundle(
        root=root,
        frames=tuple(frames),
        cameras=cameras,
        calib=calib,
        frame_rows={},
        frames_by_id=frames_by_id,
    )

    for frame in bundle.frames:
        image = bundle.root / "raw" / "images" / DEFAULT_CAMERA / f"{frame:010d}.png"
        if not image.is_file():
            raise BundleError(f"missing rectified image: {image}")

    rows = load_frame_rows(bundle)
    if tuple(sorted(rows)) != bundle.frames:
        raise BundleError(
            f"{frames_path}: pose rows do not match cameras.json frames"
        )
    bundle = Bundle(
        root=bundle.root,
        frames=bundle.frames,
        cameras=bundle.cameras,
        calib=bundle.calib,
        frame_rows=rows,
        frames_by_id=bundle.frames_by_id,
    )
    return bundle


def window_frames(
    bundle: Bundle,
    name_or_frames: str | Iterable[int],
    *,
    allow_arbitrary: bool = False,
) -> WindowSpec:
    """Return a :class:`~reconstruct.mesh.contracts.WindowSpec`.

    The window is derived from the loaded bundle, never from a hard-coded
    segment: ``"smoke"`` means the first ten bundle frames and ``"full"`` means
    every bundle frame.  An explicit list may also be supplied; it is validated
    against the bundle and the pose contract.
    """
    if isinstance(name_or_frames, str):
        name = name_or_frames
        if name == "smoke":
            frames = tuple(bundle.frames[:10])
        elif name == "full":
            frames = tuple(bundle.frames)
        else:
            raise BundleError(f"unknown window: {name!r}")
    else:
        frames = tuple(int(frame) for frame in name_or_frames)
        if not frames:
            raise BundleError("window_frames: empty window")
        try:
            WindowSpec("window", frames)
        except ValueError as exc:
            raise BundleError(f"window_frames: invalid window: {exc}") from exc
        name = "smoke" if len(frames) == 10 else "full" if len(frames) == len(bundle.frames) else "window"

    missing = [frame for frame in frames if frame not in bundle.frames_by_id]
    if missing:
        raise BundleError(f"window_frames: frames not in bundle: {missing}")

    for frame in frames:
        rows = bundle.frame_rows.get(frame)
        if rows is None:
            raise BundleError(f"window_frames: no pose row for frame {frame}")
        if int(rows["pose_delta"][0]) != 0:
            raise BundleError(f"window_frames: frame {frame} has pose_delta != 0")
        if int(rows["pose_frame"][0]) != frame:
            raise BundleError(f"window_frames: frame {frame} resolved a different pose frame")

    return WindowSpec(name, frames)


def T_W_C(bundle: Bundle, frame: int, camera: str = DEFAULT_CAMERA) -> np.ndarray:
    """World→rectified-camera transform ``T_W_Ck_rect`` for ``frame``."""
    if camera not in IMAGE_CAMERAS:
        raise BundleError(f"T_W_C: unknown camera {camera!r}")
    entry = bundle.frames_by_id.get(int(frame))
    if entry is None:
        raise BundleError(f"T_W_C: frame {frame} is not in the bundle")
    key = f"T_W_{camera}_rect"
    if key not in entry:
        raise BundleError(f"T_W_C: frame {frame} has no {key}")
    matrix = _as_matrix(entry[key], where=f"cameras.json:frame {frame}:{key}")
    _check_bottom_row(matrix, where=f"T_W_C({frame}, {camera})")
    _check_rotation(matrix[:3, :3], where=f"T_W_C({frame}, {camera})")
    return matrix.copy()


def T_C_W(bundle: Bundle, frame: int, camera: str = DEFAULT_CAMERA) -> np.ndarray:
    """Camera→world transform ``T_Ck_rect_W = inv(T_W_Ck_rect)``."""
    return np.linalg.inv(T_W_C(bundle, frame, camera))


def T_W_V(bundle: Bundle, frame: int) -> np.ndarray:
    """World→Velodyne transform ``T_W_V = T_W_I * T_I_C0 * inv(T_V_C0)``.

    Mirrors ``raylidar::calib::ComposeTWV`` exactly.  There is no LiDAR data
    read here; only calibration and pose matrices are used.
    """
    rows = bundle.frame_rows.get(int(frame))
    if rows is None:
        raise BundleError(f"T_W_V: no pose row for frame {frame}")
    T_W_I = rows["T_W_I"]
    T_I_C0 = _as_matrix(bundle.calib["T_I_image_00"], where="calib.json:T_I_image_00")
    T_V_C0 = _as_matrix(bundle.calib["T_V_C0"], where="calib.json:T_V_C0")

    _check_bottom_row(T_W_I, where=f"T_W_I({frame})")
    _check_rotation(T_W_I[:3, :3], where=f"T_W_I({frame})", atol=ROTATION_ATOL)
    _check_rotation(T_I_C0[:3, :3], where="T_I_image_00", atol=ROTATION_ATOL)
    _check_rotation(T_V_C0[:3, :3], where="T_V_C0", atol=ROTATION_ATOL)

    transform = T_W_I @ T_I_C0 @ np.linalg.inv(T_V_C0)
    _check_bottom_row(transform, where=f"T_W_V({frame})")
    _check_rotation(
        transform[:3, :3], where=f"T_W_V({frame})", atol=STRICT_ROTATION_ATOL
    )
    return transform


def image_path(
    bundle: Bundle, frame: int, camera: str = DEFAULT_CAMERA
) -> Path:
    """Path to the rectified PNG for ``frame`` (existence checked at load)."""
    if int(frame) not in bundle.frames_by_id:
        raise BundleError(f"image_path: frame {frame} is not in the bundle")
    return bundle.root / "raw" / "images" / camera / f"{int(frame):010d}.png"


__all__ = [
    "BUNDLE_FRAME_COUNT",
    "Bundle",
    "BundleError",
    "DEFAULT_CAMERA",
    "IMAGE_CAMERAS",
    "ROTATION_ATOL",
    "STRICT_ROTATION_ATOL",
    "T_C_W",
    "T_W_C",
    "T_W_V",
    "image_path",
    "load_bundle",
    "load_frame_rows",
    "window_frames",
]
