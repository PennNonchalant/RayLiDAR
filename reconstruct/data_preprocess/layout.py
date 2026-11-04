"""KITTI-360 directory layout, poses, and timestamps."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io_utils import PreprocessError

FRAME_WIDTH = 10


def frame_name(frame: int) -> str:
    return f"{int(frame):0{FRAME_WIDTH}d}"


def image_path(dataset_root: str | Path, sequence: str, camera: str, frame: int) -> Path:
    return Path(dataset_root) / "data_2d_raw" / sequence / camera / "data_rect" / f"{frame_name(frame)}.png"


def semantic_path(dataset_root: str | Path, sequence: str, camera: str, frame: int) -> Path:
    return (
        Path(dataset_root)
        / "data_2d_semantics"
        / "train"
        / sequence
        / camera
        / "semantic"
        / f"{frame_name(frame)}.png"
    )


def lidar_path(lidar_root: str | Path, frame: int) -> Path:
    return Path(lidar_root) / "data" / f"{frame_name(frame)}.bin"


def calibration_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "calibration"


def poses_path(dataset_root: str | Path, sequence: str) -> Path:
    return Path(dataset_root) / "data_poses" / sequence / "poses.txt"


def cam0_to_world_path(dataset_root: str | Path, sequence: str) -> Path:
    return Path(dataset_root) / "data_poses" / sequence / "cam0_to_world.txt"


def image_timestamps_path(dataset_root: str | Path, sequence: str, camera: str) -> Path:
    return Path(dataset_root) / "data_2d_raw" / sequence / camera / "timestamps.txt"


def lidar_timestamps_path(lidar_root: str | Path) -> Path:
    return Path(lidar_root) / "timestamps.txt"


def _matrix_from_row_major_12(values: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return matrix


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise PreprocessError(f"missing file: {path}")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def load_poses(path: str | Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for line in _read_nonempty_lines(Path(path)):
        tokens = line.split()
        if len(tokens) != 13:
            raise PreprocessError(f"{path}: expected 13 tokens (frame + 12 values), got {len(tokens)}")
        frame = int(round(float(tokens[0])))
        if frame in poses:
            raise PreprocessError(f"{path}: duplicate frame {frame}")
        poses[frame] = _matrix_from_row_major_12([float(v) for v in tokens[1:]])
    return poses


def load_cam0_to_world(path: str | Path) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for line in _read_nonempty_lines(Path(path)):
        tokens = line.split()
        if len(tokens) != 17:
            raise PreprocessError(f"{path}: expected 17 tokens (frame + 16 values), got {len(tokens)}")
        frame = int(round(float(tokens[0])))
        if frame in result:
            raise PreprocessError(f"{path}: duplicate frame {frame}")
        result[frame] = np.asarray([float(v) for v in tokens[1:]], dtype=np.float64).reshape(4, 4)
    return result


def parse_utc_timestamp(text: str) -> int:
    """Parse ``YYYY-MM-DD HH:MM:SS.nnnnnnnnn`` into nanoseconds since the epoch."""

    text = text.strip()
    if len(text) != 29:
        raise PreprocessError(f"timestamp must be exactly 29 characters: {text!r}")
    if text[4] != "-" or text[7] != "-" or text[10] != " " or text[13] != ":" or text[16] != ":" or text[19] != ".":
        raise PreprocessError(f"timestamp separators are malformed: {text!r}")
    date_part = text[:19]
    fraction = text[20:]
    if not fraction.isdigit():
        raise PreprocessError(f"timestamp fraction is not numeric: {text!r}")
    stamp = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    seconds = int(stamp.timestamp())
    return seconds * 1_000_000_000 + int(fraction)


def load_timestamps(path: str | Path) -> list[int]:
    values = [parse_utc_timestamp(line) for line in _read_nonempty_lines(Path(path))]
    if not values:
        raise PreprocessError(f"{path}: no timestamps")
    return values


def common_frames(
    dataset_root: str | Path,
    sequence: str,
    cameras: tuple[str, ...],
    poses: dict[int, np.ndarray],
    *,
    lidar_root: str | Path | None = None,
) -> list[int]:
    """Frames that have a pose, both rectified images, and a LiDAR scan."""

    root = Path(dataset_root)
    lidar = Path(lidar_root) if lidar_root is not None else root / "data_3d_raw" / sequence / "velodyne_points"
    found: list[int] = []
    for frame in sorted(poses):
        if not lidar_path(lidar, frame).is_file():
            continue
        if all(image_path(root, sequence, camera, frame).is_file() for camera in cameras):
            found.append(frame)
    return found


def contiguous_runs(frames: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for frame in frames:
        if runs and frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs
