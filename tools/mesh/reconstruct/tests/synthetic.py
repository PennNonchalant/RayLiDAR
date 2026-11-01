"""Synthetic P2-style bundles for fast unit tests.

The files mirror the real export layout (``geometry/cameras.json``,
``geometry/frames.csv``, ``geometry/calib.json`` and the rectified PNGs) but
carry no real data, so tests stay hermetic and fast.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FRAMES = tuple(range(4615, 4675))
FX = 552.554261
FY = 552.554261
CX = 682.049453
CY = 238.769549
ROW_ORDER = (
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


def yaw(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def transform(rotation: np.ndarray, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def camera() -> dict:
    return {
        "model": "PINHOLE",
        "width": 1408,
        "height": 376,
        "K": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
        "P_rect": [[FX, 0.0, CX, 0.0], [0.0, FY, CY, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }


def pose_rows(
    frames=FRAMES, *, pose_delta: dict[int, int] | None = None
) -> list[dict]:
    rows = []
    for frame in frames:
        T_W_I = transform(yaw(0.01 * (frame - frames[0])), [float(frame), 1.0, 2.0])
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


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = ["frame", "pose_frame", "pose_delta"]
    for prefix in ("T_W_I", "image_00_rect", "image_01_rect"):
        columns += [f"{prefix}_{name}" for name in ROW_ORDER]
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
                for name, value in zip(ROW_ORDER, values):
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

    rows = pose_rows(frames, pose_delta=pose_delta)
    write_csv(geometry / "frames.csv", rows)

    camera_doc = camera() if cameras is None else cameras
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
