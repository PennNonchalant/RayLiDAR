"""Read RayLiDAR .rlf frames (written by WriteLidarFrame in src/simulation/LidarFrame.h).

Every mode (mesh, 2D Gaussian, hybrid) writes the same layout, so training code does not care which one produced it.

    from raylidar_frame import load_frame, to_torch
    frame = load_frame("out/hybrid_4615.rlf")
    frame["range"].shape   # (64, 2048) range image, -1 = miss
    batch = to_torch(frame, device="cuda")
"""

from pathlib import Path

import numpy as np

HEADER_BYTES = 128
SOURCE_NAMES = ("none", "mesh", "gaussian")
TRUST_NAMES = ("unknown", "trusted", "untrusted")


def load_frame(path):
    """Returns a dict of numpy arrays. Per-ray arrays have shape (beams, azimuths[, 3])."""
    data = Path(path).read_bytes()
    if data[:4] != b"RLFR":
        raise ValueError(f"{path}: not a RayLiDAR frame")
    version, frame, beams, azimuths = np.frombuffer(data, "<u4", 4, 4)
    if version != 1:
        raise ValueError(f"{path}: unsupported version {version}")

    pose = np.eye(4)
    pose[:3] = np.frombuffer(data, "<f8", 12, 32).reshape(3, 4)

    n = int(beams) * int(azimuths)
    offset = HEADER_BYTES
    ranges = np.frombuffer(data, "<f4", n, offset)
    offset += 4 * n
    points = np.frombuffer(data, "<f4", 3 * n, offset).reshape(n, 3)
    offset += 12 * n
    source = np.frombuffer(data, "u1", n, offset)
    offset += n
    trust = np.frombuffer(data, "u1", n, offset)
    offset += n
    if offset != len(data):
        raise ValueError(f"{path}: expected {offset} bytes, got {len(data)}")

    shape = (int(beams), int(azimuths))
    return {
        "frame": int(frame),
        "world_from_sensor": pose,
        "range": ranges.reshape(shape),
        "points": points.reshape(*shape, 3),
        "source": source.reshape(shape),
        "trust": trust.reshape(shape),
        "hit": (ranges >= 0).reshape(shape),
    }


def hit_points(frame, world=False):
    """(M, 3) points of rays that hit, in the sensor frame or (float64) in the world frame."""
    points = frame["points"][frame["hit"]].astype(np.float64)
    if world:
        pose = frame["world_from_sensor"]
        points = points @ pose[:3, :3].T + pose[:3, 3]
    return points


def to_torch(frame, device="cpu"):
    """Same dict as torch tensors (requires PyTorch)."""
    import torch

    out = {}
    for key, value in frame.items():
        if isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    import sys

    for file in sys.argv[1:]:
        f = load_frame(file)
        hit = f["hit"]
        print(f"{file}: frame {f['frame']}, {hit.sum()} / {hit.size} hits")
        for code, name in enumerate(SOURCE_NAMES[1:], 1):
            print(f"  from {name}: {(f['source'] == code).sum()}")
        for code, name in enumerate(TRUST_NAMES):
            print(f"  {name}: {(f['trust'][hit] == code).sum()}")
