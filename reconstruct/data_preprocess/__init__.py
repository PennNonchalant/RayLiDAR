"""Offline KITTI-360 data preprocessing for the reconstruct pipeline.

The package turns the official KITTI-360 sequence into the deterministic inputs
used by the mesh and Gaussian reconstruction stages:

* a frame window, its poses, timestamps, and rectified calibration;
* motion-compensated ("uncurled") LiDAR scans;
* per-point dynamic/ego masks built from 2D semantics and 3D bounding boxes;
* the frozen static GT point cache;
* the frozen camera-origin GT ray set (RLRY) and the W-frame ``gt_rays.bin``;
* the self-contained handoff bundle consumed by ``reconstruct.mesh`` and
  ``reconstruct.gaussians``.

The implementation is a pure-Python offline pipeline.
It deliberately keeps the frozen binary layouts and mask bit ABI so downstream
artifacts stay compatible.
"""

from __future__ import annotations

__all__ = [
    "bbox",
    "calib",
    "config",
    "export",
    "gt_cache",
    "io_utils",
    "layout",
    "mask",
    "projection",
    "rays",
    "semantics",
    "uncurl",
    "velodyne",
]
