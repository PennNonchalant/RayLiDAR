"""Frozen evaluation protocol shared by the Phase-5 Gaussian methods.

This package is evaluation-only.  It turns rendered RGB/depth/alpha maps into
the same ``hits.bin`` and PSNR/SSIM numbers for 3DGS and 2DGS:

* :mod:`reconstruct.gaussians.protocol.gt_rays_io` - RLGR/RLGF/RLHT binary I/O;
* :mod:`reconstruct.gaussians.protocol.cameras` - ``geometry/cameras.json`` loader;
* :mod:`reconstruct.gaussians.protocol.depth` - projection, half-up nearest pixel,
  z-depth -> range conversion, and the frozen hit rule;
* :mod:`reconstruct.gaussians.protocol.psnr_ssim` - one PSNR/SSIM harness;
* :mod:`reconstruct.gaussians.protocol.hits` - per-method hit generation.

The training path must never import this package (D-11): GT rays and rendered
maps are used for evaluation only.
"""

from __future__ import annotations

__all__ = [
    "cameras",
    "depth",
    "gt_rays_io",
    "hits",
    "psnr_ssim",
]
