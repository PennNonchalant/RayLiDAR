"""Self-contained KITTI-360 reconstruction and evaluation pipeline.

Subpackages:

- :mod:`reconstruct.data_preprocess` — KITTI-360 → frozen artifacts and bundle,
- :mod:`reconstruct.mesh` — multi-route mesh reconstruction,
- :mod:`reconstruct.gaussians` — 3DGS vs 2DGS reconstruction and report,
- :mod:`reconstruct.eval` — Phase-3 metrics,
- :mod:`reconstruct.pipeline` — full-flow helper scripts.

The package is independent of the C++ Vulkan runtime in ``src/``. Heavy optional
dependencies (torch and the OpenMVS helpers) are imported lazily by the stage that
needs them.
"""

from __future__ import annotations

__all__ = ["data_preprocess", "eval", "gaussians", "mesh", "pipeline"]
