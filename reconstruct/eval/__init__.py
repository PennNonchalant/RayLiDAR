"""Offline evaluation harness for reconstruction and LiDAR-simulation quality.

This package is the Python counterpart of the Phase-3 metrics harness
library. It reads the frozen W-frame ``gt_rays.bin`` and predicted ``hits.bin``
and writes the Phase-3 ``metrics.json`` / ``metrics.csv`` schema, so the
Gaussian comparison report keeps consuming the same fields.
"""

from __future__ import annotations

__all__ = ["aggregate", "cli", "io", "metrics", "output", "stats"]
