"""Phase 5 Gaussian-only reconstruction helpers.

Modules are imported lazily by callers; this package intentionally imports
neither torch nor the CUDA extensions at import time.
"""

__all__ = ["config", "dataset", "env", "pipeline"]
