"""Command-line entry point: ``python -m reconstruct``.

The pipeline is the single supported entry point.  It preprocesses the
prepared bundle (known-pose COLMAP model), runs the configured reconstruction
route, ingests/export the canonical mesh bundle, measures diagnostics, and
writes the report.  Individual stages remain available through
``python -m reconstruct.pipeline --stage ...``.
"""

from __future__ import annotations

from .pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
