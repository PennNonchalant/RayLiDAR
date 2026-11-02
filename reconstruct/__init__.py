"""Standalone mesh reconstruction package.

The package owns the whole mesh pipeline behind one import boundary:

- :mod:`reconstruct.contracts` — window/artifact/route data contracts,
- :mod:`reconstruct.config` — strict YAML loading + canonical config hash,
- :mod:`reconstruct.colmap_model` — known-pose COLMAP preprocessing,
- :mod:`reconstruct.routes` — lazy reconstruction-route resolution,
- :mod:`reconstruct.ingest` — canonical cleanup, export, and manifest,
- :mod:`reconstruct.g3` — diagnostic measurements,
- :mod:`reconstruct.report` — A/B report and asset promotion.

Nothing here imports the C++ tree, the evaluation harness, or any other
repository module. Heavy dependencies (``open3d``, ``trimesh``) are imported
inside the stage that needs them, so configuration and dispatch stay cheap.
"""

__all__ = [
    "cameras",
    "cleanup",
    "colmap_model",
    "config",
    "contracts",
    "dense",
    "env",
    "g3",
    "ingest",
    "manifest",
    "meshio",
    "pipeline",
    "report",
    "routes",
    "validate",
]
