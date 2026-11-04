"""Canonical COLMAP -> OpenMVS mesh reconstruction package.

The reconstruction plugs into the frozen contract layer:

- :mod:`reconstruct.mesh.contracts` — window/artifact/route data contracts,
- :mod:`reconstruct.mesh.config` — strict YAML loading + canonical config hash,
- :mod:`reconstruct.mesh.pipeline` — the stage dispatch table,
- :mod:`reconstruct.mesh.routes` — lazy route resolution.

Route helpers are imported only inside the stage that needs them, so
configuration, hashing, and stage dispatch stay import-cheap.
"""

__all__ = ["contracts", "config", "pipeline", "routes"]
