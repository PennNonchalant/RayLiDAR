"""Phase-4 multi-route mesh reconstruction package.

Every reconstruction route plugs into the same frozen contract layer:

- :mod:`tools.mesh.reconstruct.contracts` — window/artifact/route data contracts,
- :mod:`tools.mesh.reconstruct.config` — strict YAML loading + canonical config hash,
- :mod:`tools.mesh.reconstruct.pipeline` — the stage dispatch table,
- :mod:`tools.mesh.reconstruct.routes` — lazy route resolution.

Heavy dependencies (``open3d``) are imported only inside route modules so that
configuration, hashing, and stage dispatch stay import-cheap.
"""

__all__ = ["contracts", "config", "pipeline", "routes"]
