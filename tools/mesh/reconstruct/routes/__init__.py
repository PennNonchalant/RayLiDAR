"""Lazy route resolution.

Route modules import heavy dependencies (``open3d``, OpenMVS helpers), so the
package never imports them eagerly.  A missing implementation is reported as
:class:`RouteUnavailable` with the underlying reason.
"""

from __future__ import annotations

import importlib
from typing import Any

ROUTES: dict[str, str] = {
    "colmap_openmvs": "tools.mesh.reconstruct.routes.colmap_openmvs:ColmapOpenMVS",
    "open3d_alpha": "tools.mesh.reconstruct.routes.open3d_alpha:Open3DAlpha",
    "openmvs_direct": "tools.mesh.reconstruct.routes.openmvs_direct:OpenMVSDirect",
}


class RouteUnavailable(RuntimeError):
    """The requested route is unknown or its implementation is absent."""


def resolve_route(name: str) -> Any:
    """Return the route class for ``name``, importing it lazily."""
    target = ROUTES.get(name)
    if target is None:
        raise RouteUnavailable(f"unknown route: {name}")
    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RouteUnavailable(f"{name}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise RouteUnavailable(
            f"{name}: {module_name} does not define {attribute}"
        ) from exc


__all__ = ["ROUTES", "RouteUnavailable", "resolve_route"]
