"""Declared-but-unavailable route: direct OpenMVS without a COLMAP structure.

The pinned OpenMVS 2.4.0 Ubuntu bundle does not ship ``CreateStructure``, so
there is no supported way to feed raw images plus a pose file directly into
OpenMVS.  Keeping the route declared (and switchable from config) makes the
limitation explicit instead of a silent absence: every attempt fails with a
precise reason pointing at the canonical COLMAP pose-import front end.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..config import config_sha256
from ..contracts import RawRouteOutput, RouteDescriptor, WindowSpec
from . import RouteUnavailable

UNAVAILABLE_REASON = (
    "OpenMVS 2.4.0 prebuilt bundle has no CreateStructure; use colmap_openmvs"
)


class OpenMVSDirect:
    """A route that reports its precise unavailability instead of running."""

    name = "openmvs_direct"

    def __init__(
        self,
        *,
        cfg: Mapping,
        env: Mapping,
        bundle_root: str | Any | None = None,
        window_name: str | None = None,
        frames: Sequence[int] | None = None,
        colmap_result: Mapping | None = None,
        dense_result: Mapping | None = None,
        **_ignored: Any,
    ) -> None:
        self.cfg = dict(cfg or {})
        self.env = dict(env or {})
        self.bundle_root = bundle_root
        self.window_name = window_name
        self.frames = tuple(int(frame) for frame in (frames or ()))
        self.colmap_result = dict(colmap_result) if colmap_result else None
        self.dense_result = dict(dense_result) if dense_result else None

    def describe(self) -> RouteDescriptor:
        openmvs = self.env.get("openmvs") or {}
        return RouteDescriptor(
            route=self.name,
            config_sha256=config_sha256(self.cfg),
            tool_versions={"openmvs": str(openmvs.get("version") or "")},
            dense_backend=str(self.env.get("dense_backend") or ""),
        )

    def build(
        self,
        window: WindowSpec,
        workdir: str | Any,
        dense_cloud: Any | None = None,
    ) -> RawRouteOutput:
        raise RouteUnavailable(UNAVAILABLE_REASON)


__all__ = ["OpenMVSDirect", "UNAVAILABLE_REASON"]
