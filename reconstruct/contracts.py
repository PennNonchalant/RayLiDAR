"""Frozen contracts shared by every reconstruction route.

These dataclasses are the only shapes later plans may rely on.  They carry no
reconstruction logic and import nothing heavier than the standard library so
that configuration and dispatch stay cheap to import.

All coordinates are in the KITTI-360 world frame ``W`` and in metres; there is
no second frame, no scale, and no embedded model transform (``docs/sim-io.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class ContractError(ValueError):
    """Raised when a contract object is constructed with invalid values."""


@dataclass(frozen=True)
class WindowSpec:
    """A named, contiguous frame window.

    ``frames`` must be non-empty, strictly increasing, and contiguous: a smoke
    window is an exact sub-slice of the 60-frame full window, never an arbitrary
    frame selection.  Evaluation frames must not be singled out (D-02).
    """

    name: str
    frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ContractError("window name must be a non-empty string")
        frames = tuple(self.frames)
        if not frames:
            raise ContractError("window frames must be non-empty")
        if any(not isinstance(frame, int) or isinstance(frame, bool) for frame in frames):
            raise ContractError("window frames must be integers")
        for previous, current in zip(frames, frames[1:]):
            if current <= previous:
                raise ContractError("window frames must be strictly increasing")
            if current != previous + 1:
                raise ContractError("window frames must be contiguous")
        object.__setattr__(self, "frames", frames)

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class RouteDescriptor:
    """Static description of a reconstruction route and its run configuration."""

    route: str
    config_sha256: str
    tool_versions: dict[str, str] = field(default_factory=dict)
    dense_backend: str = ""


@dataclass(frozen=True)
class RawRouteOutput:
    """What a route produces before the canonical ingest step.

    Paths are workspace-local; the adapter turns them into a
    :class:`MeshArtifact` in the canonical W frame.
    """

    mesh_path: Path
    dense_cloud_path: Path | None = None
    logs_path: Path | None = None


@dataclass(frozen=True)
class MeshArtifact:
    """Canonical mesh metadata describing an exported mesh bundle.

    ``frame`` is always ``"W"`` and ``units`` always ``"m"``; the defaults are
    part of the frozen contract, not a convenience.
    """

    path: Path
    fmt: str
    vertex_count: int
    triangle_count: int
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    sha256: str
    route: str
    config_sha256: str
    components: dict
    degenerate: dict
    frame: str = "W"
    units: str = "m"
    viewer_glb: Path | None = None


@runtime_checkable
class MeshReconstructor(Protocol):
    """The single adapter boundary every reconstruction route implements."""

    @property
    def name(self) -> str:
        ...

    def describe(self) -> RouteDescriptor:
        ...

    def build(
        self,
        window: WindowSpec,
        workdir: Path,
        dense_cloud: Path | None = None,
    ) -> Path:
        ...


__all__ = [
    "ContractError",
    "MeshArtifact",
    "MeshReconstructor",
    "RawRouteOutput",
    "RouteDescriptor",
    "WindowSpec",
]
