"""Lazy route resolution plus the pipeline ``build`` stage entry.

Route modules import the OpenMVS helpers lazily, so the package never imports
them eagerly.  A missing implementation is reported as
:class:`RouteUnavailable` with the underlying reason.

Route selection is *config only*: the YAML ``route:`` value (or an explicit
``--route`` override for a single run) is the sole input.  No code path
hard-codes a route.
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import config_sha256, load_config
from ..contracts import RouteDescriptor, WindowSpec

ROUTES: dict[str, str] = {
    "colmap_openmvs": "reconstruct.mesh.routes.colmap_openmvs:ColmapOpenMVS",
}

# The canonical COLMAP -> OpenMVS route is the only supported reconstruction
# method.
STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "colmap_openmvs": ("model", "build"),
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


def route_stages(route_name: str) -> tuple[str, ...]:
    """Return the pipeline stages a route needs before ``build``."""
    if route_name not in ROUTES:
        raise RouteUnavailable(f"unknown route: {route_name}")
    return STAGE_SEQUENCES.get(route_name, ())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build(
    route_name: str,
    *,
    bundle_root: str,
    window_name: str,
    frames: Sequence[int],
    cfg: Mapping,
    workdir: str,
    env: Mapping,
    colmap_result: Mapping | None = None,
    dense_result: Mapping | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Construct the configured route, run it, and describe its output.

    Returns the dict consumed by the pipeline (persisted as ``stages/build.json``)
    and re-exported under ``route_result`` for the ingest stage.
    """
    route_class = resolve_route(route_name)
    route = route_class(
        cfg=cfg,
        env=env,
        bundle_root=bundle_root,
        window_name=window_name,
        frames=tuple(int(frame) for frame in frames),
        colmap_result=colmap_result,
        dense_result=dense_result,
    )
    descriptor: RouteDescriptor = route.describe()

    window = WindowSpec(window_name, tuple(int(frame) for frame in frames))
    output = route.build(window, Path(workdir), None)
    report = getattr(route, "report", None) or {}
    report_path = getattr(route, "report_path", None)

    payload: dict[str, Any] = {
        "route": route_name,
        "window": window_name,
        "frames": [int(frame) for frame in frames],
        "descriptor": {
            "route": descriptor.route,
            "config_sha256": descriptor.config_sha256,
            "tool_versions": dict(descriptor.tool_versions),
            "dense_backend": descriptor.dense_backend,
        },
        "mesh_path": str(output.mesh_path),
        "dense_cloud_path": (
            str(output.dense_cloud_path) if output.dense_cloud_path else None
        ),
        "logs_path": str(output.logs_path) if output.logs_path else None,
        "report_path": str(report_path) if report_path else None,
        "report": report,
        "config_sha256": config_sha256(cfg),
    }
    payload["route_result"] = dict(payload)
    return payload


# --------------------------------------------------------------------------- #
# Route-smoke preflight / fallback wiring
# --------------------------------------------------------------------------- #


def route_smoke_preflight(
    route_name: str | None,
    config_path: str | Path,
    out_root: str | Path | None = None,
) -> tuple[int, str]:
    """Validate a ``--route-smoke`` request; return ``(exit_code, message)``."""
    cfg = load_config(config_path)
    requested = route_name or str(cfg["route"])
    if requested not in ROUTES:
        return 2, f"unknown route: {requested}"
    return 0, ""


def _ab_row(route_name: str, build_payload: Mapping) -> dict[str, Any]:
    report = build_payload.get("report") or {}
    mesh = report.get("mesh") or {}
    descriptor = build_payload.get("descriptor") or {}
    return {
        "route": route_name,
        "status": "ok",
        "mesh_path": mesh.get("path") or build_payload.get("mesh_path"),
        "mesh_sha256": mesh.get("sha256"),
        "vertex_count": mesh.get("vertex_count"),
        "triangle_count": mesh.get("triangle_count"),
        "bbox": {
            "min_m": mesh.get("bbox_min_m"),
            "max_m": mesh.get("bbox_max_m"),
        },
        "wall_time_s": report.get("wall_time_s"),
        "peak_rss_mb": report.get("peak_rss_mb"),
        "dense_backend": report.get("dense_backend") or descriptor.get("dense_backend"),
        "dense_cloud_path": build_payload.get("dense_cloud_path"),
        "config_sha256": descriptor.get("config_sha256")
        or build_payload.get("config_sha256"),
    }


def write_ab_from_stages(
    out_root: str | Path,
    run_ids: Mapping[str, str],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Read each route's ``stages/build.json`` and write ``ab_smoke.json``."""
    out_root = Path(out_root)
    rows = []
    for route_name, run_id in run_ids.items():
        stage_path = out_root / "runs" / run_id / "smoke" / "stages" / "build.json"
        if not stage_path.is_file():
            raise RouteUnavailable(f"missing build stage record: {stage_path}")
        try:
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RouteUnavailable(f"corrupt build stage record {stage_path}: {exc}") from exc
        rows.append(_ab_row(route_name, payload))

    record = {
        "schema_version": 1,
        "window": "smoke",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": dict(provenance or {}),
        "routes": rows,
    }
    path = out_root / "ab_smoke.json"
    _write_json(path, record)
    return path


__all__ = [
    "ROUTES",
    "STAGE_SEQUENCES",
    "RouteUnavailable",
    "build",
    "resolve_route",
    "route_smoke_preflight",
    "route_stages",
    "write_ab_from_stages",
]
