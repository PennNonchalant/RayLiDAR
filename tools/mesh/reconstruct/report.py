"""A/B evidence, canonical asset promotion, local index, and handoff note.

This module is the last Phase-4 boundary.  It never re-runs reconstruction: it
reads the persisted stage records, writes ``ab_report.md`` with the D-12
metrics, and copies the canonical PLY/GLB/OBJ and viewer GLB into
``/data-20T/hybrid-lidar-sim/assets/mesh/<run-id>/``, re-hashing every copied
byte, writing ``SHA256SUMS`` and merging the local ``index.json``.

G3 measurements are diagnostic only.  The reconstruction package does not
define a hard pass/fail gate and never refuses promotion because a reference
value was missed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_ASSETS_ROOT = Path("/data-20T/hybrid-lidar-sim/assets/mesh")
DEFAULT_OUT_ROOT = "/data-20T/hybrid-lidar-sim/artifacts/04"
ASSETS_ROOT_ENV = "RAYLIDAR_ASSETS_ROOT"

CANONICAL_MESH_FILES: tuple[str, ...] = ("mesh.ply", "mesh.glb", "mesh.obj")
VIEWER_FILE = "mesh_viewer.glb"
MANIFEST_FILE = "mesh_manifest.json"
SUMS_FILE = "SHA256SUMS"
INVALID_MARKER = ".invalid"
RESOURCE_LIMIT = "not_run_resource_limit"

# D-04: the Open3D A/B route shares the dense cloud, so the comparison is not
# an independent MVS comparison.  The sentence is fixed for the report.
MESHER_COMPARISON_SENTENCE = (
    "The A/B comparison is a mesher comparison on the same dense point cloud, "
    "not an independent MVS comparison."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def assets_root(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    env_value = os.environ.get(ASSETS_ROOT_ENV)
    if env_value:
        return Path(env_value)
    return Path(DEFAULT_ASSETS_ROOT)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Route summaries (D-12 fields, read from persisted stage records)
# --------------------------------------------------------------------------- #


def summarise_run(
    workdir: str | Path,
    *,
    route_name: str | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    """Build one A/B row from a window workdir's persisted stage records."""
    workdir = Path(workdir)
    build = _read_json(workdir / "stages" / "build.json") or {}
    ingest = _read_json(workdir / "stages" / "ingest.json") or {}
    gate = _read_json(workdir / "g3_report.json") or {}
    resource = _read_json(workdir / "RESOURCE_LIMIT.json") or {}

    report = build.get("report") or {}
    descriptor = build.get("descriptor") or {}
    mesh = report.get("mesh") or {}
    artifact = ingest.get("artifact") or {}
    components = artifact.get("components") or (gate.get("metrics") or {}).get(
        "components"
    ) or {}
    degenerate = artifact.get("degenerate") or (gate.get("metrics") or {}).get(
        "degenerate"
    ) or {}
    metrics = gate.get("metrics") or {}

    if resource:
        status = str(resource.get("status") or RESOURCE_LIMIT)
        error = error or str(resource.get("error") or "")
    elif not build and status == "ok":
        status = "not_run"
        error = error or f"missing build stage record: {workdir / 'stages' / 'build.json'}"

    outputs = report.get("outputs") or {}
    dense_cloud_entry = outputs.get("dense_cloud") or {}
    dense_cloud_path = (
        dense_cloud_entry.get("path")
        or build.get("dense_cloud_path")
        or ""
    )
    dense_name = Path(str(dense_cloud_path)).name if dense_cloud_path else ""
    if dense_name == "scene_dense.ply":
        dense_cloud_source = "openmvs_densify"
    else:
        dense_cloud_source = (
            report.get("dense_backend")
            or descriptor.get("dense_backend")
            or artifact.get("dense_backend")
        )

    return {
        "route": str(route_name or build.get("route") or artifact.get("route") or ""),
        "run_id": workdir.parent.name if workdir.parent.name else workdir.name,
        "window": build.get("window") or gate.get("window") or workdir.name,
        "status": status,
        "error": error,
        "config_sha256": (
            descriptor.get("config_sha256")
            or build.get("config_sha256")
            or artifact.get("config_sha256")
        ),
        "mesh_sha256": artifact.get("sha256") or mesh.get("sha256"),
        "source_mesh_sha256": mesh.get("sha256") or artifact.get("sha256"),
        "vertex_count": artifact.get("vertex_count") or mesh.get("vertex_count"),
        "triangle_count": artifact.get("triangle_count") or mesh.get("triangle_count"),
        "bbox_min_m": artifact.get("bbox_min_m") or mesh.get("bbox_min_m"),
        "bbox_max_m": artifact.get("bbox_max_m") or mesh.get("bbox_max_m"),
        "components_count": components.get("count"),
        "largest_component_fraction": components.get("largest_fraction"),
        "degenerate_fraction": degenerate.get("fraction"),
        "build_wall_time_s": report.get("wall_time_s"),
        "peak_rss_mb": report.get("peak_rss_mb"),
        "dense_backend": (
            report.get("dense_backend")
            or descriptor.get("dense_backend")
            or artifact.get("dense_backend")
        ),
        "dense_cloud_source": dense_cloud_source,
        "dense_cloud_path": str(dense_cloud_path) if dense_cloud_path else None,
        "g3_status": gate.get("status") or ("measured" if gate else None),
        "g3_measurements": gate.get("measurements") or {},
        "g3_metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _bbox(route: Mapping[str, Any]) -> str:
    low = route.get("bbox_min_m")
    high = route.get("bbox_max_m")
    if not isinstance(low, list) or not isinstance(high, list):
        return "-"
    try:
        return (
            "[" + ", ".join(f"{float(value):.2f}" for value in low) + " .. "
            + ", ".join(f"{float(value):.2f}" for value in high) + "]"
        )
    except (TypeError, ValueError):
        return "-"


def _check_metric(route: Mapping[str, Any], name: str, key: str) -> Any:
    measurement = (route.get("g3_measurements") or {}).get(name) or {}
    measured = measurement.get("measured") or {}
    return measured.get(key)


def _z_band_state(route: Mapping[str, Any]) -> str:
    """Render the G3-AMEND-01 z sanity band for one route."""
    band = (
        ((route.get("g3_measurements") or {}).get("bbox_ratio_ok") or {}).get(
            "measured"
        )
        or {}
    ).get("height_band")
    if not isinstance(band, dict):
        return "-"
    ok = bool(band.get("z_min_ok")) and bool(band.get("z_max_ok")) and bool(
        band.get("overlap")
    )
    return "within" if ok else "outside"


def _ab_markdown(
    artifact: Mapping[str, Any],
    g3_report: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    run_id: str | None,
) -> str:
    thresholds = g3_report.get("reference_thresholds") or {}
    directions = (
        (g3_report.get("metrics") or {}).get("camera_outside") or {}
    ).get("directions") or {}
    smoke_frames = g3_report.get("smoke_range_frames") or (cfg.get("windows") or {}).get(
        "smoke"
    ) or []
    canonical_backend = None
    for route in routes:
        if route.get("route") == cfg.get("route"):
            canonical_backend = route.get("dense_cloud_source") or route.get("dense_backend")
            break
    if canonical_backend is None and routes:
        canonical_backend = routes[0].get("dense_cloud_source") or routes[0].get(
            "dense_backend"
        )
    sources = {
        route.get("dense_cloud_source") or route.get("dense_backend")
        for route in routes
    }
    sources.discard(None)

    lines: list[str] = []
    lines.append("# Phase 4 A/B reconstruction report")
    lines.append("")
    lines.append(f"- Generated: `{_utc_now()}`")
    lines.append(f"- Canonical run: `{run_id or '-'}` (route `{cfg.get('route')}`)")
    gt_frames = g3_report.get("gt_frames") or ["-"]
    lines.append(
        f"- Windows: smoke `{smoke_frames[0] if smoke_frames else '-'}-"
        f"{smoke_frames[-1] if smoke_frames else '-'}` (D-02); "
        f"full `{gt_frames[0]}-{gt_frames[-1]}`"
    )
    lines.append(
        "- Frame and units: `W` (KITTI-360 world), metres; no transform, no "
        "rescaling, no fitting was applied to any mesh (`alignment_applied: false`)."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        f"The canonical route is `{cfg.get('route')}`; its dense backend is "
        f"`{canonical_backend or 'unknown'}`. No mesh was aligned, scaled, or "
        "fitted to the camera trajectory or to the GT extent."
    )
    lines.append("")
    if sources == {"openmvs_densify"}:
        lines.append(MESHER_COMPARISON_SENTENCE)
        lines.append("")
    elif sources == {"colmap_cuda"}:
        lines.append(
            "Both runnable routes mesh the same COLMAP CUDA PatchMatch fused "
            "cloud, so the comparison is a mesher comparison on one dense input; "
            "it is not an independent MVS comparison."
        )
        lines.append("")
    else:
        lines.append(
            "The routes do not share a dense cloud: the canonical route densifies "
            "with OpenMVS `DensifyPointCloud` (`scene_dense.ply`), while the Open3D "
            "route meshes the COLMAP CUDA PatchMatch fused cloud (`fused.ply`). "
            "The A/B therefore differs in both densification and meshing; it is "
            "not an independent MVS comparison either way."
        )
        lines.append("")
    lines.append("## Route comparison")
    lines.append("")
    lines.append(
        "| route | status | config sha256 | mesh sha256 | vertices | triangles "
        "| bbox (m) | components | largest fraction | degenerate fraction "
        "| build wall (s) | peak RSS (MB) | dense backend | dense cloud source |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for route in routes:
        sha = route.get("mesh_sha256")
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(route.get("route")),
                    _cell(route.get("status")),
                    _cell((route.get("config_sha256") or "-")[:16]),
                    _cell((sha or "-")[:16]),
                    _cell(route.get("vertex_count")),
                    _cell(route.get("triangle_count")),
                    _cell(_bbox(route)),
                    _cell(route.get("components_count")),
                    _cell(route.get("largest_component_fraction")),
                    _cell(route.get("degenerate_fraction")),
                    _cell(route.get("build_wall_time_s")),
                    _cell(route.get("peak_rss_mb")),
                    _cell(route.get("dense_backend")),
                    _cell(route.get("dense_cloud_source")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## G3 diagnostics (full window; reference values are not enforced)")
    lines.append("")
    lines.append(
        "| route | status | bbox ratio (x,y) | z band | forward hit rate | near hits "
        "| components | largest fraction | degenerate fraction | smoke median (m) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for route in routes:
        ratio = _check_metric(route, "bbox_ratio_ok", "ratio")
        if isinstance(ratio, list):
            ratio_text = ", ".join(
                "-" if value is None else f"{float(value):.3f}" for value in ratio
            )
        else:
            ratio_text = "-"
        near = _check_metric(route, "camera_outside_ok", "near_hit_count")
        median = _check_metric(route, "smoke_range_ok", "median_abs_error_m")
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(route.get("route")),
                    _cell(route.get("g3_status")),
                    _cell(ratio_text),
                    _cell(_z_band_state(route)),
                    _cell(_check_metric(route, "camera_outside_ok", "forward_hit_rate")),
                    _cell(near),
                    _cell(_check_metric(route, "components_ok", "component_count")),
                    _cell(_check_metric(route, "components_ok", "largest_component_fraction")),
                    _cell(_check_metric(route, "degenerate_ok", "degenerate_fraction")),
                    _cell(median),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Frozen G3 definitions")
    lines.append("")
    amendment = g3_report.get("amendment") or {}
    if amendment.get("id"):
        lines.append(
            f"- G3 amendment `{amendment.get('id')}` "
            f"(approved {amendment.get('approved_at_utc') or '-'}): the GT extent "
            "is the camera-visible static subset "
            f"(`camera < {thresholds.get('gt_visible_camera_max', '-')}`); x/y "
            "footprint plus a z sanity band; forward hit rate has a lower bound "
            "only; the largest-component threshold is "
            f"`{thresholds.get('largest_component_fraction_min', '-')}`."
        )
        lines.append("")
    lines.append("Camera-outside directions, camera frame (`+z` optical axis), D-07:")
    lines.append("")
    for name, direction in directions.items():
        lines.append(
            f"- `{name}`: [{', '.join(f'{float(v):.4f}' for v in direction)}]"
        )
    lines.append("")
    lines.append("Reference values (diagnostic only; not enforced):")
    lines.append("")
    for key in (
        "bbox_ratio_min",
        "bbox_ratio_max",
        "bbox_axes",
        "mesh_z_min_margin_m",
        "mesh_z_max_margin_m",
        "forward_hit_rate_min",
        "near_hit_max_m",
        "largest_component_fraction_min",
        "degenerate_fraction_max",
        "smoke_median_abs_error_m_max",
        "ray_t_min_m",
        "camera_outside_t_max_m",
        "gt_visible_camera_max",
    ):
        if key in thresholds:
            lines.append(f"- `{key}` = {thresholds[key]}")
    lines.append("")
    lines.append(
        f"Smoke range frames (D-02): `{smoke_frames[0] if smoke_frames else '-'}`"
        f"-`{smoke_frames[-1] if smoke_frames else '-'}`."
    )
    lines.append("")
    lines.append("## What this does not show")
    lines.append("")
    lines.append(
        "- No full-window range, Chamfer distance, F-score, or ray recall: those "
        "are Phase 5/6 metrics computed on GPU hits."
    )
    lines.append(
        "- No textured or photorealistic realism: RGB is reconstruction input, not "
        "a material model."
    )
    lines.append(
        "- No independent MVS comparison: the canonical route densifies with "
        "OpenMVS and the Open3D route meshes the COLMAP CUDA cloud; neither was "
        "compared against an independent MVS implementation."
    )
    lines.append(
        "- No alignment evidence: nothing was aligned, fitted, or rescaled; a "
        "failing route stayed failing."
    )
    lines.append("")
    return "\n".join(lines)


def _handoff_markdown(
    artifact: Mapping[str, Any],
    g3_report: Mapping[str, Any],
    run_id: str,
    asset_dir: Path,
    files: Mapping[str, str],
) -> str:
    manifest = asset_dir / MANIFEST_FILE
    lines = [
        "# Canonical mesh handoff",
        "",
        f"- Run: `{run_id}`",
        f"- Asset directory: `{asset_dir}`",
        f"- Canonical PLY: `{asset_dir / 'mesh.ply'}`",
        f"- Canonical GLB: `{asset_dir / 'mesh.glb'}`",
        f"- Canonical OBJ: `{asset_dir / 'mesh.obj'}`",
        f"- Viewer GLB (decimated copy, not canonical): `{asset_dir / VIEWER_FILE}`",
        f"- Manifest: `{manifest}`",
        f"- Checksums: `{asset_dir / SUMS_FILE}`",
        "",
        "## Frame and units",
        "",
        "- Frame: `W` (KITTI-360 world frame, right-handed, x forward, y left, z up).",
        "- Units: metres.",
        "- GLB nodes carry identity transforms; OBJ contains triangles only.",
        (
            "- Mesh SHA-256 (PLY): `"
            + str(artifact.get("sha256") or files.get("mesh.ply") or "-")
            + "`"
        ),
        "- G3 diagnostics: recorded (no hard pass/fail criterion; see g3_report.json).",
        "",
        "## Validation",
        "",
        "```bash",
        f"raylidar_cli mesh validate --manifest {manifest} "
        f"--run-dir {asset_dir} --verify-hashes",
        "```",
        "",
        "## Consumer contract",
        "",
        "Member 1 (Vulkan/BLAS) and member 2 (GPU Ray Query) load the canonical "
        "`mesh.ply`/`mesh.glb`/`mesh.obj` directly. There is no conversion step, "
        "no re-scaling, and no frame change: the files are already `W` in metres.",
        "",
    ]
    return "\n".join(lines)


def write(
    artifact: Mapping[str, Any],
    g3_report: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    workdir: str,
    env: Mapping[str, Any],
    *,
    run_id: str | None = None,
    asset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``ab_report.md`` and ``HANDOFF.md``; return the paths."""
    out_root = Path(str(cfg.get("out_root") or DEFAULT_OUT_ROOT))
    routes = list(routes or [])
    if not routes:
        routes = [summarise_run(workdir)]

    sources = {
        route.get("dense_cloud_source") or route.get("dense_backend")
        for route in routes
    }
    sources.discard(None)
    fallback_active = sources == {"openmvs_densify"}

    ab_text = _ab_markdown(artifact, g3_report, routes, cfg, run_id)
    ab_path = out_root / "ab_report.md"
    _write_text(ab_path, ab_text)

    handoff_path = out_root / "HANDOFF.md"
    if run_id:
        destination = Path(asset_dir) if asset_dir else assets_root() / str(run_id)
        _write_text(
            handoff_path,
            _handoff_markdown(artifact, g3_report, str(run_id), destination, {}),
        )
    elif handoff_path.exists():
        handoff_path.unlink()

    return {
        "ab_report_path": str(ab_path),
        "handoff_path": str(handoff_path) if handoff_path.exists() else None,
        "run_id": run_id,
        "route_count": len(routes),
        "fallback_active": fallback_active,
        "g3_status": g3_report.get("status") or "measured",
    }


# --------------------------------------------------------------------------- #
# Promotion and the local index
# --------------------------------------------------------------------------- #


def _manifest_file_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    files = manifest.get("files") or {}
    for entry in files.values():
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        digest = entry.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            hashes[Path(path).name] = digest
    viewer = manifest.get("viewer") or {}
    if isinstance(viewer, dict) and viewer.get("path") and viewer.get("sha256"):
        hashes[Path(str(viewer["path"])).name] = str(viewer["sha256"])
    return hashes


def promote(
    run_dir: str | Path,
    run_id: str,
    g3_report: Mapping[str, Any],
    manifest_path: str | Path,
    *,
    assets: str | Path | None = None,
) -> dict[str, Any]:
    """Copy the canonical artifact into the asset root after re-hashing it."""
    run_dir = Path(run_dir)
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest is None:
        raise RuntimeError(f"cannot promote: manifest is missing or invalid: {manifest_path}")

    expected = _manifest_file_hashes(manifest)
    required = list(CANONICAL_MESH_FILES) + [VIEWER_FILE, MANIFEST_FILE]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"cannot promote: missing canonical file(s) {missing} under {run_dir}"
        )
    absent_hashes = [name for name in required if name != MANIFEST_FILE and name not in expected]
    if absent_hashes:
        raise RuntimeError(
            f"cannot promote: manifest does not hash {absent_hashes}"
        )

    destination = assets_root(assets) / str(run_id)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / INVALID_MARKER
    if marker.exists():
        marker.unlink()

    for name in required:
        shutil.copy2(run_dir / name, destination / name)

    mismatches: list[str] = []
    for name, digest in expected.items():
        copied = destination / name
        if not copied.is_file():
            mismatches.append(f"{name}: missing after copy")
            continue
        actual = _sha256_file(copied)
        if actual != digest:
            mismatches.append(f"{name}: {actual} != {digest}")

    if mismatches:
        _write_text(
            marker,
            "promotion aborted: copied bytes disagree with the manifest\n"
            + "\n".join(mismatches)
            + "\n",
        )
        raise RuntimeError(
            "cannot promote: copied mesh bytes disagree with the manifest: "
            + "; ".join(mismatches)
        )

    # The delivered manifest records the diagnostic measurements it was
    # promoted with; there is no gate verdict.
    delivered_manifest = json.loads((destination / MANIFEST_FILE).read_text(encoding="utf-8"))
    delivered_manifest["g3"] = {
        "status": g3_report.get("status") or "measured",
        "measurements": g3_report.get("measurements") or {},
        "report": g3_report,
    }
    _write_json(destination / MANIFEST_FILE, delivered_manifest)

    checksummed = sorted(set(expected) | {MANIFEST_FILE})
    sums_lines = [
        f"{_sha256_file(destination / name)}  {name}" for name in checksummed
    ]
    _write_text(destination / SUMS_FILE, "\n".join(sums_lines) + "\n")

    files = {
        name: _sha256_file(destination / name)
        for name in required
        if (destination / name).is_file()
    }
    return {
        "asset_dir": str(destination),
        "files": files,
        "manifest_sha256": files.get(MANIFEST_FILE),
        "manifest": delivered_manifest,
    }


def update_index(
    asset_dir: str | Path,
    run_id: str,
    manifest: Mapping[str, Any],
    g3_report: Mapping[str, Any],
    *,
    assets: str | Path | None = None,
) -> Path:
    """Merge this run into ``assets/mesh/index.json`` idempotently."""
    asset_dir = Path(asset_dir)
    root = Path(assets) if assets is not None else asset_dir.parent
    index_path = root / "index.json"
    index = _read_json(index_path) or {}
    if not isinstance(index.get("runs"), list):
        index["runs"] = []
    index["schema_version"] = 1

    files = manifest.get("files") or {}
    ply_sha = ((files.get("ply") or {}).get("sha256")) or None
    manifest_sha = _sha256_file(asset_dir / MANIFEST_FILE)
    entry = {
        "run_id": str(run_id),
        "route": manifest.get("route"),
        "created_at_utc": manifest.get("created_at_utc"),
        "asset_dir": str(asset_dir),
        "mesh_sha256": ply_sha,
        "manifest_sha256": manifest_sha,
        "vertex_count": manifest.get("vertex_count"),
        "triangle_count": manifest.get("triangle_count"),
        "frames": list((manifest.get("inputs") or {}).get("frames") or []),
        "g3_status": g3_report.get("status") or "measured",
        "dense_backend": manifest.get("dense_backend"),
        "url": None,
        "download_sha256": None,
    }

    runs = [run for run in index["runs"] if str(run.get("run_id")) != str(run_id)]
    runs.append(entry)
    runs.sort(key=lambda run: str(run.get("run_id")))
    index["runs"] = runs
    _write_json(index_path, index)
    return index_path


# --------------------------------------------------------------------------- #
# CLI used by scripts/phase4.sh --promote
# --------------------------------------------------------------------------- #


def _discover_routes(
    out_root: Path,
    pinned: Mapping[str, str],
    exclude_run_id: str | None,
) -> list[dict[str, Any]]:
    """Find the newest full-window run per runnable route."""
    if pinned:
        rows = []
        for route, run_id in pinned.items():
            workdir = out_root / "runs" / str(run_id) / "full"
            rows.append(summarise_run(workdir, route_name=route))
        return rows

    newest: dict[str, tuple[float, Path]] = {}
    runs_dir = out_root / "runs"
    if runs_dir.is_dir():
        for candidate in sorted(runs_dir.iterdir()):
            workdir = candidate / "full"
            build = workdir / "stages" / "build.json"
            resource = workdir / "RESOURCE_LIMIT.json"
            if not build.is_file() and not resource.is_file():
                continue
            if exclude_run_id and candidate.name == exclude_run_id:
                continue
            route = None
            payload = _read_json(build) if build.is_file() else None
            if payload:
                route = payload.get("route")
            if not route:
                route = candidate.name.split("-")[0]
            stamp = max(
                path.stat().st_mtime
                for path in (build, resource)
                if path.is_file()
            )
            if route not in newest or stamp > newest[route][0]:
                newest[route] = (stamp, workdir)
    return [
        summarise_run(workdir, route_name=route)
        for route, (_, workdir) in sorted(newest.items())
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.mesh.reconstruct.report",
        description="Phase-4 A/B report and canonical asset promotion",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", help="window workdir (default: out_root/runs/<id>/full)")
    parser.add_argument("--assets-root")
    parser.add_argument(
        "--ab-route",
        action="append",
        default=[],
        metavar="NAME=RUN_ID",
        help="pin an A/B route run instead of discovering it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from .config import load_config

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_root = Path(str(cfg["out_root"]))
    workdir = (
        Path(args.workdir)
        if args.workdir
        else out_root / "runs" / args.run_id / "full"
    )
    gate_path = workdir / "g3_report.json"
    g3_report = _read_json(gate_path)
    if g3_report is None:
        print(f"missing gate report: {gate_path}", file=sys.stderr)
        return 3
    run_dir = workdir / "canonical"
    manifest_path = run_dir / MANIFEST_FILE
    artifact = (_read_json(workdir / "stages" / "ingest.json") or {}).get("artifact")
    if artifact is None:
        print(f"missing ingest stage record under {workdir}", file=sys.stderr)
        return 3

    pinned: dict[str, str] = {}
    for item in args.ab_route:
        name, _, run_id = item.partition("=")
        if name and run_id:
            pinned[name] = run_id
    routes = _discover_routes(out_root, pinned, args.run_id)
    canonical_route = str(cfg.get("route"))
    routes = [row for row in routes if row.get("route") != canonical_route]
    routes.insert(0, summarise_run(workdir, route_name=canonical_route))

    written = write(
        artifact,
        g3_report,
        routes,
        cfg,
        str(workdir),
        {},
        run_id=args.run_id,
        asset_dir=assets_root(args.assets_root) / args.run_id,
    )
    print(f"ab_report: {written['ab_report_path']}")

    promoted = promote(
        run_dir,
        args.run_id,
        g3_report,
        manifest_path,
        assets=args.assets_root,
    )
    index_path = update_index(
        promoted["asset_dir"],
        args.run_id,
        promoted["manifest"],
        g3_report,
        assets=args.assets_root,
    )
    print(f"asset_dir: {promoted['asset_dir']}")
    print(f"index: {index_path}")
    if written["handoff_path"]:
        print(f"handoff: {written['handoff_path']}")
    return 0


__all__ = [
    "DEFAULT_ASSETS_ROOT",
    "DEFAULT_OUT_ROOT",
    "MESHER_COMPARISON_SENTENCE",
    "RESOURCE_LIMIT",
    "assets_root",
    "main",
    "promote",
    "summarise_run",
    "update_index",
    "write",
]

if __name__ == "__main__":
    raise SystemExit(main())
