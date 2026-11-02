"""Route-agnostic canonical ingest: one adapter for every reconstruction route.

Routes only produce a raw mesh path.  This module is the single boundary that
turns that path into the canonical Phase-4 artifact (RECON-03, D-10):

1. load the raw mesh and require finite float64 W-frame metre geometry,
2. assert the metric/frame envelope **without** translating, rotating, scaling,
   or fitting anything (a failure is loud, never "repaired"),
3. weld, drop degenerates/specks, and drop unreferenced vertices,
4. export the full canonical PLY/GLB/OBJ plus a separate decimated viewer GLB,
5. build the manifest from the written bytes and hand it to the C++ gate.

There is deliberately no code path that aligns a mesh to the camera trajectory
or to GT.  The only evaluation-side data used here is the camera-centre AABB
taken from the P2 bundle, and it is used solely as a rejection envelope.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import cleanup, manifest, meshio, validate
from .config import config_sha256

# A metric street mesh is tens of metres; 500 m per axis means a unit bug.
MESH_EXTENT_LIMIT_M = 500.0
# The mesh must overlap the observed corridor, not drift elsewhere in W.
CAMERA_ENVELOPE_MARGIN_M = 100.0
DEFAULT_VIEWER_MAX_TRIANGLES = 1_000_000

# Reporting metadata only: behaviour never branches on the route name.  The
# authoritative provenance field is ``config_sha256``.
_CONFIG_PATHS = {
    "colmap_openmvs": "tools/mesh/reconstruct/configs/canonical_colmap_openmvs.yaml",
    "open3d_alpha": "tools/mesh/reconstruct/configs/open3d_alpha.yaml",
    "openmvs_direct": "tools/mesh/reconstruct/configs/openmvs_direct.yaml",
}


class IngestError(RuntimeError):
    """The route mesh cannot become a canonical W-frame metre artifact."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _camera_centres(bundle_root: str | Path, frames: Sequence[int]) -> np.ndarray:
    """Return one W-frame camera centre per requested frame.

    The P2 export already carries ``T_W_image_00_rect``; its translation is the
    camera centre in W.  ``cameras.json`` is preferred, with ``frames.csv`` as
    the equivalent fallback.  Missing frames are an error: silently dropping a
    frame would weaken the envelope assertion.
    """
    root = Path(bundle_root)
    wanted = [int(frame) for frame in frames]
    if not wanted:
        raise IngestError("ingest requires at least one frame for the envelope")

    centres: dict[int, list[float]] = {}
    cameras_path = root / "geometry" / "cameras.json"
    if cameras_path.is_file():
        payload = json.loads(cameras_path.read_text(encoding="utf-8"))
        for entry in payload.get("frames") or []:
            frame = entry.get("frame")
            matrix = (
                entry.get("T_W_image_00_rect")
                or entry.get("T_W_Ck_rect")
                or entry.get("T_W_C0_rect")
            )
            if frame is None or matrix is None:
                continue
            centres[int(frame)] = [
                float(matrix[0][3]),
                float(matrix[1][3]),
                float(matrix[2][3]),
            ]

    frames_csv = root / "frames.csv"
    if frames_csv.is_file():
        with frames_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    frame = int(row["frame"])
                except (KeyError, TypeError, ValueError):
                    continue
                if frame in centres:
                    continue
                try:
                    centres[frame] = [
                        float(row["image_00_rect_tx"]),
                        float(row["image_00_rect_ty"]),
                        float(row["image_00_rect_tz"]),
                    ]
                except (KeyError, TypeError, ValueError):
                    continue

    missing = [frame for frame in wanted if frame not in centres]
    if missing:
        raise IngestError(
            f"no camera geometry for frames {missing} under {root}; "
            "the W/metre envelope assertion cannot run"
        )
    return np.asarray([centres[frame] for frame in wanted], dtype=np.float64)


def _load_route_mesh(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly one triangle mesh as float64 vertices and int64 faces."""
    import trimesh  # imported lazily; ingest is only needed by the ingest stage

    path = Path(mesh_path)
    if not path.is_file():
        raise IngestError(f"route mesh is missing: {path}")
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if len(geometries) != 1:
            raise IngestError(
                f"route mesh must contain exactly one geometry, got {len(geometries)}"
            )
        loaded = geometries[0]
    if not isinstance(loaded, trimesh.Trimesh):
        raise IngestError(f"route mesh is not a triangle mesh: {path}")

    vertices = np.ascontiguousarray(np.asarray(loaded.vertices, dtype=np.float64))
    faces = np.asarray(loaded.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise IngestError("route mesh vertices must have shape (N, 3)")
    if faces.size == 0:
        raise IngestError("route mesh has no triangles")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise IngestError("route mesh faces must be triangles")
    if not np.issubdtype(faces.dtype, np.integer):
        raise IngestError("route mesh face indices must be integers")
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    if vertices.shape[0] == 0:
        raise IngestError("route mesh has no vertices")
    if not np.isfinite(vertices).all():
        raise IngestError("route mesh has non-finite vertex coordinates")
    if faces.min() < 0 or faces.max() >= vertices.shape[0]:
        raise IngestError("route mesh has an out-of-range face index")
    return vertices, faces


def _assert_metre_w(vertices: np.ndarray, centres: np.ndarray) -> dict[str, Any]:
    """Reject anything that is not a finite, metre, W-frame mesh at the corridor.

    This is the anti-alignment gate: it measures and rejects; it never repairs.
    """
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extent = bbox_max - bbox_min
    envelope_min = centres.min(axis=0) - CAMERA_ENVELOPE_MARGIN_M
    envelope_max = centres.max(axis=0) + CAMERA_ENVELOPE_MARGIN_M

    reasons: list[str] = []
    if np.any(extent >= MESH_EXTENT_LIMIT_M):
        reasons.append(
            f"bbox extent {extent.tolist()} exceeds {MESH_EXTENT_LIMIT_M:g} m per axis"
        )
    intersects = bool(
        np.all(bbox_min <= envelope_max) and np.all(bbox_max >= envelope_min)
    )
    inside_any = bool(
        np.any(
            np.all(
                (vertices >= envelope_min) & (vertices <= envelope_max),
                axis=1,
            )
        )
    )
    if not intersects:
        reasons.append("mesh AABB does not intersect the camera-centre envelope")
    if not inside_any:
        reasons.append("no mesh vertex lies inside the camera-centre envelope")

    if reasons:
        raise IngestError(
            "route mesh is not a metric W-frame mesh at the observed corridor: "
            + "; ".join(reasons)
            + f" | mesh_bbox={bbox_min.tolist()}..{bbox_max.tolist()}"
            + f" | camera_envelope={envelope_min.tolist()}..{envelope_max.tolist()}"
        )
    return {
        "bbox_min_m": [float(value) for value in bbox_min],
        "bbox_max_m": [float(value) for value in bbox_max],
        "camera_envelope_min_m": [float(value) for value in envelope_min],
        "camera_envelope_max_m": [float(value) for value in envelope_max],
    }


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                # The container runs as root against a bind-mounted repo owned
                # by the host user; without this, git refuses with "dubious
                # ownership" and the manifest would silently lose its git SHA.
                "-c",
                f"safe.directory={validate.REPO_ROOT}",
                "-C",
                str(validate.REPO_ROOT),
                "rev-parse",
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _dense_cloud_sha256(route_result: Mapping[str, Any]) -> str:
    report = route_result.get("report") or {}
    outputs = report.get("outputs") or {}
    entry = outputs.get("dense_cloud") or {}
    digest = entry.get("sha256") or report.get("dense_cloud_sha256")
    if isinstance(digest, str) and len(digest) == 64:
        return digest
    dense_path = route_result.get("dense_cloud_path")
    if dense_path and Path(str(dense_path)).is_file():
        return _sha256_file(Path(str(dense_path)))
    return ""


def ingest(
    route_result: Mapping[str, Any],
    bundle_root: str,
    window_name: str,
    frames: Sequence[int],
    cfg: Mapping[str, Any],
    workdir: str,
    env: Mapping[str, Any],
) -> dict[str, Any]:
    """Ingest one route mesh into the canonical W-frame metre artifact.

    Returns ``{"artifact": ..., "manifest_path": ..., "report_path": ...}``.
    The canonical files, ``mesh_manifest.json``, ``SHA256SUMS``, and
    ``ingest_report.json`` are written under ``<workdir>/canonical/``.
    """
    started = time.time()
    route = str(route_result.get("route") or cfg.get("route") or "")
    if not route:
        raise IngestError("route mesh has no route name")

    mesh_path = route_result.get("mesh_path")
    if not mesh_path:
        raise IngestError("route result has no mesh_path")

    vertices, faces = _load_route_mesh(mesh_path)
    centres = _camera_centres(bundle_root, frames)
    envelope = _assert_metre_w(vertices, centres)
    source_stats = meshio.mesh_stats(vertices, faces)
    source_sha256 = _sha256_file(Path(str(mesh_path)))

    params = cfg.get("params") or {}
    cleanup_params = params.get("cleanup") or {}
    viewer_params = params.get("viewer") or {}
    min_component_triangles = int(
        cleanup_params.get(
            "min_component_triangles", cleanup.DEFAULT_MIN_COMPONENT_TRIANGLES
        )
    )
    target_triangles = int(
        viewer_params.get("max_triangles", DEFAULT_VIEWER_MAX_TRIANGLES)
    )

    cleaned = cleanup.clean_mesh(
        vertices, faces, min_component_triangles=min_component_triangles
    )
    stats = cleaned["stats"]

    canonical_dir = Path(workdir) / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    written = meshio.write_all(
        canonical_dir, "mesh", cleaned["vertices"], cleaned["faces"]
    )
    ply_path = canonical_dir / written["files"]["ply"]["path"]

    # The canonical mesh is never decimated (D-11).  The viewer copy is a
    # separate file; when the canonical is already small it is a plain copy and
    # ``applied`` records that honestly.
    applied = bool(stats["triangle_count"] > target_triangles)
    if applied:
        viewer_vertices, viewer_faces = cleanup.decimate(
            cleaned["vertices"], cleaned["faces"], target_triangles
        )
    else:
        viewer_vertices, viewer_faces = cleaned["vertices"], cleaned["faces"]
    viewer_path = canonical_dir / "mesh_viewer.glb"
    meshio.write_glb(viewer_path, viewer_vertices, viewer_faces)
    viewer_vertex_count, viewer_triangle_count, _, _ = meshio.read_back_counts(
        viewer_path, "glb"
    )
    viewer_sha256 = _sha256_file(viewer_path)
    decimation = {
        "canonical_decimated": False,
        "viewer": {
            "applied": applied,
            "method": "quadric",
            "target_triangles": int(target_triangles),
            "vertex_count": int(viewer_vertex_count),
            "triangle_count": int(viewer_triangle_count),
        },
    }

    report = route_result.get("report") or {}
    descriptor = route_result.get("descriptor") or {}
    config_sha = (
        route_result.get("config_sha256")
        or descriptor.get("config_sha256")
        or config_sha256(dict(cfg))
    )
    config_path = (
        route_result.get("config_path")
        or _CONFIG_PATHS.get(route)
        or manifest.DEFAULT_CONFIG_PATH
    )
    dense_backend = str(
        descriptor.get("dense_backend")
        or report.get("dense_backend")
        or env.get("dense_backend")
        or ""
    )
    tool_versions = dict(
        descriptor.get("tool_versions") or report.get("versions") or {}
    )
    inputs = {
        "bundle_root": str(bundle_root),
        "frames": [int(frame) for frame in frames],
        "image_camera": str(params.get("image_camera", "image_00")),
        "source_mesh_sha256": source_sha256,
        "dense_cloud_sha256": _dense_cloud_sha256(route_result),
    }

    built = manifest.build_manifest(
        {
            "route": route,
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "files": written["files"],
            "run_dir": str(canonical_dir),
            "ply_path": str(ply_path),
            "components": {
                "count": int(stats["components_count"]),
                "largest_fraction": float(stats["largest_component_fraction"]),
            },
            "degenerate": {
                "count": int(stats["degenerate_removed_count"]),
                "fraction": float(stats["degenerate_fraction"]),
            },
            "decimation": decimation,
            "viewer": {
                "enabled": True,
                "path": viewer_path.name,
                "absolute_path": str(viewer_path),
            },
            "inputs": inputs,
            "dense_backend": dense_backend,
            "tool_versions": tool_versions,
            "git_sha": _git_sha(),
            "g3": {"pass": False, "checks": {}},
        }
    )
    manifest_path = manifest.write_manifest(canonical_dir, built)

    try:
        validation = validate.validate_run(canonical_dir, verify_hashes=True)
    except validate.MeshValidationError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise IngestError(f"canonical mesh failed the C++ gate: {detail}") from exc

    sha256sums_path = canonical_dir / "SHA256SUMS"
    report_path = canonical_dir / "ingest_report.json"
    _write_json(
        report_path,
        {
            "schema_version": 1,
            "route": route,
            "window": window_name,
            "frames": [int(frame) for frame in frames],
            "source_mesh": {
                "path": str(mesh_path),
                "sha256": source_sha256,
                **source_stats,
            },
            "envelope": envelope,
            "cleanup": stats,
            "canonical": {
                "path": str(ply_path),
                "format": "ply",
                "vertex_count": int(built["vertex_count"]),
                "triangle_count": int(built["triangle_count"]),
                "bbox_min_m": list(built["bbox_min_m"]),
                "bbox_max_m": list(built["bbox_max_m"]),
                "files": {
                    fmt: {
                        "path": entry["path"],
                        "sha256": entry["sha256"],
                        "bytes": int(entry["bytes"]),
                        "vertex_count": int(entry["vertex_count"]),
                        "triangle_count": int(entry["triangle_count"]),
                    }
                    for fmt, entry in written["files"].items()
                },
            },
            "viewer": {
                "path": str(viewer_path),
                "sha256": viewer_sha256,
                **decimation["viewer"],
            },
            "decimation": decimation,
            "manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            },
            "sha256sums": {
                "path": str(sha256sums_path),
                "sha256": _sha256_file(sha256sums_path),
            },
            "validation": validation,
            "inputs": inputs,
            "dense_backend": dense_backend,
            "tool_versions": tool_versions,
            "git_sha": built["git_sha"],
            "wall_time_s": round(time.time() - started, 3),
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
            ),
        },
    )

    artifact = {
        "path": str(ply_path),
        "fmt": "ply",
        "vertex_count": int(built["vertex_count"]),
        "triangle_count": int(built["triangle_count"]),
        "bbox_min_m": [float(value) for value in built["bbox_min_m"]],
        "bbox_max_m": [float(value) for value in built["bbox_max_m"]],
        "sha256": built["files"]["ply"]["sha256"],
        "route": route,
        "config_sha256": config_sha,
        "components": dict(built["components"]),
        "degenerate": dict(built["degenerate"]),
        "frame": "W",
        "units": "m",
        "viewer_glb": str(viewer_path),
    }
    return {
        "artifact": artifact,
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }


__all__ = [
    "CAMERA_ENVELOPE_MARGIN_M",
    "DEFAULT_VIEWER_MAX_TRIANGLES",
    "IngestError",
    "MESH_EXTENT_LIMIT_M",
    "ingest",
]
