"""A/B reconstruction route: Open3D alpha shape on the shared dense cloud.

This route is a *mesher* comparison, not an independent MVS pipeline (D-04).
It consumes the dense point cloud produced for the same run (COLMAP CUDA
``fused.ply``, or the explicitly approved ``openmvs_densify`` fallback) and
builds a triangle mesh with Open3D's alpha shape:

1. read the cloud and reject non-finite coordinates,
2. crop to the camera-centre observation envelope (never to GT),
3. voxel-downsample (``voxel_smoke`` / ``voxel_full``),
4. estimate normals and orient them consistently,
5. ``create_from_point_cloud_alpha_shape`` with ``alpha = alpha_factor * voxel``,
6. keep the largest connected component and write the raw PLY.

Poisson and TSDF are intentionally absent (D-05).  Evaluation data never enters
construction: the crop uses camera centres only and the report always states
``gt_used_for_construction: false``.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import config_sha256
from ..contracts import RawRouteOutput, RouteDescriptor, WindowSpec
from ..meshio import mesh_stats, write_ply

MESH_EXTENT_LIMIT_M = 500.0
MIN_INPUT_POINTS = 10_000


class RouteError(RuntimeError):
    """The route cannot run, or its output violates a metre/W invariant."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def camera_envelope(
    camera_centers: Sequence[Sequence[float]],
    *,
    xy_m: float = 60.0,
    z_range: Sequence[float] = (-10.0, 30.0),
) -> tuple[np.ndarray, np.ndarray]:
    """AABB of the camera centres expanded in x/y and offset in z.

    Offsets are relative to the camera-centre AABB: ``x/y +=-xy_m`` and
    ``z += [z_range[0], z_range[1]]``.  The envelope is construction-only and is
    derived from camera poses, never from GT or LiDAR.
    """
    centers = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    if centers.shape[0] == 0:
        raise RouteError("camera_envelope requires at least one camera centre")
    if not np.all(np.isfinite(centers)):
        raise RouteError("camera centres contain non-finite coordinates")
    low = centers.min(axis=0)
    high = centers.max(axis=0)
    low[0] -= float(xy_m)
    high[0] += float(xy_m)
    low[1] -= float(xy_m)
    high[1] += float(xy_m)
    low[2] += float(z_range[0])
    high[2] += float(z_range[1])
    return low, high


def crop_mask(
    points: np.ndarray, low: Sequence[float], high: Sequence[float]
) -> np.ndarray:
    """Boolean mask of points inside the inclusive envelope."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RouteError(f"points must have shape (N, 3), got {points.shape}")
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    inside = np.all(points >= low, axis=1) & np.all(points <= high, axis=1)
    return inside


def largest_component(mesh: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Keep the largest connected triangle component and drop unused vertices."""
    import open3d as o3d

    clusters, counts, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(clusters)
    cluster_counts = np.asarray(counts)
    if cluster_counts.size == 0 or int(cluster_counts.max()) == 0:
        raise RouteError("alpha shape produced no connected component")

    best = int(cluster_counts.argmax())
    triangles = np.asarray(mesh.triangles)[cluster_ids == best]
    kept = o3d.geometry.TriangleMesh()
    kept.vertices = mesh.vertices
    kept.triangles = o3d.utility.Vector3iVector(triangles)
    kept.remove_unreferenced_vertices()
    vertices = np.asarray(kept.vertices, dtype=np.float64)
    faces = np.asarray(kept.triangles, dtype=np.int64)
    return vertices, faces, {
        "component_count": int(cluster_counts.size),
        "largest_triangle_count": int(cluster_counts.max()),
        "total_triangle_count": int(cluster_counts.sum()),
        "largest_fraction": float(cluster_counts.max() / cluster_counts.sum()),
    }


class Open3DAlpha:
    """Open3D alpha-shape mesher over the shared dense cloud."""

    name = "open3d_alpha"

    def __init__(
        self,
        *,
        cfg: Mapping,
        env: Mapping,
        bundle_root: str | Path | None = None,
        window_name: str | None = None,
        frames: Sequence[int] | None = None,
        colmap_result: Mapping | None = None,
        dense_result: Mapping | None = None,
        camera_centers: Sequence[Sequence[float]] | None = None,
        **_ignored: Any,
    ) -> None:
        self.cfg = dict(cfg or {})
        self.env = dict(env or {})
        self.bundle_root = Path(bundle_root) if bundle_root else None
        self.window_name = window_name
        self.frames = tuple(int(frame) for frame in (frames or ()))
        self.colmap_result = dict(colmap_result) if colmap_result else None
        self.dense_result = dict(dense_result) if dense_result else None
        self._explicit_centers = (
            np.asarray(camera_centers, dtype=np.float64)
            if camera_centers is not None
            else None
        )
        self.report: dict[str, Any] | None = None
        self.report_path: Path | None = None

    def describe(self) -> RouteDescriptor:
        packages = self.env.get("packages") or {}
        return RouteDescriptor(
            route=self.name,
            config_sha256=config_sha256(self.cfg),
            tool_versions={"open3d": str(packages.get("open3d") or "")},
            dense_backend=str(self.env.get("dense_backend") or ""),
        )

    def _camera_centers(self, window: WindowSpec) -> np.ndarray:
        if self._explicit_centers is not None:
            return self._explicit_centers
        if self.bundle_root is None:
            raise RouteError(
                "open3d_alpha needs bundle_root or explicit camera centres"
            )
        from ..cameras import T_W_C, load_bundle

        bundle = load_bundle(self.bundle_root)
        return np.asarray(
            [T_W_C(bundle, int(frame))[:3, 3] for frame in window.frames],
            dtype=np.float64,
        )

    def _dense_cloud_source(self, dense_cloud: Path) -> str:
        if self.dense_result:
            explicit = self.dense_result.get("dense_cloud_source")
            if explicit:
                return str(explicit)
            backend = self.dense_result.get("dense_backend")
            if backend == "openmvs_densify":
                return "openmvs_densify"
            if backend:
                return str(backend)
        return "openmvs_densify" if dense_cloud.name == "scene_dense.ply" else "unknown"

    def build(
        self,
        window: WindowSpec,
        workdir: str | Path,
        dense_cloud: Path | None = None,
    ) -> RawRouteOutput:
        """Mesh ``dense_cloud`` with an alpha shape in the W frame."""
        started = time.time()
        if dense_cloud is None:
            raise RouteError(
                "open3d_alpha requires a dense cloud; run the dense stage or "
                "resolve the dense-backend checkpoint"
            )
        dense_cloud = Path(dense_cloud)
        if not dense_cloud.is_file():
            raise RouteError(
                f"dense cloud is missing: {dense_cloud}; run the dense stage or "
                "resolve the dense-backend checkpoint"
            )

        import open3d as o3d

        params = (self.cfg.get("params") or {}).get("open3d") or {}
        voxel = float(
            params.get("voxel_smoke", 0.10)
            if window.name == "smoke"
            else params.get("voxel_full", 0.05)
        )
        alpha_factor = float(params.get("alpha_factor", 4.0))
        normal_k = int(params.get("normal_k", 30))
        xy_m = float(params.get("envelope_xy_m", 60.0))
        z_range = tuple(params.get("envelope_z_m", (-10.0, 30.0)))
        if len(z_range) != 2:
            raise RouteError("params.open3d.envelope_z_m must have two values")

        workdir = Path(workdir)
        out_dir = workdir / "open3d"
        out_dir.mkdir(parents=True, exist_ok=True)

        point_cloud = o3d.io.read_point_cloud(str(dense_cloud))
        points = np.asarray(point_cloud.points, dtype=np.float64)
        if points.shape[0] <= MIN_INPUT_POINTS:
            raise RouteError(
                f"dense cloud has {points.shape[0]} points; at least "
                f"{MIN_INPUT_POINTS + 1} are required"
            )
        if not np.all(np.isfinite(points)):
            raise RouteError("dense cloud contains non-finite coordinates")

        centers = self._camera_centers(window)
        low, high = camera_envelope(centers, xy_m=xy_m, z_range=z_range)
        keep = crop_mask(points, low, high)
        cropped = np.ascontiguousarray(points[keep])
        if cropped.shape[0] < 32:
            raise RouteError(
                f"only {cropped.shape[0]} dense points lie inside the camera "
                f"envelope {low.tolist()}..{high.tolist()}"
            )

        cropped_cloud = o3d.geometry.PointCloud()
        cropped_cloud.points = o3d.utility.Vector3dVector(cropped)
        if voxel <= 0.0:
            raise RouteError(f"voxel size must be positive, got {voxel}")
        downsampled = cropped_cloud.voxel_down_sample(voxel)
        down_points = np.asarray(downsampled.points, dtype=np.float64)
        if down_points.shape[0] < 32:
            raise RouteError(
                f"voxel {voxel:g} m left only {down_points.shape[0]} points"
            )

        downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
        )
        downsampled.orient_normals_consistent_tangent_plane(normal_k)

        alpha = alpha_factor * voxel
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            downsampled, alpha
        )
        vertices, faces, component = largest_component(mesh)
        stats = mesh_stats(vertices, faces)
        if stats["triangle_count"] <= 0:
            raise RouteError("alpha shape produced no triangles")
        extent = np.asarray(stats["bbox_max_m"]) - np.asarray(stats["bbox_min_m"])
        if np.any(extent >= MESH_EXTENT_LIMIT_M):
            raise RouteError(
                f"alpha mesh extent {extent.tolist()} m exceeds the "
                f"{MESH_EXTENT_LIMIT_M:g} m per-axis limit"
            )

        raw_path = out_dir / "open3d_alpha_raw.ply"
        write_ply(raw_path, vertices, faces)
        raw_sha256 = _sha256_file(raw_path)
        wall_time = time.time() - started

        source = self._dense_cloud_source(dense_cloud)
        report = {
            "route": self.name,
            "window": window.name,
            "frames": [int(frame) for frame in window.frames],
            "dense_cloud_path": str(dense_cloud),
            "dense_cloud_sha256": _sha256_file(dense_cloud),
            "dense_cloud_source": source,
            "dense_backend": str(self.env.get("dense_backend") or ""),
            "params": {
                "voxel_size": voxel,
                "voxel_smoke": float(params.get("voxel_smoke", 0.10)),
                "voxel_full": float(params.get("voxel_full", 0.05)),
                "alpha": alpha,
                "alpha_factor": alpha_factor,
                "normal_k": normal_k,
                "normals_mode": "consistent_tangent_plane",
                "envelope_xy_m": xy_m,
                "envelope_z_m": [float(value) for value in z_range],
            },
            "point_counts": {
                "input": int(points.shape[0]),
                "after_envelope": int(cropped.shape[0]),
                "after_voxel": int(down_points.shape[0]),
            },
            "envelope": {
                "min_m": [float(value) for value in low],
                "max_m": [float(value) for value in high],
                "camera_center_count": int(centers.shape[0]),
            },
            "largest_component": component,
            "mesh": {
                "path": str(raw_path),
                "sha256": raw_sha256,
                "bytes": int(raw_path.stat().st_size),
                **stats,
            },
            "wall_time_s": round(wall_time, 3),
            "peak_rss_mb": _peak_rss_mb(),
            "rss_basis": "ru_maxrss of this process",
            "gt_used_for_construction": False,
            "descriptor": {
                "route": self.name,
                "config_sha256": config_sha256(self.cfg),
                "tool_versions": self.describe().tool_versions,
                "dense_backend": str(self.env.get("dense_backend") or ""),
            },
        }
        report_path = out_dir / "open3d_report.json"
        report_path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        log_path = out_dir / "open3d.log"
        log_path.write_text(
            "\n".join(
                [
                    f"route: {self.name}",
                    f"window: {window.name} ({window.frames[0]}..{window.frames[-1]})",
                    f"dense cloud: {dense_cloud} ({points.shape[0]} points)",
                    f"dense cloud source: {source}",
                    f"envelope: {low.tolist()} .. {high.tolist()}",
                    f"points after envelope: {cropped.shape[0]}",
                    f"voxel: {voxel:g} m -> {down_points.shape[0]} points",
                    f"alpha: {alpha:g} m",
                    f"triangles: {stats['triangle_count']}",
                    f"bbox: {stats['bbox_min_m']} .. {stats['bbox_max_m']}",
                    f"wall time: {wall_time:.3f} s",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        self.report = report
        self.report_path = report_path
        return RawRouteOutput(
            mesh_path=raw_path,
            dense_cloud_path=dense_cloud,
            logs_path=log_path,
        )


__all__ = [
    "MIN_INPUT_POINTS",
    "Open3DAlpha",
    "RouteError",
    "camera_envelope",
    "crop_mask",
    "largest_component",
]
