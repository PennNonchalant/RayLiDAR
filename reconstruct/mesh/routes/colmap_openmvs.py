"""Canonical reconstruction route: known-pose COLMAP model -> OpenMVS.

The model stage (``colmap_model``) already produced a metric, pose-frozen
COLMAP text model.  This route imports it with ``InterfaceCOLMAP`` and runs the
verified OpenMVS chain:

``InterfaceCOLMAP -> DensifyPointCloud -> ReconstructMesh -> RefineMesh``

Every command is a fixed argument list executed with ``subprocess.run`` without
a shell, using absolute binary paths recorded in ``env_report.json``.
Each command's argv, exit code, output tail, wall time, and running peak RSS are
written to ``openmvs_report.json``; the output mesh is asserted to be finite,
non-empty, and in W-frame metres before it is returned.

The dense point cloud used by the mesher is produced by OpenMVS's own
``DensifyPointCloud`` (this is the canonical route).  No mask flags are passed in
this phase: the smoke window is zero-dynamic and ``-m`` means different things
in ``DensifyPointCloud`` and ``RefineMesh`` (``--mask-path`` vs ``--mesh-file``).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shlex
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import config_sha256
from ..contracts import RawRouteOutput, RouteDescriptor, WindowSpec
from ..meshio import mesh_stats

DEFAULT_OPENMVS_ROOT = Path(
    os.environ.get(
        "RAYLIDAR_OPENMVS_ROOT", "/data-20T/hybrid-lidar-sim/tools/openmvs-2.4.0"
    )
)
REQUIRED_BINARIES: tuple[str, ...] = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
)

# A metric street mesh is tens of metres; 500 m per axis means a unit/scale bug.
MESH_EXTENT_LIMIT_M = 500.0
# The mesh must overlap the observed corridor, not drift elsewhere in W.
CAMERA_ENVELOPE_MARGIN_M = 100.0

DEFAULT_TIMEOUTS_S: dict[str, float] = {
    "InterfaceCOLMAP": 600.0,
    "DensifyPointCloud": 7200.0,
    "ReconstructMesh": 3600.0,
    "RefineMesh": 3600.0,
}

EXPECTED_OUTPUTS: dict[str, str] = {
    "scene": "scene.mvs",
    "dense_cloud": "scene_dense.ply",
    "dense_scene": "scene_dense.mvs",
    "mesh": "scene_mesh.ply",
    "refined": "scene_mesh_refined.ply",
}


class RouteError(RuntimeError):
    """The route cannot run, or its output violates a metre/W invariant."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 1)


def openmvs_binaries(env: Mapping) -> dict[str, Path]:
    """Resolve the four OpenMVS executables from the environment report."""
    entry = (env.get("openmvs") or {}) if isinstance(env, Mapping) else {}
    binaries = entry.get("binaries") or {}
    root = Path(entry.get("root") or DEFAULT_OPENMVS_ROOT)
    resolved: dict[str, Path] = {}
    for name in REQUIRED_BINARIES:
        configured = (binaries.get(name) or {}).get("path")
        candidate = Path(configured) if configured else root / name
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RouteError(f"OpenMVS binary is not executable: {candidate}")
        resolved[name] = candidate
    return resolved


def plan_commands(
    *,
    binaries: Mapping[str, Path],
    workspace: Path,
    images_dir: Path,
    out_dir: Path,
    threads: int,
    params: Mapping | None = None,
) -> list[dict[str, Any]]:
    """Return the four verified OpenMVS commands in chain order.

    ``-o`` is always explicit so the output names never depend on a binary's
    default naming.  ``--estimate-roi 0 --crop-to-roi 0`` keep the whole observed
    corridor: the default ROI estimate on a sparse known-pose cloud crops valid
    geometry away (measured on the smoke window).
    """
    openmvs = (params or {}).get("openmvs") or {}
    out_dir = Path(out_dir)
    scene = out_dir / EXPECTED_OUTPUTS["scene"]
    dense_scene = out_dir / EXPECTED_OUTPUTS["dense_scene"]
    dense_cloud = out_dir / EXPECTED_OUTPUTS["dense_cloud"]
    mesh = out_dir / EXPECTED_OUTPUTS["mesh"]
    refined = out_dir / EXPECTED_OUTPUTS["refined"]
    common = ["--max-threads", str(int(threads))]

    return [
        {
            "name": "InterfaceCOLMAP",
            "argv": [
                str(binaries["InterfaceCOLMAP"]),
                "-i",
                str(workspace),
                "-o",
                str(scene),
                "--image-folder",
                str(images_dir),
                *common,
            ],
            "timeout_s": DEFAULT_TIMEOUTS_S["InterfaceCOLMAP"],
        },
        {
            "name": "DensifyPointCloud",
            "argv": [
                str(binaries["DensifyPointCloud"]),
                str(scene),
                "-o",
                str(dense_cloud),
                "--estimate-roi",
                "0",
                "--crop-to-roi",
                "0",
                *common,
            ],
            "timeout_s": DEFAULT_TIMEOUTS_S["DensifyPointCloud"],
        },
        {
            "name": "ReconstructMesh",
            "argv": [
                str(binaries["ReconstructMesh"]),
                str(dense_scene),
                "-p",
                str(dense_cloud),
                "-o",
                str(mesh),
                "--remove-spurious",
                f"{int(openmvs.get('remove_spurious', 4))}",
                "--close-holes",
                f"{int(openmvs.get('close_holes', 30))}",
                "--smooth",
                f"{int(openmvs.get('smooth', 2))}",
                *common,
            ],
            "timeout_s": DEFAULT_TIMEOUTS_S["ReconstructMesh"],
        },
        {
            "name": "RefineMesh",
            "argv": [
                str(binaries["RefineMesh"]),
                str(dense_scene),
                "-m",
                str(mesh),
                "-o",
                str(refined),
                "--scales",
                f"{int(openmvs.get('refine_scales', 1))}",
                "--max-face-area",
                f"{float(openmvs.get('max_face_area', 16)):g}",
                *common,
            ],
            "timeout_s": DEFAULT_TIMEOUTS_S["RefineMesh"],
        },
    ]


def resolve_outputs(
    out_dir: str | Path, expected: Mapping[str, str] | None = None
) -> dict[str, Path]:
    """Resolve expected output names, listing the directory on any miss."""
    out_dir = Path(out_dir)
    expected = dict(expected or EXPECTED_OUTPUTS)
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for key, name in expected.items():
        candidate = out_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            resolved[key] = candidate
        else:
            missing.append(name)
    if missing:
        present = sorted(path.name for path in out_dir.iterdir()) if out_dir.is_dir() else []
        raise RouteError(
            f"OpenMVS did not produce {missing} in {out_dir}; present files: {present}"
        )
    return resolved


def validate_mesh_envelope(
    stats: Mapping[str, Any],
    camera_centers: Sequence[Sequence[float]],
    *,
    max_extent_m: float = MESH_EXTENT_LIMIT_M,
    margin_m: float = CAMERA_ENVELOPE_MARGIN_M,
) -> None:
    """Assert the mesh is metric-scale and overlaps the camera corridor.

    Raises :class:`RouteError` with the measured bbox; it never rescales or
    translates the mesh to fit.
    """
    bbox_min = np.asarray(stats["bbox_min_m"], dtype=np.float64)
    bbox_max = np.asarray(stats["bbox_max_m"], dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise RouteError(f"invalid bbox: min={bbox_min} max={bbox_max}")
    if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)):
        raise RouteError(f"mesh bbox is not finite: min={bbox_min} max={bbox_max}")

    extent = bbox_max - bbox_min
    if np.any(extent >= max_extent_m):
        raise RouteError(
            f"mesh extent {extent.tolist()} m exceeds the {max_extent_m:g} m "
            f"per-axis limit (bbox {bbox_min.tolist()}..{bbox_max.tolist()}); "
            "this indicates a unit or frame error"
        )

    centers = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    if centers.shape[0] == 0:
        raise RouteError("cannot validate the mesh without camera centres")
    envelope_min = centers.min(axis=0) - float(margin_m)
    envelope_max = centers.max(axis=0) + float(margin_m)
    disjoint = np.any(bbox_max < envelope_min) or np.any(bbox_min > envelope_max)
    if disjoint:
        raise RouteError(
            f"mesh bbox {bbox_min.tolist()}..{bbox_max.tolist()} is disjoint from "
            f"the camera envelope {envelope_min.tolist()}..{envelope_max.tolist()} "
            f"(margin {margin_m:g} m); this indicates a frame or scale error"
        )


def camera_centers_from_model(
    model_dir: str | Path, window: WindowSpec
) -> list[np.ndarray]:
    """Camera centres for ``window`` from the triangulated COLMAP text model."""
    from ..colmap_model import camera_centers, parse_images_txt

    model_dir = Path(model_dir)
    poses = parse_images_txt(model_dir / "images.txt")
    centers = camera_centers(poses)
    ordered: list[np.ndarray] = []
    for frame in window.frames:
        name = f"{int(frame):010d}.png"
        if name not in centers:
            raise RouteError(f"model {model_dir} has no pose for {name}")
        ordered.append(centers[name])
    return ordered


def read_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh into ``(vertices, faces)`` with lazy Open3D import."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise RouteError(f"mesh {path} has no vertices or triangles")
    return vertices, faces


def _ensure_link(target: Path, source: Path) -> None:
    """Point ``target`` at ``source``; never overwrite a real directory."""
    source = Path(source).resolve()
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if target.is_dir():
            raise RouteError(f"refusing to replace real directory {target}")
        target.unlink()
    target.symlink_to(source)


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _run_command(
    command: Mapping[str, Any],
    *,
    cwd: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Run one OpenMVS command with a list argv and record everything."""
    import subprocess

    argv = [str(part) for part in command["argv"]]
    started = time.time()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=float(command["timeout_s"]),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RouteError(f"failed to run {command['name']}: {exc}") from exc
    wall_time = time.time() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + shlex.join(argv) + "\n")
        log.write(output)
        if output and not output.endswith("\n"):
            log.write("\n")
    record = {
        "name": command["name"],
        "argv": argv,
        "exit_code": int(completed.returncode),
        "wall_time_s": round(wall_time, 3),
        "peak_rss_mb": _peak_rss_mb(),
        "stdout_tail": _tail(completed.stdout or "", 10),
        "stderr_tail": _tail(completed.stderr or "", 10),
    }
    if completed.returncode != 0:
        raise RouteError(
            f"{command['name']} exited {completed.returncode}:\n{_tail(output)}"
        )
    return record


class ColmapOpenMVS:
    """Canonical route: COLMAP known-pose model -> OpenMVS densify/mesh/refine."""

    name = "colmap_openmvs"

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
        **_ignored: Any,
    ) -> None:
        self.cfg = dict(cfg or {})
        self.env = dict(env or {})
        self.bundle_root = Path(bundle_root) if bundle_root else None
        self.window_name = window_name
        self.frames = tuple(int(frame) for frame in (frames or ()))
        self.colmap_result = dict(colmap_result) if colmap_result else None
        self.dense_result = dict(dense_result) if dense_result else None
        self.report: dict[str, Any] | None = None
        self.report_path: Path | None = None

    def describe(self) -> RouteDescriptor:
        openmvs = self.env.get("openmvs") or {}
        colmap = self.env.get("colmap") or {}
        return RouteDescriptor(
            route=self.name,
            config_sha256=config_sha256(self.cfg),
            tool_versions={
                "colmap": str(colmap.get("version") or ""),
                "openmvs": str(openmvs.get("version") or ""),
            },
            dense_backend=str(self.env.get("dense_backend") or ""),
        )

    def _workspace_paths(self, workdir: Path, window: WindowSpec) -> tuple[Path, Path, Path]:
        if not self.colmap_result:
            raise RouteError(
                "colmap_openmvs requires the model stage's colmap_result"
            )
        model_dir = Path(self.colmap_result["sparse_tri_txt_dir"])
        images_dir = Path(self.colmap_result["images_dir"])
        for name in ("cameras.txt", "images.txt", "points3D.txt"):
            if not (model_dir / name).is_file():
                raise RouteError(f"triangulated model is missing {name}: {model_dir / name}")
        for frame in window.frames:
            image = images_dir / f"{int(frame):010d}.png"
            if not image.is_file():
                raise RouteError(f"window image is missing: {image}")

        out_dir = Path(workdir) / "openmvs"
        out_dir.mkdir(parents=True, exist_ok=True)
        workspace = out_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        _ensure_link(workspace / "sparse", model_dir)
        _ensure_link(workspace / "images", images_dir)
        return out_dir, workspace, images_dir

    def build(
        self,
        window: WindowSpec,
        workdir: str | Path,
        dense_cloud: Path | None = None,
    ) -> RawRouteOutput:
        """Run the four-command OpenMVS chain and return the refined mesh."""
        started = time.time()
        workdir = Path(workdir)
        out_dir, workspace, images_dir = self._workspace_paths(workdir, window)

        for name in EXPECTED_OUTPUTS.values():
            candidate = out_dir / name
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()

        params = self.cfg.get("params") or {}
        threads = int(params.get("threads", 8))
        binaries = openmvs_binaries(self.env)
        commands = plan_commands(
            binaries=binaries,
            workspace=workspace,
            images_dir=images_dir,
            out_dir=out_dir,
            threads=threads,
            params=params,
        )

        log_path = out_dir / "openmvs.log"
        log_path.write_text("", encoding="utf-8")
        records: list[dict[str, Any]] = []
        for command in commands:
            records.append(_run_command(command, cwd=out_dir, log_path=log_path))

        outputs = resolve_outputs(out_dir)
        vertices, faces = read_mesh(outputs["refined"])
        stats = mesh_stats(vertices, faces)
        centers = camera_centers_from_model(
            Path(self.colmap_result["sparse_tri_txt_dir"]), window
        )
        validate_mesh_envelope(stats, centers)

        described_outputs: dict[str, dict[str, Any]] = {}
        for key, path in outputs.items():
            entry: dict[str, Any] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            if key in ("mesh", "refined"):
                mesh_vertices, mesh_faces = read_mesh(path)
                entry.update(mesh_stats(mesh_vertices, mesh_faces))
            described_outputs[key] = entry

        report = {
            "route": self.name,
            "window": window.name,
            "frames": [int(frame) for frame in window.frames],
            "argv": records,
            "outputs": described_outputs,
            "mesh": {
                "path": str(outputs["refined"]),
                "sha256": _sha256_file(outputs["refined"]),
                "bytes": int(outputs["refined"].stat().st_size),
                **stats,
            },
            "dense_cloud_path": str(outputs["dense_cloud"]),
            "wall_time_s": round(time.time() - started, 3),
            "peak_rss_mb": _peak_rss_mb(),
            "rss_basis": "ru_maxrss of children (running maximum)",
            "threads": threads,
            "params": {
                "remove_spurious": int(
                    (params.get("openmvs") or {}).get("remove_spurious", 4)
                ),
                "close_holes": int((params.get("openmvs") or {}).get("close_holes", 30)),
                "smooth": int((params.get("openmvs") or {}).get("smooth", 2)),
                "refine_scales": int(
                    (params.get("openmvs") or {}).get("refine_scales", 1)
                ),
                "max_face_area": float(
                    (params.get("openmvs") or {}).get("max_face_area", 16)
                ),
                "estimate_roi": False,
                "crop_to_roi": False,
            },
            "versions": self.describe().tool_versions,
            "masks_used": False,
            "descriptor": {
                "route": self.name,
                "config_sha256": config_sha256(self.cfg),
                "tool_versions": self.describe().tool_versions,
                "dense_backend": str(self.env.get("dense_backend") or ""),
            },
        }
        report_path = out_dir / "openmvs_report.json"
        report_path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        self.report = report
        self.report_path = report_path

        return RawRouteOutput(
            mesh_path=outputs["refined"],
            dense_cloud_path=outputs["dense_cloud"],
            logs_path=log_path,
        )


__all__ = [
    "CAMERA_ENVELOPE_MARGIN_M",
    "ColmapOpenMVS",
    "EXPECTED_OUTPUTS",
    "MESH_EXTENT_LIMIT_M",
    "REQUIRED_BINARIES",
    "RouteError",
    "camera_centers_from_model",
    "openmvs_binaries",
    "plan_commands",
    "read_mesh",
    "resolve_outputs",
    "validate_mesh_envelope",
]
