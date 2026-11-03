"""Dense-backend resolution and the COLMAP PatchMatch/stereo-fusion driver.

D-04 locks the Open3D mesher input to a COLMAP dense fused cloud.  The
installed CPU COLMAP has no CUDA PatchMatch, so a *silent* switch to another
producer would change the provenance of the A/B.  Instead:

* the backend is resolved from the recorded environment probe, and
* when no CUDA build exists the stage writes ``DENSE_BACKEND_BLOCKED.json``
  with the three explicit options and exits 3 (never substitutes a backend).

With a CUDA build the stage runs ``image_undistorter → patch_match_stereo →
stereo_fusion`` on the fixed-pose model and records the build, CUDA visibility,
hashes, and peak memory in ``dense_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_COLMAP_CUDA = Path(
    os.environ.get(
        "RAYLIDAR_COLMAP_CUDA", "/data-20T/conda_envs/colmap-cuda/bin/colmap"
    )
)
MIN_FUSED_POINTS = 10_000
FUSED_SAMPLE_SIZE = 10_000
COLMAP_TIMEOUT_S = 7200.0

BLOCKED_OPTIONS = (
    {
        "id": "openmvs_densify",
        "effect": "Open3D mesher consumes the OpenMVS densified cloud; A/B remains "
        "a mesher comparison; provenance deviation recorded",
    },
    {
        "id": "install_cuda_colmap",
        "effect": "retry conda-forge colmap cuda build and keep D-04 exact",
    },
    {
        "id": "drop_open3d",
        "effect": "ship canonical route only; RECON-07 unmet",
    },
)
BLOCKED_REASON = (
    "COLMAP 4.0.4 CPU has no CUDA patch_match_stereo; no recorded backend "
    "produces the D-04 fused cloud"
)

_PLY_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


class DenseBackendUnavailable(RuntimeError):
    """No dense backend satisfies D-04 on this machine."""

    exit_code = 3


def resolve_dense_backend(cfg: Mapping, env: Mapping) -> str:
    """Return the dense backend implied by config + environment probe."""
    params = cfg.get("params") or {}
    backend = params.get("dense_backend", "auto")
    cuda = (env.get("colmap_cuda") or {}) if isinstance(env, Mapping) else {}
    has_cuda = bool(cuda.get("has_cuda")) and bool(cuda.get("path"))

    if backend == "auto":
        if has_cuda:
            return "colmap_cuda"
        raise DenseBackendUnavailable(
            "dense_backend=auto but the environment probe found no CUDA COLMAP"
        )
    if backend == "colmap_cuda":
        if not has_cuda:
            raise DenseBackendUnavailable(
                "dense_backend=colmap_cuda but the CUDA COLMAP is missing"
            )
        return "colmap_cuda"
    raise ValueError(f"unknown dense_backend: {backend!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_banner(binary: Path) -> str:
    """Capture the first line of a COLMAP ``-h`` banner (best effort)."""
    try:
        completed = subprocess.run(
            [str(binary), "-h"],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    output = (completed.stdout or "") + (completed.stderr or "")
    for line in output.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _run_tool(command: Sequence, log_path: Path) -> str:
    parts = [str(part) for part in command]
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(parts) + "\n")
    try:
        completed = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=COLMAP_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"failed to run {parts[0]}: {exc}") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(output)
        if output and not output.endswith("\n"):
            log.write("\n")
    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-20:])
        raise RuntimeError(
            f"{Path(parts[0]).name} {parts[1] if len(parts) > 1 else ''} "
            f"exited {completed.returncode}:\n{tail}"
        )
    return output


# --------------------------------------------------------------------------- #
# PLY inspection
# --------------------------------------------------------------------------- #


def _parse_ply_header(handle) -> tuple[str, int, list[tuple[str, str]], int]:
    """Parse a PLY header; return ``(format, vertex_count, properties, header_len)``."""
    if handle.readline().strip() != b"ply":
        raise RuntimeError("not a PLY file")
    ply_format = ""
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    header_bytes = len(b"ply\n")
    while True:
        line = handle.readline()
        if not line:
            raise RuntimeError("PLY header has no end_header")
        header_bytes += len(line)
        stripped = line.strip()
        if stripped == b"end_header":
            break
        tokens = stripped.split()
        if not tokens or tokens[0] == b"comment":
            continue
        if tokens[0] == b"format":
            ply_format = tokens[1].decode("ascii")
        elif tokens[0] == b"element":
            in_vertex = tokens[1] == b"vertex"
            if in_vertex:
                vertex_count = int(tokens[2])
        elif tokens[0] == b"property" and in_vertex:
            if tokens[1] == b"list":
                raise RuntimeError("PLY vertex list properties are not supported")
            properties.append((tokens[2].decode("ascii"), tokens[1].decode("ascii")))
    return ply_format, vertex_count, properties, header_bytes


def read_ply_points(
    path: str | Path, limit: int = FUSED_SAMPLE_SIZE
) -> tuple[int, np.ndarray]:
    """Return ``(vertex_count, xyz sample)`` from an ascii/binary PLY.

    Only ``x``, ``y``, ``z`` are read; other vertex properties are skipped.  The
    sample has at most ``limit`` rows and keeps the original order.
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"missing PLY: {path}")
    with path.open("rb") as handle:
        ply_format, vertex_count, properties, header_bytes = _parse_ply_header(handle)
        names = [name for name, _ in properties]
        if not all(axis in names for axis in ("x", "y", "z")):
            raise RuntimeError(f"{path}: PLY has no x/y/z vertex properties")
        indices = [names.index(axis) for axis in ("x", "y", "z")]
        take = min(int(limit), vertex_count)
        if ply_format == "ascii":
            points = np.empty((take, 3), dtype=np.float64)
            for row in range(take):
                line = handle.readline()
                if not line:
                    raise RuntimeError(f"{path}: truncated ascii vertex list")
                tokens = line.split()
                points[row] = [float(tokens[index]) for index in indices]
            return vertex_count, points

        if ply_format not in ("binary_little_endian", "binary_big_endian"):
            raise RuntimeError(f"{path}: unsupported PLY format {ply_format!r}")
        endian = "<" if ply_format == "binary_little_endian" else ">"
        format_chars = []
        for _, type_name in properties:
            if type_name not in _PLY_TYPES:
                raise RuntimeError(f"{path}: unsupported PLY type {type_name!r}")
            format_chars.append(_PLY_TYPES[type_name][0])
        packed = struct.Struct(endian + "".join(format_chars))
        points = np.empty((take, 3), dtype=np.float64)
        for row in range(take):
            chunk = handle.read(packed.size)
            if len(chunk) != packed.size:
                raise RuntimeError(f"{path}: truncated binary vertex list")
            values = packed.unpack(chunk)
            points[row] = [values[index] for index in indices]
        return vertex_count, points


def assert_fused_cloud(path: Path, *, minimum: int = MIN_FUSED_POINTS) -> dict:
    """Validate the fused cloud and return its point-count metadata."""
    count, sample = read_ply_points(path, FUSED_SAMPLE_SIZE)
    if count < minimum:
        raise RuntimeError(
            f"fused cloud has {count} points, fewer than the required {minimum}"
        )
    if sample.size and not bool(np.all(np.isfinite(sample))):
        raise RuntimeError("fused cloud sample contains non-finite coordinates")
    return {
        "point_count": count,
        "sampled_points": int(sample.shape[0]),
        "finite_sample": True,
    }


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _blocked_payload(window_name: str, frames: Iterable[int]) -> dict:
    return {
        "reason": BLOCKED_REASON,
        "window": window_name,
        "frames": [int(frame) for frame in frames],
        "options": [dict(option) for option in BLOCKED_OPTIONS],
    }


def produce(
    bundle_root: str | Path,
    window_name: str,
    frames: Iterable[int],
    cfg: dict,
    workdir: str | Path,
    env: Mapping,
    colmap_result: dict | None,
) -> dict:
    """Run the dense stage for the resolved backend."""
    started = time.time()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frames = [int(frame) for frame in frames]

    try:
        backend = resolve_dense_backend(cfg, env)
    except DenseBackendUnavailable as exc:
        blocked_path = workdir / "DENSE_BACKEND_BLOCKED.json"
        blocked_path.write_text(
            json.dumps(_blocked_payload(window_name, frames), sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"dense backend blocked: {blocked_path}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc

    if not isinstance(colmap_result, dict):
        raise ValueError("dense stage requires the model stage's colmap_result")
    images_dir = Path(colmap_result["images_dir"])
    sparse_txt_dir = Path(colmap_result["sparse_tri_txt_dir"])
    if not sparse_txt_dir.is_dir():
        raise ValueError(f"missing triangulated text model: {sparse_txt_dir}")

    params = cfg.get("params") or {}
    colmap_cfg = params.get("colmap") or {}
    threads = int(params.get("threads", 8))
    cuda = (env.get("colmap_cuda") or {}) if isinstance(env, Mapping) else {}
    binary = Path(cuda.get("path") or DEFAULT_COLMAP_CUDA)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"CUDA COLMAP binary is not executable: {binary}")

    dense_dir = workdir / "dense"
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    log_path = dense_dir / "dense_log.txt"
    log_path.write_text("", encoding="utf-8")

    _run_tool(
        [
            binary,
            "image_undistorter",
            "--image_path",
            images_dir,
            "--input_path",
            sparse_txt_dir,
            "--output_path",
            dense_dir,
            "--output_type",
            "COLMAP",
        ],
        log_path,
    )

    _run_tool(
        [
            binary,
            "patch_match_stereo",
            "--workspace_path",
            dense_dir,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "1" if colmap_cfg.get("geom_consistency", True) else "0",
            "--PatchMatchStereo.depth_min",
            f"{float(colmap_cfg.get('depth_min', 1.0)):.6g}",
            "--PatchMatchStereo.depth_max",
            f"{float(colmap_cfg.get('depth_max', 80.0)):.6g}",
        ],
        log_path,
    )

    fused_path = dense_dir / "fused.ply"
    _run_tool(
        [
            binary,
            "stereo_fusion",
            "--workspace_path",
            dense_dir,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            fused_path,
        ],
        log_path,
    )

    cloud = assert_fused_cloud(fused_path)
    report = {
        "window": window_name,
        "frames": frames,
        "dense_backend": backend,
        "fused_ply": str(fused_path),
        "fused_ply_size_bytes": fused_path.stat().st_size,
        "fused_ply_sha256": _sha256_file(fused_path),
        "point_count": cloud["point_count"],
        "sampled_points": cloud["sampled_points"],
        "finite_sample": cloud["finite_sample"],
        "colmap": {
            "path": str(binary),
            "build": cuda.get("build"),
            "banner": cuda.get("banner") or _binary_banner(binary),
            "geom_consistency": bool(colmap_cfg.get("geom_consistency", True)),
            "depth_min": float(colmap_cfg.get("depth_min", 1.0)),
            "depth_max": float(colmap_cfg.get("depth_max", 80.0)),
        },
        "threads": threads,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "wall_time_s": round(time.time() - started, 3),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 1
        ),
        "images_dir": str(images_dir),
        "sparse_tri_txt_dir": str(sparse_txt_dir),
        "dense_dir": str(dense_dir),
        "workdir": str(workdir),
    }
    report_path = workdir / "dense_report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return {
        **report,
        "report_path": str(report_path),
        "dense_result": {
            "dense_backend": backend,
            "fused_ply": str(fused_path),
            "point_count": cloud["point_count"],
            "report_path": str(report_path),
            "report": report,
        },
    }


__all__ = [
    "BLOCKED_OPTIONS",
    "DenseBackendUnavailable",
    "MIN_FUSED_POINTS",
    "assert_fused_cloud",
    "produce",
    "read_ply_points",
    "resolve_dense_backend",
]
