"""Environment probe for the Phase-4 reconstruction runtime.

The module only *inspects* the runtime; it never installs anything.  Tool
provisioning happens through explicit container shell steps and is recorded in
``env_report.json`` (versions, banners, hashes, GPU, dense-backend verdict).

The report is the single machine-readable answer to "what exactly produced this
mesh?", so every value is captured from the live binary rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_COLMAP = Path(
    os.environ.get("RAYLIDAR_COLMAP", "/data-20T/conda_envs/colmap/bin/colmap")
)
DEFAULT_COLMAP_CUDA = Path(
    os.environ.get(
        "RAYLIDAR_COLMAP_CUDA", "/data-20T/conda_envs/colmap-cuda/bin/colmap"
    )
)
DEFAULT_OPENMVS_ROOT = Path(
    os.environ.get(
        "RAYLIDAR_OPENMVS_ROOT", "/data-20T/hybrid-lidar-sim/tools/openmvs-2.4.0"
    )
)
DEFAULT_OUT = Path("/data-20T/hybrid-lidar-sim/artifacts/04/env_report.json")

OPENMVS_BINARIES = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
)

_VERSION_RE = re.compile(r"COLMAP\s+([0-9]+(?:\.[0-9]+)*)")


def _run(binary: Path | str, arguments: list[str], timeout: float = 120.0):
    """Run a tool and return ``(returncode, combined output)``.

    Missing or hung binaries return ``(None, "")`` instead of raising: a probe
    must describe a broken environment, not crash on it.
    """
    try:
        completed = subprocess.run(
            [str(binary), *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return None, ""
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


def banner_has_cuda(banner: str) -> bool:
    """True only for a build banner that positively states CUDA support.

    ``without CUDA`` must not match; the check is on the literal phrase
    ``with CUDA`` (the COLMAP banner prints ``without CUDA`` for CPU builds).
    """
    return "with CUDA" in banner


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _colmap_info(path: Path) -> dict:
    if not path.is_file() or not os.access(path, os.X_OK):
        return {"path": str(path), "banner": "", "has_cuda": False, "version": ""}
    _, banner = _run(path, ["-h"])
    match = _VERSION_RE.search(banner)
    has_cuda = banner_has_cuda(banner) and _run(path, ["patch_match_stereo", "-h"])[0] == 0
    return {
        "path": str(path),
        "banner": banner.strip(),
        "has_cuda": has_cuda,
        "version": match.group(1) if match else "",
    }


def _colmap_cuda_info(path: Path) -> dict:
    info = _colmap_info(path)
    return {
        "path": info["path"] if info["banner"] else None,
        "has_cuda": info["has_cuda"],
        "build": info["version"] or None,
    }


def _openmvs_info(root: Path) -> dict:
    metadata_path = root / ".bundle.json"
    metadata: dict = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    binaries: dict[str, dict] = {}
    for name in OPENMVS_BINARIES:
        binary = root / name
        if binary.is_file():
            binaries[name] = {
                "path": str(binary),
                "sha256": _sha256_file(binary),
            }
    return {
        "root": str(root),
        "version": str(metadata.get("version", "2.4.0")),
        "zip_sha256": str(metadata.get("zip_sha256", "")),
        "binaries": binaries,
    }


def _gpu_name() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    returncode, output = _run(
        executable, ["--query-gpu=name", "--format=csv,noheader"], timeout=30.0
    )
    if returncode != 0:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else None


def _bundle_info(bundle_root: Path) -> dict:
    images = bundle_root / "raw" / "images" / "image_00"
    pngs = sorted(images.glob("*.png")) if images.is_dir() else []
    frames: list[int] = []
    for path in pngs:
        try:
            frames.append(int(path.stem))
        except ValueError:
            continue
    return {
        "root": str(bundle_root),
        "image_00_count": len(pngs),
        "first_frame": min(frames) if frames else 0,
        "last_frame": max(frames) if frames else 0,
    }


def probe_environment(
    bundle_root: Path,
    colmap_path: Path | None = None,
    colmap_cuda_path: Path | None = None,
    openmvs_root: Path | None = None,
) -> dict:
    """Return the machine-readable environment report (schema_version 1)."""
    bundle_root = Path(bundle_root)
    colmap_path = Path(colmap_path) if colmap_path is not None else DEFAULT_COLMAP
    colmap_cuda_path = (
        Path(colmap_cuda_path) if colmap_cuda_path is not None else DEFAULT_COLMAP_CUDA
    )
    openmvs_root = (
        Path(openmvs_root) if openmvs_root is not None else DEFAULT_OPENMVS_ROOT
    )

    colmap = _colmap_info(colmap_path)
    colmap_cuda = _colmap_cuda_info(colmap_cuda_path)
    dense_backend = (
        "colmap_cuda"
        if colmap_cuda["path"] is not None and colmap_cuda["has_cuda"]
        else "unavailable_cuda"
    )
    logical_cores = os.cpu_count() or 1
    configured_threads = os.environ.get("OMP_NUM_THREADS") or os.environ.get(
        "RAYLIDAR_THREADS"
    )
    try:
        threads = int(configured_threads) if configured_threads else logical_cores
    except ValueError:
        threads = logical_cores

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "packages": {
            "open3d": _package_version("open3d"),
            "numpy": _package_version("numpy"),
            "trimesh": _package_version("trimesh"),
            "pyyaml": _package_version("PyYAML"),
        },
        "colmap": colmap,
        "colmap_cuda": colmap_cuda,
        "openmvs": _openmvs_info(openmvs_root),
        "hardware": {"logical_cores": logical_cores, "gpu": _gpu_name()},
        "dense_backend": dense_backend,
        "threads": threads,
        "bundle": _bundle_info(bundle_root),
    }


def write_env_report(out_path: Path, report: dict) -> str:
    """Write canonical JSON atomically and return the SHA-256 of the bytes."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, out_path)
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.mesh.reconstruct.env",
        description="Probe the Phase-4 reconstruction toolchain",
    )
    parser.add_argument(
        "--bundle-root",
        default="/data-20T/hybrid-lidar-sim/exports/p2-60f-step10-4615-4674",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    report = probe_environment(Path(args.bundle_root))
    digest = write_env_report(Path(args.out), report)
    print(f"env_report: {args.out}")
    print(f"sha256: {digest}")
    print(f"dense_backend: {report['dense_backend']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
