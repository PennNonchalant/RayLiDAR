"""Environment probe and hashable report for the Phase 5 Gaussian runtime.

The report is the provenance record every later Phase 5 artifact points at:
pinned upstream commits and submodules, torch/CUDA/compiler versions, compiled
rasterizer hashes, GPU, and disk state.

Usage:
    python -m reconstruct.gaussians.env \
        --upstream-root /data-20T/hybrid-lidar-sim/reconstruct/gaussians/upstream \
        --artifacts-root /data-20T/hybrid-lidar-sim/artifacts/05 \
        --out /data-20T/hybrid-lidar-sim/artifacts/05/env_report.json

Exit codes: 0 when the verdict is ``ready``, 5 when ``blocked``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

PINNED = {
    "gaussian_splatting": "54c035f7834b564019656c3e3fcc3646292f727d",
    "diff_gaussian_rasterization": "9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0",
    "2d_gaussian_splatting": "f3e3b9fa67bbd1c75e05167ff37391d8dab2a678",
    "diff_surfel_rasterization": "e0ed0207b3e0669960cfad70852200a4a5847f61",
}

# The approved CUDA toolkit used to compile the extensions (plan <interfaces>).
APPROVED_CUDA_HOME = Path("/data-20T/conda_envs/mast3r")
DEFAULT_DATA_ROOT = Path("/data-20T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _git_head(repo_path: Path) -> str:
    if not Path(repo_path).is_dir():
        return ""
    return _run_git(["-C", str(repo_path), "rev-parse", "HEAD"])


def _submodule_head(repo_path: Path, submodule_path: str) -> str:
    if not Path(repo_path).is_dir():
        return ""
    out = _run_git(["-C", str(repo_path), "submodule", "status", "--", submodule_path])
    if not out:
        return ""
    first = out.splitlines()[0]
    token = first.lstrip("+- ").split()
    return token[0] if token else ""


def _repo_report(
    repo_path: Path,
    expected_head: str,
    raster_name: str,
    raster_expected: str,
) -> dict[str, Any]:
    head = _git_head(repo_path)
    raster_head = _submodule_head(repo_path, f"submodules/{raster_name}")
    simple_head = _submodule_head(repo_path, "submodules/simple-knn")
    return {
        "path": str(repo_path),
        "head": head,
        "expected": expected_head,
        "match": head == expected_head,
        "submodules": {
            raster_name: {
                "path": f"submodules/{raster_name}",
                "head": raster_head,
                "expected": raster_expected,
                "match": raster_head == raster_expected,
            },
            "simple_knn": {"head": simple_head},
        },
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
    except Exception:  # pragma: no cover - defensive
        return "missing"


def _torch_versions() -> tuple[str, str]:
    try:
        import torch
    except Exception:
        return "missing", "missing"
    return str(torch.__version__), str(torch.version.cuda or "missing")


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False, "name": None, "capability": None}
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False
    if not available:
        return {"available": False, "name": None, "capability": None}
    try:
        capability = list(torch.cuda.get_device_capability(0))
        name = str(torch.cuda.get_device_name(0))
    except Exception:
        return {"available": False, "name": None, "capability": None}
    return {"available": True, "name": name, "capability": capability}


def _first_line(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        return None
    for line in lines:
        if "release" in line:
            return line
    return lines[0]


def _default_cuda_home() -> str | None:
    env_value = os.environ.get("CUDA_HOME")
    if env_value:
        return env_value
    if (APPROVED_CUDA_HOME / "bin" / "nvcc").is_file():
        return str(APPROVED_CUDA_HOME)
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        return str(CUDA_HOME) if CUDA_HOME else None
    except Exception:
        return None


def _compiler_info(artifacts_root: Path) -> dict[str, Any]:
    gcc12 = shutil.which("gcc-12")
    gcc12_version = _first_line([gcc12, "--version"]) if gcc12 else None
    cuda_home = _default_cuda_home()
    nvcc_version = None
    if cuda_home:
        nvcc = Path(cuda_home) / "bin" / "nvcc"
        if nvcc.is_file():
            nvcc_version = _first_line([str(nvcc), "--version"])

    # A fresh container does not keep gcc-12 from the provisioning job. Fall
    # back to the compiler record the setup script persisted at build time.
    record_path = Path(artifacts_root) / "setup" / "compiler.json"
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        gcc12 = gcc12 or record.get("gcc12_path")
        gcc12_version = gcc12_version or record.get("gcc12_version")
        cuda_home = cuda_home or record.get("cuda_home")
        nvcc_version = nvcc_version or record.get("nvcc_version")

    return {
        "gcc12_path": gcc12,
        "gcc12_version": gcc12_version,
        "cc": os.environ.get("CC") or None,
        "cxx": os.environ.get("CXX") or None,
        "cuda_home": cuda_home,
        "nvcc_version": nvcc_version,
    }


def _extension_report(name: str, compiled_module: str | None = None) -> dict[str, Any]:
    """Import an extension package and hash its compiled shared object.

    The rasterizer packages re-export a ``_C`` extension; ``simple_knn`` is
    itself a namespace package whose compiled module is ``simple_knn._C``.
    """
    target = compiled_module or name
    try:
        importlib.import_module(name)
        module = importlib.import_module(target)
    except Exception:
        return {"importable": False, "file": "", "sha256": ""}
    file_path = getattr(module, "__file__", None) or ""
    digest = ""
    if file_path and Path(file_path).is_file():
        digest = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return {"importable": True, "file": file_path, "sha256": digest}


def _disk_info(artifacts_root: Path) -> dict[str, Any]:
    data_root = Path(os.environ.get("HLS_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    if not data_root.exists():
        data_root = Path(artifacts_root)
    while not data_root.exists() and data_root != data_root.parent:
        data_root = data_root.parent
    usage = shutil.disk_usage(data_root)
    return {
        "data_root": str(data_root),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
    }


def probe_environment(upstream_root: Path, artifacts_root: Path) -> dict[str, Any]:
    """Collect the full environment report. Never raises for missing inputs."""
    upstream_root = Path(upstream_root)
    artifacts_root = Path(artifacts_root)

    torch_version, torch_cuda = _torch_versions()
    gs = _repo_report(
        upstream_root / "gaussian-splatting",
        PINNED["gaussian_splatting"],
        "diff-gaussian-rasterization",
        PINNED["diff_gaussian_rasterization"],
    )
    tdgs = _repo_report(
        upstream_root / "2d-gaussian-splatting",
        PINNED["2d_gaussian_splatting"],
        "diff-surfel-rasterization",
        PINNED["diff_surfel_rasterization"],
    )
    # simple-knn has no __init__.py; hash its compiled _C module, not the
    # (namespace) top-level package.
    extensions = {
        "diff_gaussian_rasterization": _extension_report(
            "diff_gaussian_rasterization", "diff_gaussian_rasterization._C"
        ),
        "diff_surfel_rasterization": _extension_report(
            "diff_surfel_rasterization", "diff_surfel_rasterization._C"
        ),
        "simple_knn": _extension_report("simple_knn._C"),
    }
    gpu = _gpu_info()

    reasons: list[str] = []
    if not gs["match"]:
        reasons.append(
            f"gaussian_splatting HEAD expects {PINNED['gaussian_splatting']}, got {gs['head'] or 'missing'}"
        )
    gs_raster = gs["submodules"]["diff-gaussian-rasterization"]
    if not gs_raster["match"]:
        reasons.append(
            "gaussian_splatting/submodules/diff-gaussian-rasterization expects "
            f"{PINNED['diff_gaussian_rasterization']}, got {gs_raster['head'] or 'missing'}"
        )
    if not tdgs["match"]:
        reasons.append(
            f"2d_gaussian_splatting HEAD expects {PINNED['2d_gaussian_splatting']}, got {tdgs['head'] or 'missing'}"
        )
    tdgs_raster = tdgs["submodules"]["diff-surfel-rasterization"]
    if not tdgs_raster["match"]:
        reasons.append(
            "2d_gaussian_splatting/submodules/diff-surfel-rasterization expects "
            f"{PINNED['diff_surfel_rasterization']}, got {tdgs_raster['head'] or 'missing'}"
        )
    for name, extension in extensions.items():
        if not extension["importable"]:
            reasons.append(f"extension {name} is not importable")
    if not gpu["available"]:
        reasons.append("CUDA GPU is not available")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "packages": {
            "torch": torch_version,
            "torch_cuda": torch_cuda,
            "numpy": _package_version("numpy"),
            "scikit_image": _package_version("scikit-image"),
            "matplotlib": _package_version("matplotlib"),
            "tensorboard": _package_version("tensorboard"),
            "plyfile": _package_version("plyfile"),
            "pytest": _package_version("pytest"),
        },
        "gpu": gpu,
        "compiler": _compiler_info(artifacts_root),
        "upstream": {
            "gaussian_splatting": gs,
            "2d_gaussian_splatting": tdgs,
        },
        "extensions": extensions,
        "disk": _disk_info(artifacts_root),
        "verdict": "blocked",
        "blocked_reasons": reasons,
    }
    report["verdict"] = "blocked" if reasons else "ready"
    return report


def write_env_report(path: Path, report: dict[str, Any]) -> str:
    """Write canonical JSON atomically and return the SHA-256 of the bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    data = payload.encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the Phase 5 Gaussian runtime")
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    report = probe_environment(args.upstream_root, args.artifacts_root)
    digest = write_env_report(args.out, report)
    print(f"report: {args.out}")
    print(f"verdict: {report['verdict']}")
    print(f"sha256: {digest}")
    for reason in report["blocked_reasons"]:
        print(f"blocked: {reason}", file=sys.stderr)
    return 0 if report["verdict"] == "ready" else 5


if __name__ == "__main__":
    raise SystemExit(main())
