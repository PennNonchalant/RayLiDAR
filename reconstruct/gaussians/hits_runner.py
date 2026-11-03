"""Generate one ordered RLHT ``hits.bin`` per Gaussian method.

Both methods go through the shared :func:`reconstruct.gaussians.protocol.hits.build_hits`
path with the frozen ``depth_convention.json`` decision, the frozen alpha
threshold, and the full 60-frame camera-0 GT ray set.  The builder also writes
the frozen ``occluded == 0`` subset.  This stage only checks identity/coverage
of the generated files; it never computes a range, P-R-F, CD, or miss metric.

If ``depth_convention.json`` is absent, the probe stage runs first so a hit set
can never be built on an unfrozen convention.
"""

from __future__ import annotations

import numpy as np

from pathlib import Path

from . import probe
from .protocol.hits import build_hits
from .run_utils import read_json, sha256_file, utc_now, write_json_atomic

SCHEMA_VERSION = 1
METHODS = ("3dgs", "2dgs")
SUPPORTED_CONVENTIONS = ("z_depth", "range")
REQUIRED_HASH_KEYS = (
    "hits_sha256",
    "hits_unoccluded_sha256",
    "gt_rays_unoccluded_sha256",
)
HITS_DIRNAME = "hits"


def gt_rays_count(cfg: dict) -> int:
    """Return the frozen camera-0 ray count from the RLGR header."""
    path = Path(probe.gt_rays_path(cfg))
    header = path.read_bytes()[:32]
    if len(header) < 32 or header[:4] != b"RLGR":
        raise RuntimeError(f"{path}: not a valid RLGR gt_rays.bin")
    return int(np.frombuffer(header[8:12], dtype="<u4")[0])


def _config_digest(context: dict, cfg: dict) -> str:
    digest = context.get("config_sha256")
    if digest:
        return str(digest)
    from .config import config_sha256  # noqa: PLC0415 - avoid an import cycle

    return config_sha256(cfg)


def locate_run_manifest(cfg: dict, method: str) -> Path:
    """Return the non-smoke run manifest for one method."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    path = (
        probe.artifacts_root(cfg)
        / "runs"
        / f"{method}-{cfg['run_id']}"
        / "run_manifest.json"
    )
    if not path.is_file():
        raise RuntimeError(f"run manifest missing: {path}; run --train-{method} first")
    manifest = read_json(path)
    if bool(manifest.get("smoke")):
        raise RuntimeError(f"{path}: smoke runs cannot produce Phase-5 hits")
    if manifest.get("method") != method:
        raise RuntimeError(
            f"{path}: method {manifest.get('method')!r} != {method!r}"
        )
    return path


def read_convention(path: str | Path) -> dict:
    """Read ``depth_convention.json`` and require a frozen decision."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"depth convention missing: {path}; run the probe stage first"
        )
    document = read_json(path)
    decision = document.get("decision")
    if decision not in SUPPORTED_CONVENTIONS:
        raise RuntimeError(
            f"{path}: decision {decision!r} is not one of "
            f"{list(SUPPORTED_CONVENTIONS)}"
        )
    return document


def run_method(cfg: dict, method: str, convention: str) -> dict:
    """Build ``hits.bin`` + the unoccluded subset for one method."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(f"unsupported depth convention: {convention!r}")
    alpha_threshold = float(cfg["protocol"]["alpha_threshold"])

    render_dir = probe.locate_render_dir(cfg, method)
    run_manifest_path = locate_run_manifest(cfg, method)
    run_manifest = read_json(run_manifest_path)

    out_dir = probe.artifacts_root(cfg) / HITS_DIRNAME / method
    manifest = build_hits(
        probe.gt_rays_path(cfg),
        probe.flags_path(cfg),
        probe.cameras_json_path(cfg),
        render_dir,
        method,
        cfg["frames"],
        out_dir,
        convention=convention,
        alpha_threshold=alpha_threshold,
    )

    counts = manifest.get("counts") or {}
    total = int(counts.get("total", -1))
    expected = gt_rays_count(cfg)
    if total != expected:
        raise RuntimeError(
            f"{method}: hits total {total} != camera-0 total {expected}"
        )
    for key in REQUIRED_HASH_KEYS:
        value = manifest.get(key)
        if not (isinstance(value, str) and len(value) == 64):
            raise RuntimeError(f"{method}: hits manifest is missing {key}")
    recorded_convention = (manifest.get("protocol") or {}).get("convention")
    if recorded_convention != convention:
        raise RuntimeError(
            f"{method}: hits manifest convention {recorded_convention!r} != "
            f"frozen {convention!r}"
        )
    recorded_alpha = (manifest.get("protocol") or {}).get("alpha_threshold")
    if recorded_alpha != alpha_threshold:
        raise RuntimeError(
            f"{method}: hits manifest alpha_threshold {recorded_alpha!r} != "
            f"{alpha_threshold!r}"
        )

    hits_path = out_dir / "hits.bin"
    hits_manifest_path = out_dir / "hits_manifest.json"
    render_manifest_path = render_dir / "render_manifest.json"
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": f"hits-{method}",
        "method": method,
        "generated_at_utc": utc_now(),
        "convention": convention,
        "alpha_threshold": alpha_threshold,
        "hits_dir": str(out_dir),
        "hits": str(hits_path),
        "hits_sha256": manifest["hits_sha256"],
        "hits_unoccluded": str(out_dir / "hits_unoccluded.bin"),
        "hits_unoccluded_sha256": manifest["hits_unoccluded_sha256"],
        "gt_rays_unoccluded": str(out_dir / "gt_rays_unoccluded.bin"),
        "gt_rays_unoccluded_sha256": manifest["gt_rays_unoccluded_sha256"],
        "counts": counts,
        "reasons": manifest.get("reasons", {}),
        "hits_manifest": str(hits_manifest_path),
        "hits_manifest_sha256": sha256_file(hits_manifest_path),
        "render_dir": str(render_dir),
        "render_manifest": str(render_manifest_path),
        "render_manifest_sha256": sha256_file(render_manifest_path),
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "point_cloud_sha256": run_manifest.get("point_cloud_sha256"),
        "checkpoint_sha256": run_manifest.get("checkpoint_sha256"),
    }
    write_json_atomic(
        probe.artifacts_root(cfg) / "stages" / f"hits-{method}.json", record
    )
    return record


def run(context: dict) -> dict:
    """Pipeline stage: generate hits for both methods with one convention."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("hits stage requires context['cfg']")

    convention_file = probe.convention_path(cfg)
    if not convention_file.is_file():
        probe.run(context)
    document = read_convention(convention_file)
    decision = document["decision"]

    results = {
        method: run_method(cfg, method, decision) for method in METHODS
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "hits",
        "generated_at_utc": utc_now(),
        "config_sha256": _config_digest(context, cfg),
        "depth_convention": str(convention_file),
        "depth_convention_sha256": sha256_file(convention_file),
        "decision": decision,
        "alpha_threshold": float(cfg["protocol"]["alpha_threshold"]),
        "expected_total": gt_rays_count(cfg),
        "methods": results,
    }
    write_json_atomic(
        probe.artifacts_root(cfg) / HITS_DIRNAME / "hits_summary.json", summary
    )
    return summary


__all__ = [
    "HITS_DIRNAME",
    "METHODS",
    "REQUIRED_HASH_KEYS",
    "SUPPORTED_CONVENTIONS",
    "locate_run_manifest",
    "read_convention",
    "run",
    "run_method",
]
