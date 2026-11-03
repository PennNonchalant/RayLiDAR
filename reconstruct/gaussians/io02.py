"""IO-02: export the frozen camera-0 Phase-2 rays as a W-frame ``gt_rays.bin``.

The Python exporter in :mod:`reconstruct.data_preprocess.export` owns the
conversion, hash verification, and count checks. This stage resolves the
Phase-5 paths from the config, verifies the emitted manifest, and persists
``<artifacts_root>/stages/io02.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DEFAULT_DATA_ROOT = Path("/data-20T/hybrid-lidar-sim")
DEFAULT_ARTIFACTS_ROOT = Path(
    os.environ.get("RAYLIDAR_ARTIFACTS_ROOT", DEFAULT_DATA_ROOT / "artifacts" / "05")
)
DEFAULT_BUNDLE = Path(
    os.environ.get("RECONSTRUCT_BUNDLE", DEFAULT_DATA_ROOT / "exports" / "reconstruct-bundle")
)
DEFAULT_CAMERA = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def default_config() -> dict[str, Any]:
    """The default config used when the stage is invoked without a pipeline context."""
    return {
        "schema_version": 1,
        "run_id": "p5-gaussian-60f",
        "paths": {
            "bundle": str(DEFAULT_BUNDLE),
            "artifacts_root": str(DEFAULT_ARTIFACTS_ROOT),
        },
    }


def config_fingerprint(config_path: Path | None = None) -> str:
    cfg = _load_config(config_path) if config_path is not None else default_config()
    return _canonical_sha256(cfg)


def _load_config(path: Path) -> dict[str, Any]:
    import yaml  # local import: the stage must stay importable without PyYAML

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("cfg.paths must be a mapping")
    artifacts_root = Path(paths.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT))
    bundle = Path(paths.get("bundle", DEFAULT_BUNDLE))
    phase2 = Path(paths.get("phase2", artifacts_root.parent / "02"))
    out_dir = Path(paths.get("io02_dir", artifacts_root / "io02"))
    return {
        "artifacts_root": artifacts_root,
        "bundle": bundle,
        "phase2": phase2,
        "out_dir": out_dir,
        "rays_dir": phase2 / "rays",
        "rays_manifest": phase2 / "rays" / "rays_manifest.json",
        "freeze_json": phase2 / "rays" / "RAY_SET_FREEZE.json",
        "gt_manifest": phase2 / "gt" / "gt_manifest.json",
        "cameras_json": bundle / "geometry" / "cameras.json",
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    data = payload.encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def run_export(
    cfg: dict[str, Any],
    config_sha256: str,
    cli: Path | None = None,
    camera: int = DEFAULT_CAMERA,
) -> dict[str, Any]:
    """Export ``gt_rays.bin`` with the Python preprocessing module."""
    from ..data_preprocess.export import export_gt_rays as python_export_gt_rays

    paths = resolve_paths(cfg)
    ray_manifest = json.loads(Path(paths["rays_manifest"]).read_text(encoding="utf-8"))
    camera_counts = (ray_manifest.get("totals") or {}).get("camera_counts") or {}
    expected_total = int(camera_counts.get("image_00", -1))
    if expected_total <= 0:
        raise RuntimeError(
            f"{paths['rays_manifest']}: totals.camera_counts.image_00 must be positive"
        )
    manifest = python_export_gt_rays(
        paths["cameras_json"],
        paths["rays_manifest"],
        paths["gt_manifest"],
        paths["rays_dir"],
        paths["out_dir"],
        camera=int(camera),
        expected_total=expected_total,
    )

    manifest_path = paths["out_dir"] / "gt_rays_manifest.json"
    gt_rays_path = paths["out_dir"] / "gt_rays.bin"
    total = int(manifest.get("counts", {}).get("total", -1))
    if total != expected_total:
        raise RuntimeError(
            f"gt_rays manifest total {total} != rays manifest camera-0 total "
            f"{expected_total}"
        )
    recorded = str(manifest.get("output", {}).get("sha256", ""))
    actual = _sha256_file(gt_rays_path)
    if recorded != actual:
        raise RuntimeError(f"gt_rays.bin SHA-256 mismatch: manifest {recorded!r}, actual {actual!r}")

    stage = {
        "schema_version": SCHEMA_VERSION,
        "stage": "io02",
        "generated_at_utc": _utc_now(),
        "config_sha256": config_sha256,
        "camera": DEFAULT_CAMERA,
        "count": total,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "gt_rays": str(gt_rays_path),
        "gt_rays_sha256": actual,
        "flags": str(paths["out_dir"] / "gt_rays_flags.bin"),
    }
    stage_path = paths["artifacts_root"] / "stages" / "io02.json"
    _write_json_atomic(stage_path, stage)
    stage["stage_path"] = str(stage_path)
    return stage


def run(ctx: dict[str, Any]) -> dict[str, Any]:
    cfg = ctx.get("cfg")
    if cfg is None:
        cfg = default_config()
    if not isinstance(cfg, dict):
        raise ValueError("ctx.cfg must be a mapping")
    config_sha256 = str(ctx.get("config_sha256") or _canonical_sha256(cfg))
    return run_export(cfg, config_sha256)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the camera-0 W-frame gt_rays.bin")
    parser.add_argument("--config", type=Path, default=None, help="Phase-5 YAML (optional)")
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--phase2", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cli", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    parser.add_argument("--config-check", action="store_true", help="print config_sha256 and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config) if args.config is not None else default_config()
    paths = cfg.setdefault("paths", {})
    if args.artifacts_root is not None:
        paths["artifacts_root"] = str(args.artifacts_root)
    if args.bundle is not None:
        paths["bundle"] = str(args.bundle)
    if args.phase2 is not None:
        paths["phase2"] = str(args.phase2)
    if args.out_dir is not None:
        paths["io02_dir"] = str(args.out_dir)

    config_sha256 = _canonical_sha256(cfg)
    if args.config_check:
        print(config_sha256)
        return 0
    if args.camera != DEFAULT_CAMERA:
        print("error: the frozen IO-02 export is camera 0 only", file=sys.stderr)
        return 2

    stage = run_export(cfg, config_sha256, cli=args.cli, camera=args.camera)
    print(f"stage: {stage['stage_path']}")
    print(f"manifest: {stage['manifest']}")
    print(f"count: {stage['count']}")
    print(f"gt_rays_sha256: {stage['gt_rays_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
