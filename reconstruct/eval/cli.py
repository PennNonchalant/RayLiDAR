"""``python -m reconstruct.eval`` command line."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .aggregate import evaluate_all
from .io import read_gt_rays, read_hits, validate_hits_against_rays
from .output import (
    build_metrics_json,
    count_csv_rows,
    validate_metrics_csv,
    validate_metrics_json,
    write_metrics_csv,
    write_metrics_json,
)
from .stats import EvalError

CAVEATS = (
    "gaussian_depth_proxy",
    "training_view_psnr_ssim_not_nvs",
    "single_seed_no_variance",
    "coverage_denominator_frozen_camera0_population",
    "mesh_sha256_is_point_cloud_artifact",
)


def _sha256(path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _real(args: argparse.Namespace) -> int:
    rays, ray_sha = read_gt_rays(args.gt_rays)
    hits, hit_sha = read_hits(args.hits)
    validate_hits_against_rays(rays, hits)
    report = evaluate_all(rays, hits)

    provenance_path = Path(args.provenance)
    predictor_parameters = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    inputs = {
        "mode": args.mode,
        "config_sha256": _sha256(Path(args.run_config)),
        "protocol_sha256": _sha256(Path(args.protocol_doc)),
        "ray_set_sha256": ray_sha,
        "mesh_sha256": args.model_sha256,
        "hit_set_sha256": hit_sha,
        "scene_generator": args.scene_generator,
        "scene_seed": int(args.seed or 0),
        "scene_box_count": 0,
        "scene_thin_box_count": 0,
        "predictor": "rendered_depth_alpha_nearest_pixel",
        "predictor_parameters": predictor_parameters,
    }
    document = build_metrics_json(report, inputs, args.mode, _utc_now())
    document["caveats"] = list(CAVEATS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = write_metrics_json(out_dir / "metrics.json", document)
    csv_path = write_metrics_csv(out_dir / "metrics.csv", document)
    validate_metrics_json(document)
    validate_metrics_csv(csv_path, document)

    manifest = {
        "schema_version": 1,
        "kind": "real_eval",
        "generated_at_utc": _utc_now(),
        "mode": args.mode,
        "scene_generator": args.scene_generator,
        "seed": int(args.seed or 0),
        "inputs": {
            "gt_rays": {"path": str(args.gt_rays), "sha256": ray_sha, "count": len(rays)},
            "hits": {"path": str(args.hits), "sha256": hit_sha, "count": len(hits)},
        },
        "model_sha256": args.model_sha256,
        "row_count": len(document["rows"]),
        "coverage_policy": "frozen_camera0_valid_population_per_frame_and_bucket",
        "caveats": list(CAVEATS),
    }
    (out_dir / "real_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"eval real: wrote {metrics_path} and {csv_path} ({len(rays)} rays)")
    return 0


def _validate(args: argparse.Namespace) -> int:
    document = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    validate_metrics_json(document)
    validate_metrics_csv(args.csv, document)
    print(f"eval validate: {args.metrics} OK ({len(document['rows'])} rows)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reconstruct.eval")
    sub = parser.add_subparsers(dest="command", required=True)

    real = sub.add_parser("real")
    real.add_argument("--gt-rays", required=True)
    real.add_argument("--hits", required=True)
    real.add_argument("--out-dir", required=True)
    real.add_argument("--mode", required=True)
    real.add_argument("--run-config", required=True)
    real.add_argument("--protocol-config", default="")
    real.add_argument("--protocol-doc", required=True)
    real.add_argument("--model-sha256", required=True)
    real.add_argument("--scene-generator", required=True)
    real.add_argument("--seed", default="0")
    real.add_argument("--provenance", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--metrics", required=True)
    validate.add_argument("--csv", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "real":
            return _real(args)
        return _validate(args)
    except (EvalError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
