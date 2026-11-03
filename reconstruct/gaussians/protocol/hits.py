"""Build one ordered ``hits.bin`` per method from rendered depth/alpha maps.

Every camera-0 GT ray is sampled with :func:`reconstruct.gaussians.protocol.depth.sample_ray`
at the half-up nearest pixel.  The output has exactly one RLHT record per RLGR
record, in the same order and with the same ``ray_id``.  A miss is
``hit=0, t_pred=+inf, prim_id=0xFFFFFFFF``.  ``prim_id`` is always the
sentinel because the rendered-depth proxy does not expose a triangle id.

The builder also writes the frozen ``occluded == 0`` subset
(``gt_rays_unoccluded.bin`` + ``hits_unoccluded.bin``) and a manifest with
counts, per-frame reasons, input/output hashes, and the frozen protocol
parameters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..run_utils import sha256_file, utc_now, write_json_atomic
from .cameras import load_cameras
from .depth import REASONS, sample_ray
from .gt_rays_io import (
    PRIM_ID_UNAVAILABLE,
    filter_gt_rays,
    filter_hits,
    read_flags,
    read_gt_rays,
    write_hits,
)

SCHEMA_VERSION = 1
DEFAULT_RESOLUTION = (1408, 376)
FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _empty_reasons() -> dict[str, int]:
    return {reason: 0 for reason in REASONS}


def _validate_hits(gt_rays: list[dict], hits: list[dict]) -> None:
    if len(hits) != len(gt_rays):
        raise RuntimeError(
            f"hits count {len(hits)} does not match gt_rays count {len(gt_rays)}"
        )
    for index, (gt_ray, hit) in enumerate(zip(gt_rays, hits)):
        if int(hit["ray_id"]) != int(gt_ray["ray_id"]):
            raise RuntimeError(
                f"hit {index}: ray_id {hit['ray_id']} != gt ray_id "
                f"{gt_ray['ray_id']}"
            )
        if int(hit["prim_id"]) != PRIM_ID_UNAVAILABLE:
            raise RuntimeError(f"hit {index}: prim_id must be 0xFFFFFFFF")
        t_pred = float(hit["t_pred"])
        if hit["hit"]:
            if not (math.isfinite(t_pred) and t_pred > 0.0):
                raise RuntimeError(f"hit {index}: a hit needs finite t_pred")
            if not (
                float(gt_ray["t_min"]) <= t_pred <= float(gt_ray["t_max"])
            ):
                raise RuntimeError(
                    f"hit {index}: t_pred {t_pred!r} outside "
                    f"[{gt_ray['t_min']!r}, {gt_ray['t_max']!r}]"
                )
        else:
            if not (math.isinf(t_pred) and t_pred > 0.0):
                raise RuntimeError(f"hit {index}: a miss must use t_pred=+inf")


def _load_maps(
    render_dir: Path,
    frame_list: list[int],
    width: int,
    height: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    depth_cache: dict[int, np.ndarray] = {}
    alpha_cache: dict[int, np.ndarray] = {}
    for frame in frame_list:
        depth_path = render_dir / f"{frame:010d}_depth.npy"
        alpha_path = render_dir / f"{frame:010d}_alpha.npy"
        depth = np.load(depth_path, allow_pickle=False)
        alpha = np.load(alpha_path, allow_pickle=False)
        for name, array, path in (
            ("depth", depth, depth_path),
            ("alpha", alpha, alpha_path),
        ):
            if array.ndim != 2:
                raise ValueError(f"{path}: {name} must be a 2-D array")
            if array.dtype not in FLOAT_DTYPES:
                raise ValueError(
                    f"{path}: {name} dtype must be float32/float64, "
                    f"found {array.dtype}"
                )
            if tuple(int(value) for value in array.shape) != (height, width):
                raise ValueError(
                    f"{path}: {name} shape {array.shape} does not match "
                    f"({height}, {width})"
                )
        depth_cache[frame] = depth
        alpha_cache[frame] = alpha
    return depth_cache, alpha_cache


def _check_render_manifest(
    render_dir: Path, width: int, height: int
) -> None:
    manifest_path = render_dir / "render_manifest.json"
    if not manifest_path.is_file():
        return
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolution = document.get("resolution")
    if resolution is None:
        return
    if [int(value) for value in resolution] != [width, height]:
        raise ValueError(
            f"{manifest_path}: resolution {resolution} does not match "
            f"[{width}, {height}]"
        )


def build_hits(
    gt_rays_path: str | Path,
    flags_path: str | Path,
    cameras_json: str | Path,
    render_dir: str | Path,
    method: str,
    frames: Iterable[int],
    out_dir: str | Path,
    convention: str = "z_depth",
    alpha_threshold: float = 0.5,
    resolution: tuple[int, int] | None = None,
    expected_frames: int = 60,
) -> dict:
    """Generate ``hits.bin``, the unoccluded subset, and ``hits_manifest.json``.

    ``resolution``/``expected_frames`` are test seams: production uses the
    frozen ``[1408, 376]`` and 60 camera entries.
    """
    if not method:
        raise ValueError("method must be a non-empty string")

    gt_rays = read_gt_rays(gt_rays_path)
    for index, gt_ray in enumerate(gt_rays):
        if int(gt_ray["camera"]) != 0:
            raise ValueError(
                f"gt ray {index}: camera {gt_ray['camera']} is not camera 0; "
                "Phase 5 evaluates camera-0 rays only"
            )
    flags = read_flags(flags_path)
    if len(flags) != len(gt_rays):
        raise RuntimeError(
            f"flags count {len(flags)} does not match gt_rays count {len(gt_rays)}"
        )
    for index, (gt_ray, flag) in enumerate(zip(gt_rays, flags)):
        if int(flag["ray_id"]) != int(gt_ray["ray_id"]):
            raise RuntimeError(
                f"flag {index}: ray_id {flag['ray_id']} != gt ray_id "
                f"{gt_ray['ray_id']}"
            )
        if int(flag["occluded"]) not in (0, 1):
            raise RuntimeError(f"flag {index}: occluded must be 0 or 1")

    cameras = load_cameras(cameras_json, expected_frames=expected_frames)
    width, height = (
        (int(resolution[0]), int(resolution[1]))
        if resolution is not None
        else (int(cameras["width"]), int(cameras["height"]))
    )
    if (width, height) != (int(cameras["width"]), int(cameras["height"])):
        raise ValueError(
            f"resolution {(width, height)} disagrees with cameras "
            f"({cameras['width']}, {cameras['height']})"
        )

    frame_list = [int(frame) for frame in frames]
    if not frame_list:
        raise ValueError("frames must not be empty")
    if len(set(frame_list)) != len(frame_list):
        raise ValueError("frames must not contain duplicates")

    render_root = Path(render_dir)
    _check_render_manifest(render_root, width, height)
    depth_cache, alpha_cache = _load_maps(
        render_root, frame_list, width, height
    )

    reason_counts = _empty_reasons()
    per_frame = {
        frame: {
            "total": 0,
            "hits": 0,
            "misses": 0,
            "reasons": _empty_reasons(),
        }
        for frame in frame_list
    }

    hits: list[dict] = []
    for gt_ray in gt_rays:
        frame = int(gt_ray["frame"])
        if frame not in depth_cache:
            raise ValueError(
                f"gt ray {gt_ray['ray_id']} uses frame {frame} with no render"
            )
        sample = sample_ray(
            depth_cache[frame],
            alpha_cache[frame],
            gt_ray,
            cameras["frames"][frame]["T_C_W"],
            cameras["K"],
            alpha_threshold=alpha_threshold,
            convention=convention,
        )
        reason = sample["reason"]
        reason_counts[reason] += 1
        bucket = per_frame[frame]
        bucket["total"] += 1
        bucket["reasons"][reason] += 1
        if sample["hit"]:
            bucket["hits"] += 1
            hits.append(
                {
                    "ray_id": int(gt_ray["ray_id"]),
                    "hit": True,
                    "t_pred": float(sample["t_pred"]),
                    "prim_id": PRIM_ID_UNAVAILABLE,
                }
            )
        else:
            bucket["misses"] += 1
            hits.append(
                {
                    "ray_id": int(gt_ray["ray_id"]),
                    "hit": False,
                    "t_pred": math.inf,
                    "prim_id": PRIM_ID_UNAVAILABLE,
                }
            )

    _validate_hits(gt_rays, hits)

    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    hits_path = output_root / "hits.bin"
    write_hits(hits_path, hits)

    keep_ids = [int(flag["ray_id"]) for flag in flags if int(flag["occluded"]) == 0]
    keep_set = set(keep_ids)
    unoccluded_gt_path = output_root / "gt_rays_unoccluded.bin"
    unoccluded_hits_path = output_root / "hits_unoccluded.bin"
    kept_rays = filter_gt_rays(gt_rays_path, unoccluded_gt_path, keep_set)
    kept_hits = filter_hits(hits_path, unoccluded_hits_path, keep_set)
    if kept_rays != keep_ids or kept_hits != keep_ids:
        raise RuntimeError("unoccluded subset did not preserve the frozen order")

    total_hits = sum(1 for hit in hits if hit["hit"])
    unoccluded_hits = sum(
        1 for hit in hits if hit["hit"] and int(hit["ray_id"]) in keep_set
    )
    unoccluded_total = len(keep_ids)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "render_dir": str(render_root),
        "gt_rays_sha256": sha256_file(gt_rays_path),
        "flags_sha256": sha256_file(flags_path),
        "hits_sha256": sha256_file(hits_path),
        "hits_unoccluded_sha256": sha256_file(unoccluded_hits_path),
        "gt_rays_unoccluded_sha256": sha256_file(unoccluded_gt_path),
        "counts": {
            "total": len(gt_rays),
            "hits": total_hits,
            "misses": len(gt_rays) - total_hits,
            "unoccluded_total": unoccluded_total,
            "unoccluded_hits": unoccluded_hits,
            "unoccluded_misses": unoccluded_total - unoccluded_hits,
        },
        "reasons": dict(reason_counts),
        "per_frame": {
            str(frame): per_frame[frame] for frame in frame_list
        },
        "protocol": {
            "alpha_threshold": float(alpha_threshold),
            "convention": convention,
            "pixel_rounding": "half_up",
            "depth_units": "z_depth",
            "range_formula": "z * ||((u-cx)/fx,(v-cy)/fy,1)||",
        },
        "generated_at_utc": utc_now(),
    }
    write_json_atomic(output_root / "hits_manifest.json", manifest)
    return manifest


__all__ = ["DEFAULT_RESOLUTION", "build_hits"]
