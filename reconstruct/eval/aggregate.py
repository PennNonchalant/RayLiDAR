"""Aggregation of ray samples into the frozen Phase-3 report shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import Cloud, compute_chamfer, compute_prf, gt_points, predicted_points
from .stats import (
    BUCKET_NAMES,
    PRIMARY_BUCKETS,
    PRIMARY_TAU_M,
    TAUS_M,
    bucket_of,
    make_cell,
    mean_and_sample_std,
)


@dataclass
class Sample:
    ray_id: int
    frame: int
    camera: int
    bucket: int
    t_gt: float
    t_min: float
    t_max: float
    origin_w: np.ndarray
    dir_w: np.ndarray
    t_pred: float

    @property
    def hit(self) -> bool:
        return bool(np.isfinite(self.t_pred))


def _errors(samples: list[Sample]) -> list[float]:
    return [sample.t_pred - sample.t_gt for sample in samples if sample.hit]


def _hits(samples: list[Sample]) -> int:
    return sum(1 for sample in samples if sample.hit)


def _cell(samples: list[Sample], n_raw: int) -> dict:
    return make_cell(n_raw, len(samples), _hits(samples), _errors(samples))


def _prf_and_cd(samples: list[Sample]) -> tuple[dict, dict]:
    predicted = predicted_points(samples)
    gt = gt_points(samples)
    pred_cloud = Cloud(predicted) if len(predicted) else None
    gt_cloud = Cloud(gt) if len(gt) else None
    block = {}
    for tau in TAUS_M:
        block[f"{tau:.2f}"] = {
            "nearest": compute_prf(predicted, gt, tau, False, pred_cloud=pred_cloud, gt_cloud=gt_cloud),
            "greedy": compute_prf(predicted, gt, tau, True, pred_cloud=pred_cloud, gt_cloud=gt_cloud),
        }
    return block, compute_chamfer(predicted, gt, pred_cloud=pred_cloud, gt_cloud=gt_cloud)


def _group_report(samples: list[Sample], n_raw: int) -> dict:
    prf, cd = _prf_and_cd(samples)
    return {"cell": _cell(samples, n_raw), "prf": prf, "cd": cd}


def evaluate_all(rays: list, hits: list) -> dict:
    samples: list[Sample] = []
    raw: dict[tuple[int, int, int], int] = {}
    raw_camera: dict[int, int] = {}
    raw_frame: dict[int, int] = {}
    for ray, hit in zip(rays, hits):
        bucket = bucket_of(ray.t_gt)
        samples.append(
            Sample(
                ray_id=ray.ray_id,
                frame=ray.frame,
                camera=ray.camera,
                bucket=bucket,
                t_gt=ray.t_gt,
                t_min=ray.t_min,
                t_max=ray.t_max,
                origin_w=ray.origin_w,
                dir_w=ray.dir_w,
                t_pred=float(hit.t_pred) if hit.hit else float("inf"),
            )
        )
        key = (ray.frame, ray.camera, bucket)
        raw[key] = raw.get(key, 0) + 1
        raw_camera[ray.camera] = raw_camera.get(ray.camera, 0) + 1
        raw_frame[ray.frame] = raw_frame.get(ray.frame, 0) + 1

    frames = sorted({sample.frame for sample in samples})
    primary = [sample for sample in samples if sample.bucket != PRIMARY_BUCKETS]
    secondary = [sample for sample in samples if sample.bucket == PRIMARY_BUCKETS]

    micro_primary = _group_report(primary, len(samples))
    micro_secondary = _group_report(secondary, len(secondary))

    per_camera = []
    for camera in sorted({sample.camera for sample in samples}):
        camera_primary = [sample for sample in samples if sample.camera == camera and sample.bucket != PRIMARY_BUCKETS]
        per_camera.append({"camera": camera, "cell": _cell(camera_primary, raw_camera.get(camera, 0))})

    per_frame = []
    precisions: list[float] = []
    recalls: list[float] = []
    f_scores: list[float] = []
    for frame in frames:
        frame_samples = [sample for sample in samples if sample.frame == frame]
        frame_primary = [sample for sample in frame_samples if sample.bucket != PRIMARY_BUCKETS]
        cell = _cell(frame_primary, raw_frame.get(frame, 0))
        per_frame.append({"frame": frame, "primary": cell})
        predicted = predicted_points(frame_primary)
        gt = gt_points(frame_primary)
        prf = compute_prf(predicted, gt, PRIMARY_TAU_M, False)
        if prf["precision"] is not None:
            precisions.append(prf["precision"])
        if prf["recall"] is not None:
            recalls.append(prf["recall"])
        if prf["f_score"] is not None:
            f_scores.append(prf["f_score"])

    macro = {
        "n_frames_total": len(per_frame),
        "n_frames_with_range": sum(1 for entry in per_frame if entry["primary"]["range"]["n_hits"] != 0),
        "n_frames_without_range": 0,
    }
    macro["n_frames_without_range"] = macro["n_frames_total"] - macro["n_frames_with_range"]
    for key, path in (
        ("mae", "range"),
        ("rmse", "range"),
        ("medae", "range"),
        ("bias", "range"),
        ("p90", "range"),
        ("p95", "range"),
        ("miss_rate", "miss"),
        ("ray_recall", "miss"),
        ("coverage", "cell"),
    ):
        values = []
        for entry in per_frame:
            cell = entry["primary"]
            value = cell["coverage"] if path == "cell" else cell[path][key]
            if value is not None:
                values.append(float(value))
        mean, std = mean_and_sample_std(values)
        macro[f"{key}_mean"] = mean
        macro[f"{key}_std"] = std
    for name, values in (("precision", precisions), ("recall", recalls), ("f_score", f_scores)):
        mean, std = mean_and_sample_std(values)
        macro[f"{name}_mean"] = mean
        macro[f"{name}_std"] = std

    rows = []
    for frame in frames:
        for camera in (0, 1):
            for bucket in range(4):
                row_samples = [
                    sample
                    for sample in samples
                    if sample.frame == frame and sample.camera == camera and sample.bucket == bucket
                ]
                report = _group_report(row_samples, raw.get((frame, camera, bucket), 0))
                rows.append(
                    {
                        "frame": frame,
                        "camera": camera,
                        "bucket": BUCKET_NAMES[bucket],
                        "cell": report["cell"],
                        "prf": report["prf"],
                        "cd": report["cd"],
                    }
                )

    return {
        "rows": rows,
        "micro_primary": micro_primary,
        "micro_secondary": micro_secondary,
        "per_camera": per_camera,
        "per_frame": per_frame,
        "macro": macro,
    }
