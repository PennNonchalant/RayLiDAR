"""One frozen PSNR/SSIM harness for both Gaussian methods.

The parameters below are the protocol, not per-method options.  Both 3DGS and
2DGS are scored with the same ``skimage`` calls, the same ``data_range`` and
the same Gaussian-window SSIM.  The label is deliberately explicit: these are
**training-view reconstruction** numbers, not novel-view synthesis results
(D-04, D-12).

Identical images have zero MSE and therefore infinite PSNR; the harness writes
JSON ``null`` and ``psnr_reason="identical"`` instead of an ``Inf`` value.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from skimage import __version__ as skimage_version
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from ..run_utils import sha256_file, write_json_atomic

SCHEMA_VERSION = 1
METRIC_LABEL = "training-view reconstruction metric (not NVS)"
RESOLUTION = (1408, 376)
FROZEN_PARAMS = {
    "data_range": 255,
    "channel_axis": 2,
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
}
CSV_HEADER = "schema_version,method,frame,psnr_db,ssim,mse,psnr_reason"


def _load_protocol(config_path: str | Path) -> dict:
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{config_path}: protocol config must be a mapping")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{config_path}: unsupported schema_version "
            f"{document.get('schema_version')!r}"
        )
    params = document.get("psnr_ssim")
    if not isinstance(params, dict):
        raise ValueError(f"{config_path}: missing psnr_ssim section")
    for key, expected in FROZEN_PARAMS.items():
        if params.get(key) != expected:
            raise ValueError(
                f"{config_path}: psnr_ssim.{key} must be {expected!r}, "
                f"found {params.get(key)!r}"
            )
    label = params.get("label", METRIC_LABEL)
    if label != METRIC_LABEL:
        raise ValueError(
            f"{config_path}: psnr_ssim.label must be {METRIC_LABEL!r}"
        )
    return document


def compute_frame_metrics(gt_u8: object, pred_u8: object) -> dict:
    """Compute PSNR/SSIM for one RGB uint8 frame pair."""
    gt = np.asarray(gt_u8)
    pred = np.asarray(pred_u8)
    if gt.dtype != np.uint8 or pred.dtype != np.uint8:
        raise ValueError("compute_frame_metrics requires uint8 arrays")
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pred.shape}")
    if gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError("frames must be HxWx3 RGB arrays")

    difference = gt.astype(np.float64) - pred.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse == 0.0:
        psnr = None
        psnr_reason = "identical"
    else:
        psnr = float(
            peak_signal_noise_ratio(gt, pred, data_range=FROZEN_PARAMS["data_range"])
        )
        psnr_reason = None

    ssim = float(
        structural_similarity(
            gt,
            pred,
            data_range=FROZEN_PARAMS["data_range"],
            channel_axis=FROZEN_PARAMS["channel_axis"],
            gaussian_weights=FROZEN_PARAMS["gaussian_weights"],
            sigma=FROZEN_PARAMS["sigma"],
            use_sample_covariance=FROZEN_PARAMS["use_sample_covariance"],
        )
    )

    return {
        "psnr": psnr,
        "ssim": ssim,
        "psnr_reason": psnr_reason,
        "mse": mse,
        "shape": [int(gt.shape[0]), int(gt.shape[1])],
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(per_frame: list[dict]) -> dict:
    """Aggregate per-frame metrics with sample standard deviation (``n-1``)."""
    if not per_frame:
        raise ValueError("summarize requires at least one frame")

    psnr_values = [
        float(entry["psnr"])
        for entry in per_frame
        if entry.get("psnr") is not None
    ]
    ssim_values = [float(entry["ssim"]) for entry in per_frame]

    psnr_stats = _stats(psnr_values)
    ssim_stats = _stats(ssim_values)
    return {
        "n": len(per_frame),
        "psnr_mean": psnr_stats["mean"],
        "psnr_std": psnr_stats["std"],
        "psnr_min": psnr_stats["min"],
        "psnr_max": psnr_stats["max"],
        "ssim_mean": ssim_stats["mean"],
        "ssim_std": ssim_stats["std"],
        "ssim_min": ssim_stats["min"],
        "ssim_max": ssim_stats["max"],
    }


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if (array.shape[1], array.shape[0]) != RESOLUTION:
        raise ValueError(
            f"{path}: expected {RESOLUTION[0]}x{RESOLUTION[1]}, "
            f"found {array.shape[1]}x{array.shape[0]}"
        )
    return array


def _write_text_atomic(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return str(path)


def run_method(
    gt_dir: str | Path,
    render_dir: str | Path,
    frames: list[int],
    method: str,
    out_dir: str | Path,
    config_path: str | Path,
) -> dict:
    """Score every frame and write ``psnr_ssim.json`` + ``psnr_ssim.csv``."""
    _load_protocol(config_path)
    if not method:
        raise ValueError("method must be a non-empty string")
    if not frames:
        raise ValueError("frames must not be empty")

    gt_root = Path(gt_dir)
    render_root = Path(render_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    per_frame: list[dict] = []
    gt_hashes: dict[str, str] = {}
    render_hashes: dict[str, str] = {}
    for frame in frames:
        frame_int = int(frame)
        gt_path = gt_root / f"{frame_int:010d}.png"
        render_path = render_root / f"{frame_int:010d}_rgb.png"
        gt = _read_rgb(gt_path)
        pred = _read_rgb(render_path)
        metrics = compute_frame_metrics(gt, pred)
        per_frame.append({"frame": frame_int, **metrics})
        gt_hashes[str(frame_int)] = sha256_file(gt_path)
        render_hashes[str(frame_int)] = sha256_file(render_path)

    aggregate = summarize(per_frame)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "label": METRIC_LABEL,
        "params": {
            "skimage_version": skimage_version,
            **FROZEN_PARAMS,
        },
        "frames": [int(frame) for frame in frames],
        "resolution": [RESOLUTION[0], RESOLUTION[1]],
        "inputs": {
            "gt_dir": str(gt_root),
            "render_dir": str(render_root),
            "per_image_sha256": {"gt": gt_hashes, "render": render_hashes},
        },
        "per_frame": per_frame,
        "aggregate": aggregate,
    }
    write_json_atomic(output_root / "psnr_ssim.json", payload)

    lines = [CSV_HEADER]
    for entry in per_frame:
        psnr_field = "" if entry["psnr"] is None else repr(float(entry["psnr"]))
        reason = entry["psnr_reason"] or ""
        lines.append(
            ",".join(
                [
                    str(SCHEMA_VERSION),
                    method,
                    str(entry["frame"]),
                    psnr_field,
                    repr(float(entry["ssim"])),
                    repr(float(entry["mse"])),
                    reason,
                ]
            )
        )
    _write_text_atomic(output_root / "psnr_ssim.csv", "\n".join(lines) + "\n")
    return payload


__all__ = [
    "CSV_HEADER",
    "FROZEN_PARAMS",
    "METRIC_LABEL",
    "RESOLUTION",
    "compute_frame_metrics",
    "run_method",
    "summarize",
]
