"""Figures for the frozen 3DGS-vs-2DGS comparison report.

All figures use matplotlib's non-interactive ``Agg`` backend.  They are pure
presentation: every number comes from ``comparison.json``, which in turn comes
from validated Phase-3 metrics and the shared PSNR/SSIM harness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

from .run_utils import read_json, sha256_file  # noqa: E402

METHODS = ("3dgs", "2dgs")
BUCKETS = ("0_20", "20_40", "40_50", "50_60")
TOTAL_LOSS_TAG = "train_loss_patches/total_loss"
L1_LOSS_TAGS = (
    "train_loss_patches/l1_loss",
    "train/loss_viewpoint - l1_loss",
)


def _save(fig, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return {"path": str(path), "sha256": sha256_file(path)}


def _frame_axis(comparison: dict, method: str) -> tuple[list[int], list[float], list[float]]:
    entries = comparison["psnr_ssim"][method]["per_frame"]
    frames = [int(entry["frame"]) for entry in entries]
    psnr = [float(entry["psnr"]) if entry.get("psnr") is not None else np.nan for entry in entries]
    ssim = [float(entry["ssim"]) for entry in entries]
    return frames, psnr, ssim


def psnr_ssim_per_frame(comparison: dict, out_png: str | Path) -> dict:
    """Two subplots: per-frame PSNR and SSIM for both methods."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"3dgs": "#1f77b4", "2dgs": "#d62728"}
    for method in METHODS:
        frames, psnr, ssim = _frame_axis(comparison, method)
        axes[0].plot(frames, psnr, label=method.upper(), color=colors[method], linewidth=1.4)
        axes[1].plot(frames, ssim, label=method.upper(), color=colors[method], linewidth=1.4)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xlabel("KITTI-360 frame")
    axes[0].set_title("Training-view reconstruction metrics (not NVS)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    return _save(fig, Path(out_png))


def error_by_bucket(comparison: dict, out_png: str | Path) -> dict:
    """Grouped bars for range MAE/RMSE and miss rate per frozen bucket."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("range_mae_macro_mean", "Range MAE (m)"),
        ("range_rmse_macro_mean", "Range RMSE (m)"),
        ("miss_rate_macro_mean", "Miss rate"),
    ]
    x = np.arange(len(BUCKETS))
    width = 0.38
    colors = {"3dgs": "#1f77b4", "2dgs": "#d62728"}
    for axis, (key, title) in zip(axes, metrics):
        for offset, method in zip((-width / 2, width / 2), METHODS):
            values = []
            for bucket in BUCKETS:
                row = next(
                    item
                    for item in comparison["gt_ray"]["all"][method]["buckets"]
                    if item["bucket"] == bucket
                )
                value = row.get(key)
                values.append(np.nan if value is None else float(value))
            axis.bar(x + offset, values, width, label=method.upper(), color=colors[method])
        axis.set_xticks(x)
        axis.set_xticklabels(BUCKETS)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend()
    fig.tight_layout()
    return _save(fig, Path(out_png))


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape[2] != 3:
        raise RuntimeError(f"{path}: expected RGB, found {array.shape}")
    return array


def renders_grid(
    render_dirs: dict[str, str | Path],
    gt_dir: str | Path,
    frames: list[int],
    out_png: str | Path,
) -> dict:
    """3 sample frames x ``GT | 3DGS | 2DGS`` with no interpolation."""
    gt_root = Path(gt_dir)
    fig, axes = plt.subplots(
        len(frames), 3, figsize=(12, 3.0 * len(frames)), squeeze=False
    )
    titles = ("GT", "3DGS", "2DGS")
    for row, frame in enumerate(frames):
        frame_int = int(frame)
        images = [
            _read_rgb(gt_root / f"{frame_int:010d}.png"),
            _read_rgb(Path(render_dirs["3dgs"]) / f"{frame_int:010d}_rgb.png"),
            _read_rgb(Path(render_dirs["2dgs"]) / f"{frame_int:010d}_rgb.png"),
        ]
        for column, (title, image) in enumerate(zip(titles, images)):
            axis = axes[row][column]
            axis.imshow(image, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{title} — frame {frame_int}")
    fig.tight_layout()
    return _save(fig, Path(out_png))


def _event_file(run_dir: Path) -> Path:
    manifest = run_dir / "run_manifest.json"
    if manifest.is_file():
        if bool(read_json(manifest).get("smoke")):
            raise RuntimeError(f"{run_dir}: smoke runs are not plotted")
    if "smoke" in run_dir.name:
        raise RuntimeError(f"{run_dir}: smoke runs are not plotted")
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        raise RuntimeError(f"{run_dir}: no events.out.tfevents.* file")
    return events[0]


def _series(event_file: Path, tags: tuple[str, ...]) -> tuple[str, list[int], list[float]]:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    for tag in tags:
        if tag in available:
            scalars = accumulator.Scalars(tag)
            return tag, [int(step.step) for step in scalars], [float(step.value) for step in scalars]
    raise RuntimeError(
        f"{event_file}: none of the required tags {tags} exist; found {sorted(available)}"
    )


def training_curves(run_dirs: dict[str, str | Path], out_png: str | Path) -> dict:
    """Plot official TensorBoard total and L1 losses for both methods."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"3dgs": "#1f77b4", "2dgs": "#d62728"}
    for method in METHODS:
        if method not in run_dirs:
            raise ValueError(f"training_curves is missing run directory for {method}")
        event_file = _event_file(Path(run_dirs[method]))
        _, total_steps, total_values = _series(event_file, (TOTAL_LOSS_TAG,))
        _, l1_steps, l1_values = _series(event_file, L1_LOSS_TAGS)
        axes[0].plot(total_steps, total_values, label=method.upper(), color=colors[method], linewidth=1.2)
        axes[1].plot(l1_steps, l1_values, label=method.upper(), color=colors[method], linewidth=1.2)
    axes[0].set_ylabel("total_loss")
    axes[1].set_ylabel("l1_loss")
    axes[1].set_xlabel("training iteration")
    axes[0].set_title("Official TensorBoard training curves")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    return _save(fig, Path(out_png))


def _run_dir(cfg: dict, method: str) -> Path:
    return Path(cfg["paths"]["artifacts_root"]) / "runs" / f"{method}-{cfg['run_id']}"


def make_all_figures(cfg: dict, comparison: dict, out_dir: str | Path) -> dict:
    """Generate every report figure and return ``{name: {path, sha256}}``."""
    from . import probe  # noqa: PLC0415 - avoid an import cycle at module load

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    render_dirs = {method: probe.locate_render_dir(cfg, method) for method in METHODS}
    frames = [int(frame) for frame in cfg["frames"]]
    if len(frames) < 3:
        raise RuntimeError("renders_grid needs at least three frozen frames")
    sample_frames = [frames[0], frames[len(frames) // 2], frames[-1]]
    run_dirs = {method: _run_dir(cfg, method) for method in METHODS}
    gt_dir = Path(cfg["paths"]["bundle"]) / "raw" / "images" / "image_00"

    figures = {
        "psnr_ssim_per_frame": psnr_ssim_per_frame(
            comparison, root / "psnr_ssim_per_frame.png"
        ),
        "error_by_bucket": error_by_bucket(comparison, root / "error_by_bucket.png"),
        "renders_grid": renders_grid(
            render_dirs, gt_dir, sample_frames, root / "renders_grid.png"
        ),
        "training_curves": training_curves(run_dirs, root / "training_curves.png"),
    }
    return figures


__all__ = [
    "error_by_bucket",
    "make_all_figures",
    "psnr_ssim_per_frame",
    "renders_grid",
    "training_curves",
]
