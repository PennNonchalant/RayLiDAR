"""Load a RayLiDAR .rlf frame into PyTorch and draw a publication-style figure.

    python python/plot_frame.py build/frames/hybrid_4643.rlf [out.png]

a: range image (64 beams x 2048 azimuth steps).
b: which representation each ray hit, split by the GT-free trust label.
c: top view of the sweep in the sensor frame.
d: share of each class per beam elevation.
Writes a PNG (300 dpi) and a vector PDF next to it. Also builds a (3, 64, 2048) tensor
(range, source, trust), the usual input layout for range-image networks.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raylidar_frame import load_frame, to_torch  # noqa: E402

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
AXIS = "#898781"
GRID = "#e1e0d9"
MISS = "#e9e8e3"
SURFACE = "#ffffff"

# Point classes. Colorblind-safe as a set (validated); fixed order, never cycled.
CLASSES = [
    ("Mesh, trusted", "#2a78d6"),
    ("Mesh, untrusted", "#eb6834"),
    ("2D Gaussian", "#1baf7a"),
    ("No hit", MISS),
]

# One-hue sequential ramp for range: light = near, dark = far.
RANGE_CMAP = LinearSegmentedColormap.from_list(
    "range", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
RANGE_CMAP.set_bad(MISS)

ELEVATION = (-24.8, 2.0)  # HDL-64E, beam 0 is the lowest


def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def point_classes(frame):
    """Per ray: 0 mesh trusted (or unchecked), 1 mesh untrusted, 2 Gaussian, 3 no hit."""
    classes = np.full(frame["range"].shape, 3, dtype=np.int8)
    mesh = frame["source"] == 1
    classes[mesh] = 0
    classes[mesh & (frame["trust"] == 2)] = 1
    classes[frame["source"] == 2] = 2
    return classes


def to_tensor(frame):
    """(3, beams, azimuths) float tensor: range in metres (0 = miss), source id, trust id."""
    import torch

    t = to_torch(frame)
    rng = torch.where(t["hit"], t["range"], torch.zeros_like(t["range"]))
    return torch.stack([rng, t["source"].float(), t["trust"].float()])


def panel(ax, letter, title):
    ax.text(0, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
    ax.text(0.012, 1.06, title, transform=ax.transAxes, fontsize=7, color=INK_SECONDARY, va="bottom", ha="left")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_suffix(".png")
    frame = load_frame(path)
    hit = frame["hit"]

    try:
        x = to_tensor(frame)
        print(f"PyTorch tensor: shape {tuple(x.shape)}, dtype {x.dtype}")
    except ImportError:
        print("PyTorch not installed, skipping the tensor (the figure still works).")

    set_style()
    beams, _ = frame["range"].shape
    extent = [0, 360, *ELEVATION]
    classes = point_classes(frame)
    shares = [(classes == k).mean() * 100 for k in range(4)]

    fig = plt.figure(figsize=(7.2, 6.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.7], hspace=0.45, left=0.075, right=0.94, top=0.93,
                             bottom=0.07)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 0.015], wspace=0.02)
    mid = outer[1].subgridspec(1, 2, width_ratios=[1, 0.015], wspace=0.02)
    bottom = outer[2].subgridspec(1, 2, width_ratios=[2.35, 1], wspace=0.2)

    # Shared legend for the point classes (b, c, d).
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=5.5, markeredgewidth=0, color=color,
               label=f"{name} ({share:.1f}%)")
        for (name, color), share in zip(CLASSES, shares)
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               handletextpad=0.25, columnspacing=1.8)

    # a: range image
    ax = fig.add_subplot(top[0])
    image = ax.imshow(np.where(hit, frame["range"], np.nan), origin="lower", aspect="auto", extent=extent,
                      cmap=RANGE_CMAP, norm=LogNorm(2, 80), interpolation="nearest", rasterized=True)
    ax.set_ylabel("Elevation (°)")
    ax.set_xticks(range(0, 361, 45))
    ax.set_yticks([-20, -10, 0])
    panel(ax, "a", f"Simulated range image, KITTI-360 frame {frame['frame']} (64 beams × 2,048 steps)")
    cax = fig.add_subplot(top[1])
    bar = fig.colorbar(image, cax=cax, extend="both")
    bar.set_ticks([2, 5, 10, 20, 50], labels=["2", "5", "10", "20", "50"])
    bar.ax.minorticks_off()
    bar.set_label("Range (m)", labelpad=2)
    bar.outline.set_linewidth(0.5)
    bar.ax.tick_params(width=0.5, length=2)

    # b: point classes on the same grid
    ax = fig.add_subplot(mid[0])
    ax.imshow(classes, origin="lower", aspect="auto", extent=extent,
              cmap=ListedColormap([c for _, c in CLASSES]), vmin=-0.5, vmax=3.5, interpolation="nearest",
              rasterized=True)
    ax.set_ylabel("Elevation (°)")
    ax.set_xlabel("Azimuth (°)")
    ax.set_xticks(range(0, 361, 45))
    ax.set_yticks([-20, -10, 0])
    panel(ax, "b", "Hybrid source and trust label per ray (computed without ground truth)")

    # c: top view; rarer classes drawn last so they stay visible
    ax = fig.add_subplot(bottom[0])
    pts = frame["points"][hit]
    cls = classes[hit]
    for k in (0, 2, 1):
        sel = cls == k
        ax.scatter(pts[sel, 0], pts[sel, 1], s=0.18, color=CLASSES[k][1], linewidths=0, rasterized=True)
    ax.plot(0, 0, marker="^", color=INK, markersize=4.5, linestyle="", markeredgewidth=0)
    ax.annotate("LiDAR", (0, 0), xytext=(0, 5), textcoords="offset points", fontsize=6, color=INK, ha="center",
                va="bottom")
    low, high = np.percentile(pts[:, :2], [0.5, 99.5], axis=0)
    ax.set_xlim(low[0] - 2, high[0] + 2)
    ax.set_ylim(low[1] - 2, high[1] + 2)
    ax.set_aspect("equal")
    ax.set_anchor("NW")
    ax.set_xlabel("x, forward (m)")
    ax.set_ylabel("y, left (m)")
    panel(ax, "c", "Top view of the sweep, sensor frame")

    # d: class share per beam, as a stacked profile over elevation
    ax = fig.add_subplot(bottom[1])
    elevations = np.linspace(*ELEVATION, beams)
    per_beam = np.stack([(classes == k).mean(axis=1) * 100 for k in range(4)])
    left = np.zeros(beams)
    for k, (_, color) in enumerate(CLASSES):
        ax.fill_betweenx(elevations, left, left + per_beam[k], color=color, linewidth=0)
        left += per_beam[k]
        if k < 3:
            ax.plot(left, elevations, color=SURFACE, linewidth=0.8)  # 2 px gap between fills
    ax.set_xlim(0, 100)
    ax.set_ylim(*ELEVATION)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([-20, -10, 0])
    ax.set_xlabel("Share of rays (%)")
    ax.set_ylabel("Elevation (°)")
    panel(ax, "d", "Composition by beam")

    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
