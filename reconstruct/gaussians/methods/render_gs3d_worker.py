"""Render RGB, accumulated alpha, and expected z-depth from a trained 3DGS model.

Run with ``cwd`` set to the pinned ``gaussian-splatting`` checkout (and the
repository on ``PYTHONPATH``) so the official ``scene`` and
``gaussian_renderer`` packages are importable.

Three official-renderer passes per frame, no CUDA patch:

1. RGB pass: ``render(..., override_color=None)`` with a black background.
2. Alpha pass: ``render(..., override_color=ones)``; the rendered RGB is
   ``sum(alpha_i * T_i)`` in every channel, so ``alpha = mean(R, G, B)``.
3. Expected-z pass: ``render(..., override_color=z_color)`` where
   ``z_color = z_view / 100`` for every Gaussian.  The rendered RGB is
   ``sum((z / 100) * alpha_i * T_i)``; dividing by ``alpha`` and multiplying by
   100 gives ``E[z] = sum(z * alpha * T) / sum(alpha * T)``.

The ``/100`` scale keeps every colour inside the renderer's final
``clamp(0, 1)`` (the frozen window is at most 60 m), so the depth pass cannot be
clipped.  The rasterizer's own ``invdepths`` is a harmonic inverse-depth
accumulator (``sum((1 / z) * alpha * T)``) and is deliberately not exported.

``override_color`` requires ``separate_sh=False``: the upstream
``separate_sh=True`` branch forwards both ``shs`` and ``colors_precomp`` and
raises in the rasterizer.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402

from reconstruct.gaussians.methods.gs3d import (  # noqa: E402
    ALPHA_SOURCE,
    DEPTH_CONVENTION,
    DEPTH_SOURCE,
    EXPECTED_RESOLUTION,
    EXPECTED_Z_SCALE,
    alpha_from_white_render,
    expected_z_from_render,
    verify_upstream,
    view_space_z,
)
from reconstruct.gaussians.run_utils import sha256_file, utc_now  # noqa: E402


def _build_parser() -> tuple[ArgumentParser, ModelParams, PipelineParams]:
    parser = ArgumentParser(prog="render_gs3d_worker.py")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--out", required=True, help="render output directory")
    parser.add_argument(
        "--frames",
        required=True,
        help="comma-separated frame ids, e.g. 1820,1821",
    )
    return parser, model_params, pipeline_params


def _parse_frames(value: str) -> list[int]:
    frames = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        frames.append(int(token))
    if not frames:
        raise ValueError("--frames must contain at least one frame id")
    if len(set(frames)) != len(frames):
        raise ValueError("--frames must not contain duplicates")
    return sorted(frames)


def _frame_of(camera) -> int | None:
    try:
        return int(Path(camera.image_name).stem)
    except (TypeError, ValueError):
        return None


def _save_rgb(rgb: torch.Tensor, path: Path) -> None:
    array = (
        rgb.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).add(0.5).cpu().numpy()
    )
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _run_id_from_model_path(model_path: str) -> str:
    name = Path(model_path).name
    return name[len("3dgs-") :] if name.startswith("3dgs-") else name


def main(argv: list[str] | None = None) -> int:
    parser, model_params, pipeline_params = _build_parser()
    args = parser.parse_args(argv)
    if args.white_background:
        raise RuntimeError("the frozen Phase-5 protocol requires a black background")
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    if dataset.eval:
        raise RuntimeError("the frozen Phase-5 protocol has no evaluation split")

    frames = _parse_frames(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = verify_upstream(Path.cwd())
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=int(args.iteration), shuffle=False)
    if len(scene.getTestCameras()) != 0:
        raise RuntimeError("unexpected evaluation cameras; all 60 frames are training views")

    selected = []
    for camera in scene.getTrainCameras():
        frame = _frame_of(camera)
        if frame is not None and frame in set(frames):
            selected.append((frame, camera))
    selected.sort(key=lambda item: item[0])
    if [frame for frame, _ in selected] != frames:
        raise RuntimeError(
            "requested frames do not match the training cameras: "
            f"requested {frames}, found {[frame for frame, _ in selected]}"
        )

    width, height = EXPECTED_RESOLUTION
    background = torch.zeros(3, device="cuda")
    point_cloud = (
        Path(dataset.model_path)
        / "point_cloud"
        / f"iteration_{int(args.iteration)}"
        / "point_cloud.ply"
    )
    if not point_cloud.is_file():
        raise RuntimeError(f"point cloud missing: {point_cloud}")

    per_frame = []
    with torch.no_grad():
        for frame, camera in selected:
            package = render(
                camera,
                gaussians,
                pipeline,
                background,
                separate_sh=False,
                use_trained_exp=False,
            )
            rgb = package["render"]
            if tuple(rgb.shape[-2:]) != (height, width):
                raise RuntimeError(
                    f"frame {frame}: RGB shape {tuple(rgb.shape)} is not "
                    f"{height}x{width}"
                )

            white = torch.ones(
                (gaussians.get_xyz.shape[0], 3),
                dtype=torch.float32,
                device="cuda",
            )
            package_alpha = render(
                camera,
                gaussians,
                pipeline,
                background,
                override_color=white,
                separate_sh=False,
                use_trained_exp=False,
            )
            alpha = alpha_from_white_render(
                package_alpha["render"].detach().cpu().numpy()
            )

            # Expected z-depth via the official colour-override path: render
            # ``z_view / 100`` so the final clamp(0, 1) cannot clip a <= 60 m
            # scene, then divide by the accumulated alpha and rescale.
            xyz = gaussians.get_xyz.detach()
            xyz_h = torch.cat(
                [
                    xyz,
                    torch.ones(
                        (xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device
                    ),
                ],
                dim=1,
            )
            z_view = view_space_z(xyz_h, camera.world_view_transform)
            z_color = (z_view / EXPECTED_Z_SCALE)[:, None].repeat(1, 3)
            package_expected = render(
                camera,
                gaussians,
                pipeline,
                background,
                override_color=z_color,
                separate_sh=False,
                use_trained_exp=False,
            )
            z_render = (
                package_expected["render"][0].detach().cpu().numpy()
            )
            depth = expected_z_from_render(z_render, alpha, EXPECTED_Z_SCALE)
            if depth.shape != (height, width) or alpha.shape != (height, width):
                raise RuntimeError(
                    f"frame {frame}: depth/alpha shape "
                    f"{depth.shape}/{alpha.shape} != {(height, width)}"
                )

            rgb_path = out_dir / f"{frame:010d}_rgb.png"
            depth_path = out_dir / f"{frame:010d}_depth.npy"
            alpha_path = out_dir / f"{frame:010d}_alpha.npy"
            _save_rgb(rgb, rgb_path)
            np.save(depth_path, depth.astype(np.float32))
            np.save(alpha_path, alpha.astype(np.float32))

            per_frame.append(
                {
                    "frame": frame,
                    "rgb_sha256": sha256_file(rgb_path),
                    "depth_sha256": sha256_file(depth_path),
                    "alpha_sha256": sha256_file(alpha_path),
                    "alpha_min": float(alpha.min()),
                    "alpha_max": float(alpha.max()),
                    "z_valid_fraction": float(np.mean(depth > 0.0)),
                }
            )
            print(f"rendered frame {frame}")

    manifest = {
        "schema_version": 1,
        "method": "3dgs",
        "run_id": _run_id_from_model_path(dataset.model_path),
        "iteration": int(args.iteration),
        "resolution": [width, height],
        "background": [0, 0, 0],
        "separate_sh": False,
        "alpha_source": ALPHA_SOURCE,
        "depth_source": DEPTH_SOURCE,
        "depth_convention": DEPTH_CONVENTION,
        "depth_scale": EXPECTED_Z_SCALE,
        "upstream_head": provenance["upstream_head"],
        "point_cloud_sha256": sha256_file(point_cloud),
        "per_frame": per_frame,
        "generated_at_utc": utc_now(),
    }
    (out_dir / "render_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
