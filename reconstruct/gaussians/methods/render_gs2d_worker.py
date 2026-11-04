"""Render RGB, accumulated alpha, and z-depth from a trained official 2DGS model.

Run with ``cwd`` set to the pinned ``2d-gaussian-splatting`` checkout (and the
repository on ``PYTHONPATH``) so the official ``scene`` and
``gaussian_renderer`` packages are importable.

The official 2DGS renderer already returns both maps Phase 5 needs:

* ``surf_depth`` — with ``depth_ratio = 0.0`` this is the expected **z-depth**
  ``sum(z * alpha * T) / sum(alpha * T)``;
* ``rend_alpha`` — accumulated opacity ``1 - T_final``.

No extra pass and no CUDA patch is required.  The manifest is self-describing
so the shared hit builder can verify the conventions before sampling.
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
from scene import GaussianModel, Scene  # noqa: E402

from reconstruct.gaussians.methods.gs2d import (  # noqa: E402
    ALPHA_SOURCE,
    DEPTH_CONVENTION,
    DEPTH_RATIO,
    DEPTH_SOURCE,
    EXPECTED_RESOLUTION,
    verify_upstream,
)
from reconstruct.gaussians.run_utils import sha256_file, utc_now  # noqa: E402


def _build_parser() -> tuple[ArgumentParser, ModelParams, PipelineParams]:
    parser = ArgumentParser(prog="render_gs2d_worker.py")
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


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(array), array, np.float32(0.0)).astype(np.float32)


def _sanitize_depth(values: np.ndarray) -> np.ndarray:
    """Return float32 z-depth with non-finite values replaced by 0."""
    return _finite_or_zero(values)


def _sanitize_alpha(values: np.ndarray) -> np.ndarray:
    """Return float32 accumulated alpha, non-finite -> 0, clamped to [0, 1]."""
    return np.clip(_finite_or_zero(values), 0.0, 1.0).astype(np.float32)


def _run_id_from_model_path(model_path: str) -> str:
    name = Path(model_path).name
    return name[len("2dgs-") :] if name.startswith("2dgs-") else name


def main(argv: list[str] | None = None) -> int:
    parser, model_params, pipeline_params = _build_parser()
    args = parser.parse_args(argv)
    if args.white_background:
        raise RuntimeError("the frozen Phase-5 protocol requires a black background")
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    if dataset.eval:
        raise RuntimeError("the frozen Phase-5 protocol has no evaluation split")
    if float(pipeline.depth_ratio) != float(DEPTH_RATIO):
        raise RuntimeError(
            "the frozen Phase-5 protocol requires depth_ratio=0.0 so surf_depth "
            f"is z-depth; got {pipeline.depth_ratio}"
        )

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
            package = render(camera, gaussians, pipeline, background)
            rgb = package["render"]
            if tuple(rgb.shape[-2:]) != (height, width):
                raise RuntimeError(
                    f"frame {frame}: RGB shape {tuple(rgb.shape)} is not "
                    f"{height}x{width}"
                )

            depth = _sanitize_depth(
                package["surf_depth"][0].detach().cpu().numpy()
            )
            alpha = _sanitize_alpha(
                package["rend_alpha"][0].detach().cpu().numpy()
            )
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
                    "depth_shape": list(depth.shape),
                    "alpha_shape": list(alpha.shape),
                    "alpha_min": float(alpha.min()),
                    "alpha_max": float(alpha.max()),
                    "z_valid_fraction": float(np.mean(depth > 0.0)),
                }
            )
            print(f"rendered frame {frame}")

    manifest = {
        "schema_version": 1,
        "method": "2dgs",
        "run_id": _run_id_from_model_path(dataset.model_path),
        "iteration": int(args.iteration),
        "resolution": [width, height],
        "background": [0, 0, 0],
        "alpha_source": ALPHA_SOURCE,
        "depth_source": DEPTH_SOURCE,
        "depth_ratio": float(DEPTH_RATIO),
        "depth_convention": DEPTH_CONVENTION,
        "upstream_head": provenance["upstream_head"],
        "submodel_head": provenance["submodel_head"],
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
