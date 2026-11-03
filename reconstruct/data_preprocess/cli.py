"""Command-line entry point for the Python preprocessing pipeline.

Stages are intentionally small and resumable::

    python -m reconstruct.data_preprocess frames
    python -m reconstruct.data_preprocess mask
    python -m reconstruct.data_preprocess gt
    python -m reconstruct.data_preprocess rays
    python -m reconstruct.data_preprocess gt-rays
    python -m reconstruct.data_preprocess bundle
    python -m reconstruct.data_preprocess all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .bbox import load_dynamic_bboxes
from .calib import (
    Kitti360Calib,
    compose_T_Ck_rect_V,
    compose_T_W_V,
    load_kitti360_calib,
)
from .config import Phase2Config, load_config
from .export import assemble_bundle, export_gt_rays, write_calibration_geometry
from .gt_cache import GtPoint, build_gt_manifest, write_gt_frame
from .io_utils import PreprocessError, canonical_sha256, ensure_dir, sha256_file, utc_now, write_json
from .layout import (
    common_frames,
    contiguous_runs,
    image_path,
    lidar_path,
    lidar_timestamps_path,
    load_poses,
    load_timestamps,
    poses_path,
    semantic_path,
)
from .mask import (
    KEPT_STATIC,
    build_mask,
    build_mask_manifest,
    write_mask_artifact,
)
from .projection import assign_primary_camera_batch
from .rays import (
    build_rays,
    build_rays_manifest,
    compute_occlusion_flags,
    freeze_ray_set,
    write_ray_frame,
)
from .semantics import load_semantics
from .uncurl import uncurl_scan
from .velodyne import load_scan

def _selected_frames(config: Phase2Config, override: str | None) -> list[int]:
    if override:
        frames = sorted({int(token) for token in override.split(",") if token.strip()})
        if not frames:
            raise PreprocessError("--frames supplied no frame ids")
        return frames
    if config.frames:
        return sorted(config.frames)
    manifest = config.phase_dir / "segment" / "segment_manifest.json"
    if manifest.is_file():
        document = __import__("json").loads(manifest.read_text(encoding="utf-8"))
        selected = document.get("selected") or {}
        frames = selected.get("frames")
        if frames:
            return [int(v) for v in frames]
    poses = load_poses(poses_path(config.dataset_root, config.sequence))
    available = common_frames(config.dataset_root, config.sequence, config.cameras, poses, lidar_root=config.lidar_root)
    runs = [run for run in contiguous_runs(available) if len(run) >= config.selection.min_usable_frames]
    if runs:
        return runs[0][: config.selection.window_frames]
    raise PreprocessError(
        "cannot determine the frame window: set config.frames, provide a "
        "segment manifest, or make the raw dataset locally available"
    )


def _write_segment_manifest(config: Phase2Config, frames: list[int]) -> dict:
    segment_dir = ensure_dir(config.phase_dir / "segment")
    document = {
        "schema_version": 1,
        "sequence": config.sequence,
        "selection_mode": "full",
        "generated_at_utc": utc_now(),
        "criteria": {
            "window_frames": config.selection.window_frames,
            "step_frames": config.selection.step_frames,
            "min_usable_frames": config.selection.min_usable_frames,
            "min_scene_points_per_frame": config.selection.min_scene_points_per_frame,
        },
        "selected": {
            "start_frame": frames[0],
            "end_frame": frames[-1],
            "frame_count": len(frames),
            "frames": frames,
        },
        "per_frame": [{"frame": frame} for frame in frames],
    }
    write_json(segment_dir / "segment_manifest.json", document)
    return document


def _frame_context(config: Phase2Config) -> tuple[Kitti360Calib, dict, list[int]]:
    calib = load_kitti360_calib(config.dataset_root / "calibration")
    poses = load_poses(poses_path(config.dataset_root, config.sequence))
    timestamps = load_timestamps(lidar_timestamps_path(config.lidar_root))
    return calib, poses, timestamps


def stage_frames(config: Phase2Config, frames: list[int]) -> None:
    for frame in frames:
        if not lidar_path(config.lidar_root, frame).is_file():
            raise PreprocessError(f"missing LiDAR scan for frame {frame}")
        for camera in config.cameras:
            if not image_path(config.dataset_root, config.sequence, camera, frame).is_file():
                raise PreprocessError(f"missing {camera} image for frame {frame}")
            if not semantic_path(config.dataset_root, config.sequence, camera, frame).is_file():
                raise PreprocessError(f"missing {camera} semantics for frame {frame}")
    _write_segment_manifest(config, frames)


def stage_mask(config: Phase2Config, frames: list[int]) -> None:
    calib, poses, timestamps = _frame_context(config)
    boxes = load_dynamic_bboxes(config.bboxes_xml) if config.bboxes_xml.is_file() else []
    masks_dir = ensure_dir(config.phase_dir / "masks")
    entries: list[dict] = []
    for frame in frames:
        pose = poses.get(frame)
        next_pose = poses.get(frame + 1)
        if pose is None or next_pose is None:
            raise PreprocessError(f"frame {frame} has no exact/current+next pose")
        if frame + 1 >= len(timestamps):
            raise PreprocessError(f"frame {frame} has no successor timestamp")
        scan = load_scan(lidar_path(config.lidar_root, frame))
        T_W_V = compose_T_W_V(calib, pose)
        uncurled = uncurl_scan(
            scan[:, :3],
            T_W_V,
            compose_T_W_V(calib, next_pose),
            timestamps[frame],
            timestamps[frame + 1],
        )
        transforms = {camera: compose_T_Ck_rect_V(calib, pose, camera) for camera in config.cameras}
        semantics = {
            camera: load_semantics(semantic_path(config.dataset_root, config.sequence, camera, frame))
            for camera in config.cameras
        }
        image_sizes = {camera: (semantics[camera].width, semantics[camera].height) for camera in config.cameras}
        intrinsics = {camera: calib.K_rect[camera] for camera in config.cameras}
        result = build_mask(
            scan,
            uncurled,
            frame,
            T_W_V,
            boxes,
            semantics,
            image_sizes,
            transforms,
            intrinsics,
            config.mask,
        )
        document = write_mask_artifact(masks_dir, frame, result, scan, config.sequence)
        entries.append(document)
        print(f"[mask] frame={frame} kept_static={document['counts']['kept_static']}")
    build_mask_manifest(masks_dir, entries, config.sequence, config.config_sha256)


def _load_mask_bits(config: Phase2Config, frame: int, count: int) -> np.ndarray:
    path = config.phase_dir / "masks" / f"frame_{frame:010d}" / "points.bin"
    if not path.is_file():
        raise PreprocessError(f"missing mask artifact {path}; run the mask stage first")
    bits = np.fromfile(path, dtype=np.uint8)
    if bits.size != count:
        raise PreprocessError(f"mask point count does not match the raw scan for frame {frame}")
    return bits


def stage_gt(config: Phase2Config, frames: list[int]) -> None:
    calib, poses, _timestamps = _frame_context(config)
    gt_dir = ensure_dir(config.phase_dir / "gt")
    entries: list[dict] = []
    for frame in frames:
        pose = poses.get(frame)
        if pose is None:
            raise PreprocessError(f"frame {frame} has no exact pose")
        scan = load_scan(lidar_path(config.lidar_root, frame))
        bits = _load_mask_bits(config, frame, scan.shape[0])
        transforms = {camera: compose_T_Ck_rect_V(calib, pose, camera) for camera in config.cameras}
        intrinsics = {camera: calib.K_rect[camera] for camera in config.cameras}
        semantics = {
            camera: load_semantics(semantic_path(config.dataset_root, config.sequence, camera, frame))
            for camera in config.cameras
        }
        image_sizes = {camera: (semantics[camera].width, semantics[camera].height) for camera in config.cameras}

        # Recompute the uncurled points exactly as the mask stage did.
        next_pose = poses.get(frame + 1)
        timestamps = load_timestamps(lidar_timestamps_path(config.lidar_root))
        if next_pose is None or frame + 1 >= len(timestamps):
            raise PreprocessError(f"frame {frame} has no successor pose/timestamp for uncurl")
        T_W_V = compose_T_W_V(calib, pose)
        uncurled = uncurl_scan(
            scan[:, :3],
            T_W_V,
            compose_T_W_V(calib, next_pose),
            timestamps[frame],
            timestamps[frame + 1],
        )

        kept = np.flatnonzero(bits & KEPT_STATIC)
        if kept.size and not np.isfinite(uncurled[kept]).all():
            raise PreprocessError(f"kept-static point is not finite for frame {frame}")
        point_l = uncurled[kept]
        cameras_idx, r_gt, u, v = assign_primary_camera_batch(
            point_l,
            transforms,
            intrinsics,
            image_sizes,
            r_min_m=config.rays.r_min_m,
            r_max_m=config.rays.r_max_m,
            margin_px=config.rays.image_margin_px,
        )
        labels = np.full(kept.size, 65535, dtype=np.uint16)
        for camera_index in (0, 1):
            mask = cameras_idx == camera_index
            if not mask.any():
                continue
            camera = config.cameras[camera_index]
            semantic_image = semantics[camera]
            px = np.floor(u[mask] + 0.5).astype(np.int64)
            py = np.floor(v[mask] + 0.5).astype(np.int64)
            inside = (
                np.isfinite(u[mask])
                & np.isfinite(v[mask])
                & (px >= 0)
                & (py >= 0)
                & (px < semantic_image.width)
                & (py < semantic_image.height)
            )
            gathered = np.full(int(mask.sum()), 65535, dtype=np.uint16)
            gathered[inside] = semantic_image.labels[py[inside], px[inside]]
            labels[mask] = gathered
        camera_counts = {
            0: int((cameras_idx == 0).sum()),
            1: int((cameras_idx == 1).sum()),
            255: int((cameras_idx == 255).sum()),
        }
        points = [
            GtPoint(
                p_L=(float(point_l[index][0]), float(point_l[index][1]), float(point_l[index][2])),
                frame=frame,
                semantic_id=int(labels[index]),
                camera=int(cameras_idx[index]),
                valid=1,
                r_gt=float(r_gt[index]),
            )
            for index in range(kept.size)
        ]
        count = len(points)
        if count < config.gt_cache.min_points_per_frame:
            raise PreprocessError(f"frame {frame} has only {count} GT points")
        path = gt_dir / f"frame_{frame:010d}.bin"
        write_gt_frame(path, frame, points)
        entries.append(
            {
                "frame": frame,
                "count": count,
                "camera_00": camera_counts[0],
                "camera_01": camera_counts[1],
                "camera_unknown": camera_counts[255],
                "sha256": sha256_file(path),
                "config_sha256": config.config_sha256,
            }
        )
        print(f"[gt] frame={frame} kept={count} cam0={camera_counts[0]} cam1={camera_counts[1]}")
    build_gt_manifest(gt_dir, config.sequence, entries, config.config_sha256)


def stage_rays(config: Phase2Config, frames: list[int]) -> None:
    from .gt_cache import read_gt_frame

    calib, poses, _timestamps = _frame_context(config)
    rays_dir = ensure_dir(config.phase_dir / "rays")
    entries: list[dict] = []
    for frame in frames:
        pose = poses.get(frame)
        if pose is None:
            raise PreprocessError(f"frame {frame} has no exact pose")
        _header, gt_points = read_gt_frame(config.phase_dir / "gt" / f"frame_{frame:010d}.bin")
        transforms = {camera: compose_T_Ck_rect_V(calib, pose, camera) for camera in config.cameras}
        rays = build_rays(gt_points, frame, transforms)

        for camera_index, camera in enumerate(config.cameras):
            size = (1408, 376)
            flags = compute_occlusion_flags(
                gt_points,
                rays,
                camera_index,
                transforms[camera],
                calib.K_rect[camera],
                size[0],
                size[1],
                neighbor_px=config.rays.occlusion_neighbor_px,
                sigma_lidar_m=config.rays.occlusion_sigma_lidar_m,
                sigma_multiple=config.rays.occlusion_sigma_multiple,
                depth_threshold_m=config.rays.occlusion_depth_threshold_m,
            )
            for ray, flag in zip(rays, flags):
                if ray.camera == camera_index:
                    ray.occluded |= int(flag)

        ray_path = write_ray_frame(rays_dir / f"frame_{frame:010d}.bin", frame, rays)
        primary = np.array([1 if ray.bucket < 3 else 0 for ray in rays], dtype=np.uint8)
        secondary = np.array([1 if ray.bucket == 3 else 0 for ray in rays], dtype=np.uint8)
        unoccluded = np.array(
            [1 if (ray.bucket < 3 and not ray.occluded) else 0 for ray in rays], dtype=np.uint8
        )
        primary_path = rays_dir / f"primary_mask_{frame:010d}.bin"
        secondary_path = rays_dir / f"secondary_mask_{frame:010d}.bin"
        unoccluded_path = rays_dir / f"primary_unoccluded_mask_{frame:010d}.bin"
        primary.tofile(primary_path)
        secondary.tofile(secondary_path)
        unoccluded.tofile(unoccluded_path)

        buckets = [0, 0, 0, 0]
        cameras = {"image_00": 0, "image_01": 0}
        occluded_count = 0
        for ray in rays:
            buckets[ray.bucket] += 1
            cameras[config.cameras[ray.camera]] += 1
            occluded_count += int(bool(ray.occluded))
        entries.append(
            {
                "frame": frame,
                "files": {
                    "ray": ray_path.name,
                    "primary_mask": primary_path.name,
                    "secondary_mask": secondary_path.name,
                    "primary_unoccluded_mask": unoccluded_path.name,
                },
                "sha256": {
                    "ray": sha256_file(ray_path),
                    "primary_mask": sha256_file(primary_path),
                    "secondary_mask": sha256_file(secondary_path),
                    "primary_unoccluded_mask": sha256_file(unoccluded_path),
                },
                "n_rays": len(rays),
                "full_count": len(rays),
                "camera_counts": cameras,
                "bucket_counts": {
                    "0_20": buckets[0],
                    "20_40": buckets[1],
                    "40_50": buckets[2],
                    "50_60": buckets[3],
                },
                "occluded_count": occluded_count,
                "occluded_excluded_count": int(unoccluded.sum()),
            }
        )
        print(f"[rays] frame={frame} rays={len(rays)} occluded={occluded_count}")

    manifest = build_rays_manifest(rays_dir, entries, frames, config.sequence, config.config_sha256)
    freeze_ray_set(rays_dir, manifest, config.config_sha256)


def stage_gt_rays(config: Phase2Config, frames: list[int]) -> None:
    bundle_geometry = config.phase_dir.parent.parent / "exports"
    # The export stage writes geometry into the bundle; use the bundle geometry
    # when present, otherwise regenerate it from the raw calibration.
    candidate = _bundle_root(config) / "geometry" / "cameras.json"
    if not candidate.is_file():
        calib, poses, _ = _frame_context(config)
        write_calibration_geometry(calib, poses, frames, config.sequence, _bundle_root(config) / "geometry")
    export_gt_rays(
        _bundle_root(config) / "geometry" / "cameras.json",
        config.phase_dir / "rays" / "rays_manifest.json",
        config.phase_dir / "gt" / "gt_manifest.json",
        config.phase_dir / "rays",
        _bundle_root(config) / "derived" / "gt_rays",
    )


def _bundle_root(config: Phase2Config) -> Path:
    return config.bundle_dir


def stage_bundle(config: Phase2Config, frames: list[int]) -> None:
    out_dir = _bundle_root(config)
    info = assemble_bundle(config, frames, out_dir)
    # Export gt_rays.bin into the bundle once geometry exists.
    export_gt_rays(
        out_dir / "geometry" / "cameras.json",
        config.phase_dir / "rays" / "rays_manifest.json",
        config.phase_dir / "gt" / "gt_manifest.json",
        config.phase_dir / "rays",
        out_dir / "derived" / "gt_rays",
    )
    print(f"[bundle] wrote {info['files']} files to {info['root']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reconstruct.data_preprocess")
    parser.add_argument("stage", choices=["frames", "mask", "gt", "rays", "gt-rays", "bundle", "all"])
    parser.add_argument("--config", default=None, help="Preprocessing YAML (default: reconstruct/configs/preprocess_1820.yaml)")
    parser.add_argument("--frames", default=None, help="Comma-separated frame ids override")
    parser.add_argument("--phase-dir", default=None, help="Override the Phase-2 artifacts directory")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.phase_dir:
        from dataclasses import replace

        config = replace(config, phase_dir=Path(args.phase_dir))
    frames = _selected_frames(config, args.frames)

    if args.stage in ("frames", "all"):
        stage_frames(config, frames)
    if args.stage in ("mask", "all"):
        stage_mask(config, frames)
    if args.stage in ("gt", "all"):
        stage_gt(config, frames)
    if args.stage in ("rays", "all"):
        stage_rays(config, frames)
    if args.stage in ("gt-rays",):
        stage_gt_rays(config, frames)
    if args.stage in ("bundle", "all"):
        stage_bundle(config, frames)
    return 0
