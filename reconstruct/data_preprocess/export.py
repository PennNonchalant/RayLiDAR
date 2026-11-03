"""Bundle assembly and ``gt_rays.bin`` export.

The handoff bundle is the only construction input for the mesh and Gaussian
stages: rectified images, semantics, LiDAR, calibration, per-frame world
transforms, and the frozen derived artifacts. ``gt_rays.bin`` is the W-frame
camera-origin ray file consumed by the evaluation harness.
"""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

import numpy as np

from .calib import (
    Kitti360Calib,
    compose_T_W_Ck_rect,
    load_kitti360_calib,
)
from .config import Phase2Config
from .io_utils import PreprocessError, ensure_dir, read_json, sha256_file, utc_now, write_json
from .layout import (
    cam0_to_world_path,
    image_path,
    image_timestamps_path,
    lidar_path,
    lidar_timestamps_path,
    load_poses,
    poses_path,
    semantic_path,
)
from .rays import (
    GT_RAY_RECORD_SIZE,
    GT_RAY_VERSION,
    R_MAX_M,
    T_MIN_M,
    read_ray_frame,
    write_gt_rays,
)

ROW_ORDER = ("r11", "r12", "r13", "tx", "r21", "r22", "r23", "ty", "r31", "r32", "r33", "tz")


def _matrix_columns(matrix: np.ndarray) -> list[float]:
    return [float(matrix[row, column]) for row in range(3) for column in range(4)]


def _select_lines(path: Path, keep: set[int]) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens and int(tokens[0]) in keep:
            rows.append(line.rstrip("\n"))
    found = {int(row.split()[0]) for row in rows}
    missing = keep - found
    if missing:
        raise PreprocessError(f"{path}: missing frame ids {sorted(missing)[:10]}")
    return rows


def _selected_timestamps(path: Path, frames: list[int]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for frame in frames:
        if frame >= len(lines) or not lines[frame].strip():
            raise PreprocessError(f"{path}: empty timestamp at index {frame}")
        rows.append(lines[frame].strip())
    return rows


def write_calibration_geometry(
    calib: Kitti360Calib,
    poses: dict[int, np.ndarray],
    frames: list[int],
    sequence: str,
    out_dir: str | Path,
) -> dict:
    geometry = ensure_dir(out_dir)
    calib_json = {
        "schema_version": 1,
        "K_rect_image_00": calib.K_rect["image_00"].tolist(),
        "K_rect_image_01": calib.K_rect["image_01"].tolist(),
        "P_rect_image_00": calib.P_rect["image_00"].tolist(),
        "P_rect_image_01": calib.P_rect["image_01"].tolist(),
        "R_rect_image_00": calib.R_rect["image_00"].tolist(),
        "R_rect_image_01": calib.R_rect["image_01"].tolist(),
        "T_I_image_00": calib.T_I_Ck["image_00"].tolist(),
        "T_I_image_01": calib.T_I_Ck["image_01"].tolist(),
        "T_V_C0": calib.T_V_C0.tolist(),
    }
    write_json(geometry / "calib.json", calib_json)

    rows: list[dict] = []
    camera_entries: list[dict] = []
    for frame in frames:
        pose = poses.get(frame)
        if pose is None:
            raise PreprocessError(f"frame {frame} has no pose")
        transforms = {camera: compose_T_W_Ck_rect(calib, pose, camera) for camera in calib.K_rect}
        row: dict[str, float | int] = {"frame": frame, "pose_frame": frame, "pose_delta": 0}
        for index, value in enumerate(_matrix_columns(pose)):
            row[f"T_W_I_{ROW_ORDER[index]}"] = value
        for camera, transform in transforms.items():
            for index, value in enumerate(_matrix_columns(transform)):
                row[f"{camera}_rect_{ROW_ORDER[index]}"] = value
        row["camera_height_m"] = float(transforms["image_00"][2, 3])
        rows.append(row)

        entry = {"frame": frame}
        for camera, transform in transforms.items():
            entry[f"T_W_{camera}_rect"] = transform.tolist()
            entry[f"T_{camera}_rect_W"] = np.linalg.inv(transform).tolist()
        camera_entries.append(entry)

    fieldnames = ["frame", "pose_frame", "pose_delta"]
    fieldnames += [f"T_W_I_{name}" for name in ROW_ORDER]
    for camera in calib.K_rect:
        fieldnames += [f"{camera}_rect_{name}" for name in ROW_ORDER]
    fieldnames.append("camera_height_m")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    (geometry / "frames.csv").write_text(buffer.getvalue(), encoding="utf-8")

    cameras = {
        "schema_version": 1,
        "sequence": sequence,
        "convention": "p_A = T_A_B * p_B; T_W_Ck_rect maps rectified camera points to the KITTI-360 world frame W (metres)",
        "world_frame": "KITTI-360 poses.txt world W",
        "units": "metre",
        "max_pose_delta_frames": 0,
        "cameras": {
            camera: {
                "model": "PINHOLE",
                "width": 1408,
                "height": 376,
                "K": calib.K_rect[camera].tolist(),
                "P_rect": calib.P_rect[camera].tolist(),
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
                "note": "data_rect is already rectified; use K_rect and no distortion",
            }
            for camera in calib.K_rect
        },
        "frames": camera_entries,
    }
    write_json(geometry / "cameras.json", cameras)
    return cameras


def export_gt_rays(
    cameras_json: str | Path,
    rays_manifest: str | Path,
    gt_manifest: str | Path,
    rays_dir: str | Path,
    out_dir: str | Path,
    *,
    camera: int = 0,
    expected_total: int | None = None,
) -> dict:
    if camera not in (0, 1):
        raise PreprocessError("camera must be 0 or 1")
    camera_doc = read_json(cameras_json)
    manifest = read_json(rays_manifest)
    gt_doc = read_json(gt_manifest)
    camera_frames = {int(entry["frame"]): np.asarray(entry["T_W_image_00_rect"], dtype=np.float64) for entry in camera_doc["frames"]}
    gt_counts = {int(entry["frame"]): int(entry["camera_00"]) for entry in gt_doc["frames"]}

    exported: list[dict] = []
    flag_records: list[bytes] = []
    per_frame: list[dict] = []
    bucket_totals = [0, 0, 0, 0]
    occluded_total = 0

    for entry in manifest["frames"]:
        frame = int(entry["frame"])
        if frame not in camera_frames:
            raise PreprocessError(f"cameras.json has no frame {frame}")
        ray_path = Path(rays_dir) / entry["files"]["ray"]
        expected_hash = entry["sha256"]["ray"]
        if sha256_file(ray_path) != expected_hash:
            raise PreprocessError(f"ray file hash drift for {ray_path}")
        header_frame, rays = read_ray_frame(ray_path)
        if header_frame != frame:
            raise PreprocessError(f"ray file header frame disagrees with the manifest for {ray_path}")

        transform = camera_frames[frame]
        rotation = transform[:3, :3]
        origin = transform[:3, 3]
        count = 0
        occluded_count = 0
        buckets = [0, 0, 0, 0]
        for ray in rays:
            if ray.camera != camera:
                continue
            t_gt = float(ray.t_gt)
            if not np.isfinite(t_gt) or t_gt < T_MIN_M or t_gt > R_MAX_M:
                raise PreprocessError(f"ray {ray.ray_id} has t_gt outside the frozen [1e-4, 60] m domain")
            raw = rotation @ np.asarray(ray.dir_C, dtype=np.float64)
            norm = float(np.linalg.norm(raw))
            if not np.isfinite(norm) or norm <= 0.0:
                raise PreprocessError(f"ray {ray.ray_id} produced a degenerate W-frame direction")
            direction = raw / norm
            exported.append(
                {
                    "ray_id": int(ray.ray_id),
                    "origin_W": origin.tolist(),
                    "dir_W": direction.tolist(),
                    "t_gt": t_gt,
                    "t_min": T_MIN_M,
                    "t_max": R_MAX_M,
                    "frame": frame,
                    "camera": camera,
                }
            )
            flag_records.append(
                np.uint64(ray.ray_id).tobytes()
                + bytes([ray.bucket, 1 if ray.occluded else 0, ray.camera, 0])
                + np.uint16(ray.semantic_id).tobytes()
                + b"\x00\x00"
            )
            count += 1
            buckets[ray.bucket] += 1
            occluded_count += int(bool(ray.occluded))
        expected = entry["camera_counts"]["image_00" if camera == 0 else "image_01"]
        if count != int(expected):
            raise PreprocessError(
                f"camera {camera} count drift for frame {frame}: manifest {expected}, exported {count}"
            )
        if frame not in gt_counts or gt_counts[frame] != count:
            raise PreprocessError(f"gt_manifest.json camera_00 drift for frame {frame}")
        for index in range(4):
            bucket_totals[index] += buckets[index]
        occluded_total += occluded_count
        per_frame.append(
            {
                "frame": frame,
                "count": count,
                "occluded_count": occluded_count,
                "bucket_counts": {name: buckets[index] for index, name in enumerate(("0_20", "20_40", "40_50", "50_60"))},
            }
        )

    if expected_total is not None and len(exported) != expected_total:
        raise PreprocessError(f"camera-{camera} total drift: expected {expected_total}, exported {len(exported)}")

    target = ensure_dir(out_dir)
    gt_rays_path = write_gt_rays(target / "gt_rays.bin", exported)

    header = bytearray(32)
    header[0:4] = b"RLGF"
    header[4:6] = np.uint16(1).tobytes()
    header[6:8] = np.uint16(16).tobytes()
    header[8:12] = np.uint32(len(flag_records)).tobytes()
    flags_path = target / "gt_rays_flags.bin"
    flags_path.write_bytes(bytes(header) + b"".join(flag_records))

    document = {
        "schema_version": 1,
        "kind": "gt_rays_export",
        "generated_at_utc": utc_now(),
        "code_version": "0.1.0",
        "filter": {"camera": camera},
        "source": {
            "rays_manifest_sha256": sha256_file(rays_manifest),
            "gt_manifest_sha256": sha256_file(gt_manifest),
            "cameras_json_sha256": sha256_file(cameras_json),
        },
        "counts": {
            "total": len(exported),
            "per_frame": per_frame,
            "per_bucket": {name: bucket_totals[index] for index, name in enumerate(("0_20", "20_40", "40_50", "50_60"))},
            "occluded": occluded_total,
        },
        "output": {
            "path": gt_rays_path.name,
            "sha256": sha256_file(gt_rays_path),
            "record_size": GT_RAY_RECORD_SIZE,
            "version": GT_RAY_VERSION,
        },
        "flags": {"path": flags_path.name, "sha256": sha256_file(flags_path)},
        "camera": "image_00" if camera == 0 else "image_01",
    }
    write_json(target / "gt_rays_manifest.json", document)
    return document


def assemble_bundle(config: Phase2Config, frames: list[int], out_dir: str | Path) -> dict:
    root = ensure_dir(out_dir)
    for sub in (
        "raw/images/image_00",
        "raw/images/image_01",
        "raw/semantics/image_00",
        "raw/semantics/image_01",
        "raw/lidar",
        "raw/poses",
        "raw/timestamps",
        "raw/calibration",
        "geometry",
        "derived/rays",
        "derived/gt",
        "derived/masks",
        "derived/segment",
        "derived/g2",
        "derived/uncurl",
        "provenance",
        "meta",
    ):
        ensure_dir(root / sub)

    shutil.copytree(config.dataset_root / "calibration", root / "raw" / "calibration", dirs_exist_ok=True)
    for frame in frames:
        for camera in config.cameras:
            shutil.copy2(
                image_path(config.dataset_root, config.sequence, camera, frame),
                root / "raw" / "images" / camera / f"{frame:010d}.png",
            )
            shutil.copy2(
                semantic_path(config.dataset_root, config.sequence, camera, frame),
                root / "raw" / "semantics" / camera / f"{frame:010d}.png",
            )
        shutil.copy2(lidar_path(config.lidar_root, frame), root / "raw" / "lidar" / f"{frame:010d}.bin")

    keep = set(frames)
    (root / "raw" / "poses" / "poses.txt").write_text(
        "\n".join(_select_lines(poses_path(config.dataset_root, config.sequence), keep)) + "\n", encoding="utf-8"
    )
    (root / "raw" / "poses" / "cam0_to_world.txt").write_text(
        "\n".join(_select_lines(cam0_to_world_path(config.dataset_root, config.sequence), keep)) + "\n",
        encoding="utf-8",
    )
    for camera in config.cameras:
        timestamps_file = image_timestamps_path(config.dataset_root, config.sequence, camera)
        if timestamps_file.is_file():
            (root / "raw" / "timestamps" / f"{camera}.txt").write_text(
                "\n".join(_selected_timestamps(timestamps_file, frames)) + "\n",
                encoding="utf-8",
            )
    (root / "raw" / "timestamps" / "velodyne.txt").write_text(
        "\n".join(_selected_timestamps(lidar_timestamps_path(config.lidar_root), frames)) + "\n", encoding="utf-8"
    )

    calib = load_kitti360_calib(config.dataset_root / "calibration")
    poses = load_poses(poses_path(config.dataset_root, config.sequence))
    write_calibration_geometry(calib, poses, frames, config.sequence, root / "geometry")

    for frame in frames:
        name = f"{frame:010d}"
        for suffix in (".bin", "_meta.json"):
            source = config.phase_dir / "rays" / f"frame_{name}{suffix}"
            if source.is_file():
                shutil.copy2(source, root / "derived" / "rays" / source.name)
        gt_source = config.phase_dir / "gt" / f"frame_{name}.bin"
        if gt_source.is_file():
            shutil.copy2(gt_source, root / "derived" / "gt" / gt_source.name)
        mask_source = config.phase_dir / "masks" / f"frame_{name}"
        if mask_source.is_dir():
            shutil.copytree(mask_source, root / "derived" / "masks" / f"frame_{name}", dirs_exist_ok=True)

    for name, folder in (
        ("rays_manifest.json", "rays"),
        ("RAY_SET_FREEZE.json", "rays"),
        ("gt_manifest.json", "gt"),
        ("mask_manifest.json", "masks"),
        ("segment_manifest.json", "segment"),
        ("candidate_windows.csv", "segment"),
        ("G2_REPORT.json", "g2"),
        ("G2_REPORT.md", "g2"),
        ("UNCURL_CAVEAT.md", "g2"),
        ("uncurl_report.json", "uncurl"),
        ("azimuth_bins.csv", "uncurl"),
    ):
        source = config.phase_dir / folder / name
        if source.is_file():
            shutil.copy2(source, root / "derived" / folder / name)

    (root / "meta" / "selected_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n", encoding="utf-8")
    (root / "meta" / "selected_frames.csv").write_text(",".join(str(f) for f in frames) + "\n", encoding="utf-8")
    write_json(
        root / "meta" / "selected_manifest.json",
        {
            "schema_version": 1,
            "sequence": config.sequence,
            "selected": {
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "frame_count": len(frames),
                "frames": frames,
            },
        },
    )

    sums = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        sums.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return {"root": str(root), "frames": len(frames), "files": len(sums)}
