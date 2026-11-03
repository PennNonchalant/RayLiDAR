"""Build the frozen, GT-free Phase-5 Gaussian training dataset.

The official 3DGS and 2DGS loaders expect a COLMAP scene::

    <dataset>/
      images/<frame:010d>.png      # 60 rectified image_00 frames, 1408x376
      sparse/0/{cameras,images,points3D}.bin
      sparse/0/points3D.ply        # generated once so training never mutates inputs
      dataset_manifest.json
      SHA256SUMS

Every copied byte is hashed, every source is checked for GT/LiDAR leakage, and
the manifest is written atomically so a failed build never leaves a half-written
record.  ``points3D.ply`` is generated here (not on the trainer's first run) so
the frozen dataset bytes are final before any Gaussian training starts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DATA_ROOT, REPO_ROOT, config_sha256
from .run_utils import check_free_space, sha256_file, utc_now, write_json_atomic

SCHEMA_VERSION = 1
DATASET_KIND = "gaussian_training_dataset"

FROZEN_K: tuple[tuple[float, ...], ...] = (
    (552.554261, 0.0, 682.049453),
    (0.0, 552.554261, 238.769549),
    (0.0, 0.0, 1.0),
)
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
COLMAP_BIN = Path("/data-20T/conda_envs/colmap/bin/colmap")

FORBIDDEN_INPUTS = (
    "raw/lidar",
    "derived/gt",
)
FORBIDDEN_INPUT_PATTERN = re.compile(r"artifacts/02[^/]*/(?:gt|rays)")

MODEL_BANNER_PATTERN = re.compile(
    r"model\.cc:\d+\]\s+(Cameras|Images|Points):\s+(\d+)"
)

PLY_SCRIPT = """\
import sys

from scene.colmap_loader import read_points3D_binary
from scene.dataset_readers import storePly

xyz, rgb, _ = read_points3D_binary(sys.argv[1])
storePly(sys.argv[2], xyz, rgb)
print(len(xyz))
"""


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def _forbidden_roots() -> tuple[Path, ...]:
    return tuple((DATA_ROOT / marker).resolve() for marker in FORBIDDEN_INPUTS)


def _reject_forbidden_inputs(*paths: Path) -> None:
    """Fail when a training source lives under a GT/LiDAR tree (D-11)."""
    roots = _forbidden_roots()
    for path in paths:
        text = str(path).replace("\\", "/")
        if FORBIDDEN_INPUT_PATTERN.search(text):
            raise ValueError("GT/LiDAR input is not a training input")
        for marker in FORBIDDEN_INPUTS:
            if marker in text:
                raise ValueError("GT/LiDAR input is not a training input")
        resolved = Path(path).resolve()
        for root in roots:
            if resolved == root or resolved.is_relative_to(root):
                raise ValueError("GT/LiDAR input is not a training input")


# ---------------------------------------------------------------------------
# bundle and COLMAP validation
# ---------------------------------------------------------------------------


def _matrix_close(actual: object, expected: object, tolerance: float = 1e-9) -> bool:
    try:
        rows = [[float(value) for value in row] for row in actual]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return False
    if len(rows) != len(expected):  # type: ignore[arg-type]
        return False
    for got_row, want_row in zip(rows, expected):  # type: ignore[arg-type]
        if len(got_row) != len(want_row):
            return False
        for got, want in zip(got_row, want_row):
            if abs(got - float(want)) > tolerance:
                return False
    return True


def _validate_bundle(
    bundle: Path,
    frames: list[int],
    camera: str,
    resolution: list[int],
) -> dict[str, Any]:
    if not bundle.is_dir():
        raise ValueError(f"bundle does not exist: {bundle}")

    images_dir = bundle / "raw" / "images" / camera
    if not images_dir.is_dir():
        raise ValueError(f"bundle is missing the image directory: {images_dir}")

    expected = {f"{frame:010d}.png": frame for frame in frames}
    actual = {entry.name for entry in images_dir.iterdir() if entry.is_file()}
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise ValueError(
            f"image set mismatch in {images_dir}: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    sizes: dict[int, tuple[int, int]] = {}
    for name, frame in sorted(expected.items(), key=lambda item: item[1]):
        path = images_dir / name
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError as exc:
            raise ValueError(f"cannot read image {path}: {exc}") from exc
        if [width, height] != list(resolution):
            raise ValueError(
                f"image {path} is {width}x{height}, expected "
                f"{resolution[0]}x{resolution[1]}"
            )
        sizes[frame] = (width, height)

    geometry = bundle / "geometry"
    cameras_json = geometry / "cameras.json"
    frames_csv = geometry / "frames.csv"
    selected_frames = bundle / "meta" / "selected_frames.txt"
    for required in (cameras_json, frames_csv, selected_frames):
        if not required.is_file():
            raise ValueError(f"bundle is missing {required}")

    try:
        payload = json.loads(cameras_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {cameras_json}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{cameras_json} schema_version must be 1, got {payload.get('schema_version')!r}"
        )
    cameras = payload.get("cameras") or {}
    if camera not in cameras:
        raise ValueError(f"{cameras_json} has no camera {camera!r}")
    if not _matrix_close((cameras[camera] or {}).get("K"), FROZEN_K):
        raise ValueError(f"{cameras_json} cameras.{camera}.K differs from the frozen K")
    recorded_frames = [int(item.get("frame", -1)) for item in payload.get("frames") or []]
    if recorded_frames != frames:
        raise ValueError(
            f"{cameras_json} frames do not match the configured window "
            f"({len(recorded_frames)} vs {len(frames)})"
        )

    selected = [
        int(line.strip())
        for line in selected_frames.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if selected != frames:
        raise ValueError(f"{selected_frames} does not match the configured frames")

    csv_lines = frames_csv.read_text(encoding="utf-8").splitlines()
    if len(csv_lines) != len(frames) + 1:
        raise ValueError(
            f"{frames_csv} has {len(csv_lines)} lines, expected {len(frames) + 1}"
        )
    if not csv_lines[0].startswith("frame"):
        raise ValueError(f"{frames_csv} header must start with 'frame'")

    return {
        "sequence": payload.get("sequence"),
        "sizes": sizes,
        "source": {
            "cameras_json_sha256": sha256_file(cameras_json),
            "selected_frames_sha256": sha256_file(selected_frames),
            "frames_csv_sha256": sha256_file(frames_csv),
        },
    }


def _run_model_analyzer(model_dir: Path) -> str:
    """Run the pinned ``colmap model_analyzer`` and return its banner."""
    if not COLMAP_BIN.is_file():
        raise RuntimeError(f"colmap model_analyzer not found at {COLMAP_BIN}")
    completed = subprocess.run(
        [str(COLMAP_BIN), "model_analyzer", "--path", str(model_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    banner = f"{completed.stdout or ''}{completed.stderr or ''}"
    if completed.returncode != 0:
        raise RuntimeError(
            f"colmap model_analyzer failed ({completed.returncode}) for {model_dir}:\n"
            f"{banner}"
        )
    return banner


def _parse_model_banner(banner: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in MODEL_BANNER_PATTERN.finditer(banner):
        counts[match.group(1)] = int(match.group(2))
    return counts


def _validate_colmap(model_dir: Path, frame_count: int) -> tuple[str, dict[str, int]]:
    if not model_dir.is_dir():
        raise ValueError(f"COLMAP model does not exist: {model_dir}")
    for name in SPARSE_FILES:
        if not (model_dir / name).is_file():
            raise ValueError(f"COLMAP model is missing {model_dir / name}")

    banner = _run_model_analyzer(model_dir)
    counts = _parse_model_banner(banner)
    expected = {"Cameras": frame_count, "Images": frame_count}
    for key, value in expected.items():
        if counts.get(key) != value:
            raise ValueError(
                f"COLMAP model {key} = {counts.get(key)}, expected {value}"
            )
    points = counts.get("Points")
    if not isinstance(points, int) or points <= 0:
        raise ValueError(f"COLMAP model Points = {points!r}, expected a positive count")
    return banner, counts


def _write_points3d_ply(sparse_dir: Path, upstream: Path, expected_points: int) -> Path:
    """Generate ``points3D.ply`` with the pinned official reader + storePly."""
    points_binary = sparse_dir / "points3D.bin"
    ply_path = sparse_dir / "points3D.ply"
    environment = dict(os.environ)
    python_path = str(upstream)
    if environment.get("PYTHONPATH"):
        python_path = python_path + os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = python_path

    completed = subprocess.run(
        [sys.executable, "-c", PLY_SCRIPT, str(points_binary), str(ply_path)],
        cwd=str(upstream),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official points3D.ply generation failed "
            f"({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
        )
    try:
        count = int(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"could not parse the points3D.ply point count: {completed.stdout!r}"
        ) from exc
    if count != int(expected_points):
        raise RuntimeError(
            f"points3D.ply contains {count} points, expected {expected_points}"
        )
    return ply_path


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _code_version() -> str:
    try:
        completed = subprocess.run(
            # The container runs as root while /src is owned by the host user,
            # so git refuses the tree without an explicit safe.directory.
            ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256sums(pairs: list[tuple[str, str]]) -> str:
    return "".join(f"{digest}  {relative}\n" for digest, relative in pairs)


def build_dataset(cfg: dict, out_root: str | Path) -> dict:
    """Copy, hash, and manifest the frozen training dataset.

    Returns the manifest mapping written to ``<out_root>/dataset/dataset_manifest.json``.
    """
    frames = [int(frame) for frame in cfg["frames"]]
    camera = str(cfg["camera"])
    resolution = [int(value) for value in cfg["resolution"]]
    seed = int(cfg["seed"])
    run_id = str(cfg["run_id"])
    paths = cfg["paths"]

    bundle = Path(paths["bundle"])
    colmap_model = Path(paths["colmap_model"])
    # The official 3DGS package lives under <upstream>/<method.gs3d.upstream_subdir>;
    # `scene` is not importable from the upstream root itself.
    gs3d_subdir = (
        (cfg.get("method") or {}).get("gs3d", {}).get("upstream_subdir")
        or "gaussian-splatting"
    )
    upstream = Path(paths["upstream"]) / str(gs3d_subdir)

    _reject_forbidden_inputs(bundle, colmap_model)

    bundle_info = _validate_bundle(bundle, frames, camera, resolution)
    banner, counts = _validate_colmap(colmap_model, len(frames))

    out_root = Path(out_root)
    dataset_dir = out_root / "dataset"
    images_out = dataset_dir / "images"
    sparse_out = dataset_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    image_entries: list[dict[str, Any]] = []
    for frame in frames:
        name = f"{frame:010d}.png"
        source = bundle / "raw" / "images" / camera / name
        destination = images_out / name
        source_sha256 = sha256_file(source)
        shutil.copyfile(source, destination)
        artifact_sha256 = sha256_file(destination)
        if artifact_sha256 != source_sha256:
            raise RuntimeError(f"copy of {source} changed its bytes")
        width, height = bundle_info["sizes"][frame]
        image_entries.append(
            {
                "frame": frame,
                "source_sha256": source_sha256,
                "artifact_sha256": artifact_sha256,
                "width": width,
                "height": height,
            }
        )

    sparse_hashes: dict[str, str] = {}
    for name in SPARSE_FILES:
        source = colmap_model / name
        destination = sparse_out / name
        source_sha256 = sha256_file(source)
        shutil.copyfile(source, destination)
        artifact_sha256 = sha256_file(destination)
        if artifact_sha256 != source_sha256:
            raise RuntimeError(f"copy of {source} changed its bytes")
        sparse_hashes[name] = artifact_sha256

    ply_path = _write_points3d_ply(sparse_out, upstream, int(counts["Points"]))
    ply_sha256 = sha256_file(ply_path)

    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "run_id": run_id,
        "sequence": bundle_info["sequence"],
        "window": {"first": frames[0], "last": frames[-1], "count": len(frames)},
        "camera": camera,
        "resolution": resolution,
        "seed": seed,
        "config_sha256": config_sha256(cfg),
        "source": {
            "bundle": str(bundle),
            "colmap_model": str(colmap_model),
            "cameras_json_sha256": bundle_info["source"]["cameras_json_sha256"],
            "selected_frames_sha256": bundle_info["source"][
                "selected_frames_sha256"
            ],
            "frames_csv_sha256": bundle_info["source"]["frames_csv_sha256"],
        },
        "colmap_banner": banner.strip().splitlines(),
        "images": image_entries,
        "sparse": {
            "cameras_sha256": sparse_hashes["cameras.bin"],
            "images_sha256": sparse_hashes["images.bin"],
            "points3D_sha256": sparse_hashes["points3D.bin"],
            "points3D_ply_sha256": ply_sha256,
            "cameras": counts["Cameras"],
            "images": counts["Images"],
            "points3D": counts["Points"],
        },
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
        "no_gt_inputs": True,
        "generated_at_utc": utc_now(),
        "code_version": _code_version(),
        "free_space_bytes": int(shutil.disk_usage(str(out_root)).free),
    }

    manifest_sha256 = write_json_atomic(manifest_path, manifest)

    sums: list[tuple[str, str]] = [
        (entry["artifact_sha256"], f"images/{entry['frame']:010d}.png")
        for entry in image_entries
    ]
    for name in SPARSE_FILES:
        sums.append((sparse_hashes[name], f"sparse/0/{name}"))
    sums.append((ply_sha256, "sparse/0/points3D.ply"))
    sums.append((manifest_sha256, "dataset_manifest.json"))
    _write_text_atomic(dataset_dir / "SHA256SUMS", _sha256sums(sums))

    print(f"dataset: {dataset_dir}")
    print(f"images: {len(image_entries)}")
    print(f"points3D: {counts['Points']}")
    print(f"manifest_sha256: {manifest_sha256}")
    return manifest


def run(ctx: dict) -> dict:
    """Pipeline stage: guard free space, build the dataset, record provenance."""
    cfg = ctx.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("dataset stage requires ctx.cfg")
    artifacts_root = Path(cfg["paths"]["artifacts_root"])
    artifacts_root.mkdir(parents=True, exist_ok=True)
    free_space_bytes = check_free_space(artifacts_root)

    manifest = build_dataset(cfg, artifacts_root)

    manifest_path = artifacts_root / "dataset" / "dataset_manifest.json"
    digest = str(ctx.get("config_sha256") or config_sha256(cfg))
    stage = {
        "schema_version": SCHEMA_VERSION,
        "stage": "dataset",
        "generated_at_utc": utc_now(),
        "config_sha256": digest,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_dir": str(artifacts_root / "dataset"),
        "camera": manifest["camera"],
        "window": manifest["window"],
        "images": len(manifest["images"]),
        "points3D": manifest["sparse"]["points3D"],
        "no_gt_inputs": manifest["no_gt_inputs"],
        "free_space_bytes": free_space_bytes,
    }
    write_json_atomic(artifacts_root / "stages" / "dataset.json", stage)
    return stage


__all__ = [
    "FROZEN_K",
    "FORBIDDEN_INPUTS",
    "build_dataset",
    "run",
]
