"""Strict Phase-5 run configuration, path guards, and canonical hashing.

The YAML config is the only place a Phase-5 run chooses its frame window,
camera, resolution, seed, method parameters, and paths.  Loading is strict on
purpose:

* the configured training window (contiguous 60 frames, ``image_00``, native
  ``1408x376``) and the official method defaults are asserted;
* any key that could point training at ground truth or LiDAR data is rejected;
* any path string that could escape the approved roots is rejected.

Phase 5 must be reproducible from one config hash, so ``config_sha256`` hashes
the canonical JSON encoding of the validated mapping (sorted keys, no
whitespace, ASCII).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data-20T/hybrid-lidar-sim")
CONTAINER_ROOT = Path("/src")

SCHEMA_VERSION = 1
EXPECTED_FRAME_COUNT = 60
EXPECTED_CAMERA = "image_00"
FROZEN_RESOLUTION: tuple[int, int] = (1408, 376)
EXPECTED_RENDER: dict[str, str] = {"rgb": "png", "depth": "npy", "alpha": "npy"}
EXPECTED_DEPTH_RATIO = 0.0

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "paths",
        "frames",
        "camera",
        "resolution",
        "seed",
        "method",
        "protocol",
        "eval",
        "report",
    }
)

PATH_KEYS = ("bundle", "colmap_model", "artifacts_root", "upstream", "venv")

# A key is forbidden when one of these tokens is one of its underscore / dot /
# dash separated parts (``gt_dir``, ``lidar_path`` and ``use_alignment`` are all
# caught while unrelated words such as ``weight`` are not).
FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "gt",
        "ground_truth",
        "lidar",
        "laser",
        "alignment",
        "umeyama",
        "icp",
    }
)

# Substrings that identify evaluation-only inputs.  No training input may live
# under these trees (D-11).
FORBIDDEN_PATH_MARKERS = (
    "raw/lidar",
    "derived/gt",
)
FORBIDDEN_PATH_PATTERN = re.compile(r"artifacts/02[^/]*/(?:gt|rays)")

# Relative repository paths that are explicitly allowed by the schema.
ALLOWED_RELATIVE_PATHS = frozenset({"reconstruct/configs/eval.yaml", "reconstruct/configs/eval_protocol.md"})

_KEY_SPLIT = re.compile(r"[_\-.]")


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def _split_key(name: object) -> set[str]:
    lowered = str(name).lower()
    parts = {part for part in _KEY_SPLIT.split(lowered) if part}
    parts.add(lowered)
    return parts


def _reject_forbidden_keys(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if _split_key(key) & FORBIDDEN_KEY_TOKENS:
                raise ValueError(f"forbidden config key: {key}")
            _reject_forbidden_keys(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _reject_forbidden_keys(value)


def _reject_forbidden_paths(node: object) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _reject_forbidden_paths(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _reject_forbidden_paths(value)
    elif isinstance(node, str):
        normalised = node.replace("\\", "/")
        if FORBIDDEN_PATH_PATTERN.search(normalised):
            raise ValueError("GT/LiDAR input is not a training input")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in normalised:
                raise ValueError("GT/LiDAR input is not a training input")


def _allowed_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for root in (DATA_ROOT, CONTAINER_ROOT, REPO_ROOT):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _check_path_string(value: str) -> None:
    """Reject ``..`` traversal and any path outside the approved roots."""
    if value in ALLOWED_RELATIVE_PATHS:
        return
    normalised = value.replace("\\", "/")
    if ".." in PurePosixPath(normalised).parts:
        raise ValueError("config path must not contain '..'")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"config path must be absolute: {value}")
    resolved = path.resolve()
    for root in _allowed_roots():
        if resolved == root or resolved.is_relative_to(root):
            return
    raise ValueError(f"config path is not rooted under an approved root: {value}")


def _reject_unsafe_path_strings(node: object) -> None:
    """Check every path-looking string anywhere in the config tree."""
    if isinstance(node, dict):
        for value in node.values():
            _reject_unsafe_path_strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _reject_unsafe_path_strings(value)
    elif isinstance(node, str):
        if node in ALLOWED_RELATIVE_PATHS:
            return
        if node.startswith("/") or "/" in node:
            _check_path_string(node)


# ---------------------------------------------------------------------------
# schema sections
# ---------------------------------------------------------------------------


def _validate_paths(paths: object) -> None:
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    for key in PATH_KEYS:
        if key not in paths:
            raise ValueError(f"config.paths is missing '{key}'")
        value = paths[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"config.paths.{key} must be a non-empty path string")
        _check_path_string(value)


def _validate_frames(frames: object) -> list[int]:
    if not isinstance(frames, (list, tuple)):
        raise ValueError(f"config.frames must be a list of {EXPECTED_FRAME_COUNT} frame ids")
    try:
        values = [int(frame) for frame in frames]
    except (TypeError, ValueError) as exc:
        raise ValueError("config.frames must contain integer frame ids") from exc
    if len(values) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"config.frames must contain exactly {EXPECTED_FRAME_COUNT} frames"
        )
    if any(values[index] != values[index - 1] + 1 for index in range(1, len(values))):
        raise ValueError("config.frames must be a contiguous ascending window")
    return values


def _validate_resolution(resolution: object) -> None:
    # The rectified KITTI-360 images are 1408x376; the official renderers and
    # the render verifiers are pinned to that native resolution.
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError(f"config.resolution must be {list(FROZEN_RESOLUTION)}")
    try:
        values = [int(value) for value in resolution]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config.resolution must be {list(FROZEN_RESOLUTION)}") from exc
    if values != list(FROZEN_RESOLUTION):
        raise ValueError(f"config.resolution must be {list(FROZEN_RESOLUTION)}")


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("config.seed must be a non-negative integer")


def _validate_method(method: object, frames: list[int]) -> None:
    if not isinstance(method, dict):
        raise ValueError("config.method must be a mapping")
    frame_set = set(frames)
    for name in ("gs3d", "gs2d"):
        if name not in method:
            raise ValueError(f"config.method is missing '{name}'")
        spec = method[name]
        if not isinstance(spec, dict):
            raise ValueError(f"config.method.{name} must be a mapping")
        subdir = spec.get("upstream_subdir")
        if not isinstance(subdir, str) or not subdir:
            raise ValueError(
                f"config.method.{name}.upstream_subdir must be a non-empty string"
            )
        for key in ("iterations", "smoke_iterations"):
            value = spec.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"config.method.{name}.{key} must be a positive integer")
        resolution = spec.get("resolution")
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise ValueError(f"config.method.{name}.resolution must be an integer")
        if resolution != -1 and resolution <= 0:
            raise ValueError(
                f"config.method.{name}.resolution must be -1 or a positive integer"
            )
        smoke_frames = spec.get("smoke_frames")
        if not isinstance(smoke_frames, (list, tuple)) or len(smoke_frames) != 2:
            raise ValueError(
                f"config.method.{name}.smoke_frames must hold two frame ids"
            )
        try:
            smoke_values = [int(frame) for frame in smoke_frames]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"config.method.{name}.smoke_frames must hold integer frame ids"
            ) from exc
        if smoke_values[1] != smoke_values[0] + 1:
            raise ValueError(
                f"config.method.{name}.smoke_frames must be contiguous"
            )
        if not set(smoke_values).issubset(frame_set):
            raise ValueError(
                f"config.method.{name}.smoke_frames must lie inside config.frames"
            )
        if dict(spec.get("render") or {}) != EXPECTED_RENDER:
            raise ValueError(
                f"config.method.{name}.render must be {EXPECTED_RENDER}"
            )
    if method["gs2d"].get("depth_ratio") != EXPECTED_DEPTH_RATIO:
        raise ValueError(
            f"config.method.gs2d.depth_ratio must be {EXPECTED_DEPTH_RATIO}"
        )


def _validate_protocol(protocol: object, frames: list[int]) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("config.protocol must be a mapping")
    alpha = protocol.get("alpha_threshold")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("config.protocol.alpha_threshold must be in (0, 1]")
    if protocol.get("depth_convention") != "z_depth":
        raise ValueError("config.protocol.depth_convention must be 'z_depth'")
    if protocol.get("pixel_rounding") != "half_up":
        raise ValueError("config.protocol.pixel_rounding must be 'half_up'")
    probe_frame = protocol.get("depth_probe_frame")
    if isinstance(probe_frame, bool) or not isinstance(probe_frame, int):
        raise ValueError("config.protocol.depth_probe_frame must be an integer")
    if probe_frame not in set(frames):
        raise ValueError("config.protocol.depth_probe_frame must lie inside config.frames")
    stride = protocol.get("depth_probe_stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("config.protocol.depth_probe_stride must be >= 1")
    if protocol.get("primary_camera") != 0:
        raise ValueError("config.protocol.primary_camera must be 0")
    if protocol.get("report_all_frames") is not True:
        raise ValueError("config.protocol.report_all_frames must be true")


def _validate_eval(eval_spec: object) -> None:
    if not isinstance(eval_spec, dict):
        raise ValueError("config.eval must be a mapping")
    if eval_spec.get("protocol_config") != "reconstruct/configs/eval.yaml":
        raise ValueError("config.eval.protocol_config must be reconstruct/configs/eval.yaml")
    if eval_spec.get("protocol_doc") != "reconstruct/configs/eval_protocol.md":
        raise ValueError("config.eval.protocol_doc must be reconstruct/configs/eval_protocol.md")
    if list(eval_spec.get("modes") or []) != ["all", "unoccluded"]:
        raise ValueError("config.eval.modes must be ['all', 'unoccluded']")


def _validate_report(report: object) -> None:
    if not isinstance(report, dict):
        raise ValueError("config.report must be a mapping")
    title = report.get("title")
    if not isinstance(title, str) or not title:
        raise ValueError("config.report.title must be a non-empty string")
    labels = report.get("labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("config.report.labels must be a non-empty list")
    for label in labels:
        if not isinstance(label, str) or not label:
            raise ValueError("config.report.labels entries must be non-empty strings")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def validate_config(cfg: dict) -> dict:
    """Validate an already-parsed config mapping and return it unchanged."""
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")

    working = {key: value for key, value in cfg.items() if not key.startswith("_")}
    _reject_forbidden_keys(working)
    _reject_forbidden_paths(working)
    _reject_unsafe_path_strings(working)

    missing = sorted(TOP_LEVEL_KEYS - set(working))
    if missing:
        raise ValueError(f"missing top-level config key: {missing[0]}")

    if cfg["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {cfg['schema_version']!r}")
    if not isinstance(cfg["run_id"], str) or not cfg["run_id"]:
        raise ValueError("config.run_id must be a non-empty string")

    _validate_paths(cfg["paths"])
    frames = _validate_frames(cfg["frames"])
    if cfg["camera"] != EXPECTED_CAMERA:
        raise ValueError(f"config.camera must be {EXPECTED_CAMERA!r}")
    _validate_resolution(cfg["resolution"])
    _validate_seed(cfg["seed"])
    _validate_method(cfg["method"], frames)
    _validate_protocol(cfg["protocol"], frames)
    _validate_eval(cfg["eval"])
    _validate_report(cfg["report"])

    return cfg


def load_config(path: str | Path) -> dict:
    """Read, validate, and return a Phase-5 YAML config file."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc
    validated = validate_config(cfg)
    validated["_source"] = str(source.resolve())
    return validated


def canonical_json(cfg: dict) -> str:
    """Canonical JSON used for hashing (sorted keys, no whitespace)."""
    payload = {key: value for key, value in cfg.items() if not key.startswith("_")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(cfg: dict) -> str:
    """Return the SHA-256 of the canonicalised config as lowercase hex."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


__all__ = [
    "ALLOWED_RELATIVE_PATHS",
    "DATA_ROOT",
    "EXPECTED_CAMERA",
    "EXPECTED_DEPTH_RATIO",
    "FROZEN_RESOLUTION",
    "EXPECTED_FRAME_COUNT",
    "FORBIDDEN_KEY_TOKENS",
    "FORBIDDEN_PATH_MARKERS",
    "REPO_ROOT",
    "TOP_LEVEL_KEYS",
    "canonical_json",
    "config_sha256",
    "load_config",
    "validate_config",
]
