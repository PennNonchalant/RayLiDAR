"""Strict run-configuration loading and canonical hashing.

A config file is the *only* place a route or frame window is selected (D-17).
Loading is deliberately strict: unknown top-level keys are rejected, and any
key or path that could point construction at GT or LiDAR data is rejected with
a loud ``ValueError`` (D-09, T-04-01-03).  Evaluation data must never influence
mesh construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from .contracts import WindowSpec

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "route",
        "bundle_root",
        "out_root",
        "phase2_dir",
        "windows",
        "params",
    }
)

ROUTES = frozenset({"colmap_openmvs"})

# A key is forbidden when one of these tokens is one of its underscore/dot/dash
# separated parts (so ``gt_path`` and ``use_alignment`` are caught, while
# unrelated words such as ``weight`` are not).
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

# Substrings that identify evaluation-only inputs.  No reconstruction input may
# live under these trees.
FORBIDDEN_PATH_MARKERS = (
    "raw/lidar",
    "derived/gt",
)
FORBIDDEN_PATH_PATTERN = re.compile(r"artifacts/02[^/]*/(?:gt|rays)")

_KEY_SPLIT = re.compile(r"[_\-.]")


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
            raise ValueError("GT/LiDAR input is not a reconstruction input")
        if any(marker in normalised for marker in FORBIDDEN_PATH_MARKERS):
            raise ValueError("GT/LiDAR input is not a reconstruction input")


def _validate_windows(windows: object) -> None:
    if not isinstance(windows, dict):
        raise ValueError("config.windows must be a mapping")
    for name in ("smoke", "full"):
        if name not in windows:
            raise ValueError(f"config.windows is missing '{name}'")
        frames = windows[name]
        if not isinstance(frames, list):
            raise ValueError(f"config.windows.{name} must be a list of frame ids")
        # WindowSpec rejects empty, non-increasing, non-contiguous frames.
        WindowSpec(name, tuple(frames))


def validate_config(cfg: dict) -> dict:
    """Validate an already-parsed config mapping and return it unchanged."""
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")

    working = {key: value for key, value in cfg.items() if not key.startswith("_")}
    _reject_forbidden_keys(working)
    _reject_forbidden_paths(working)

    unknown = set(working) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level config key: {sorted(unknown)[0]}")

    missing = TOP_LEVEL_KEYS - set(working)
    if missing:
        raise ValueError(f"missing top-level config key: {sorted(missing)[0]}")

    if cfg["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version: {cfg['schema_version']!r}")

    route = cfg["route"]
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route!r}")

    for key in ("bundle_root", "out_root", "phase2_dir"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"config.{key} must be a non-empty path string")

    _validate_windows(cfg["windows"])

    if not isinstance(cfg["params"], dict):
        raise ValueError("config.params must be a mapping")

    return cfg


def load_config(path: str | Path) -> dict:
    """Read, validate, and return a YAML config file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validated = validate_config(cfg)
    validated["_source"] = str(path.resolve())
    return validated


def canonical_json(cfg: dict) -> str:
    """Canonical JSON used for hashing (sorted keys, no whitespace)."""
    # ``phase2_dir`` locates the read-only GT cache for G3 diagnostics; it is a
    # local data path, not part of the mesh construction, so it is excluded from
    # the run hash.
    payload = {
        key: value
        for key, value in cfg.items()
        if not key.startswith("_") and key != "phase2_dir"
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_sha256(cfg: dict) -> str:
    """Return the SHA-256 of the canonicalised config as lowercase hex."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


__all__ = [
    "FORBIDDEN_KEY_TOKENS",
    "FORBIDDEN_PATH_MARKERS",
    "ROUTES",
    "TOP_LEVEL_KEYS",
    "canonical_json",
    "config_sha256",
    "load_config",
    "validate_config",
]
