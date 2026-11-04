"""Freeze the rendered-depth convention from one frame before any metric.

A Gaussian renderer may export either a camera z-depth or a ray range.  One
frame (``protocol.depth_probe_frame``), sampled with a fixed stride
(``protocol.depth_probe_stride``), decides which conversion makes the rendered
value agree with the frozen GT range:

* ``t_z = z * ||((u-cx)/fx, (v-cy)/fy, 1)||`` — the z-depth hypothesis;
* ``t_direct = z`` — the already-range hypothesis.

The decision is written once to ``depth_convention.json`` together with the
off-axis evidence and is then used unchanged by every hit/metric stage.  This
stage deliberately reads no final metric: it only measures the two probe
errors, so it cannot be tuned to a metric outcome.

The probe is evaluation-only: it reads the frozen camera-0 ``gt_rays.bin`` but
never feeds back into training or rendering (D-11).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .protocol.cameras import load_cameras
from .protocol.depth import nearest_pixel, project_to_pixel, z_to_range
from .protocol.gt_rays_io import read_gt_rays
from .run_utils import read_json, sha256_file, utc_now, write_json_atomic

SCHEMA_VERSION = 1
METHODS = ("3dgs", "2dgs")
SUPPORTED_CONVENTIONS = ("z_depth", "range")
FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
OFF_AXIS_PIXELS = 50.0
RATIONALE = "off-axis pixel convention check; not a metric tuning step"


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def artifacts_root(cfg: dict) -> Path:
    """Return the Phase-5 artifact root."""
    return Path(cfg["paths"]["artifacts_root"])


def gt_rays_path(cfg: dict) -> Path:
    """Return the frozen camera-0 ``gt_rays.bin`` from IO-02."""
    return artifacts_root(cfg) / "io02" / "gt_rays.bin"


def flags_path(cfg: dict) -> Path:
    """Return the frozen ``gt_rays_flags.bin`` sidecar from IO-02."""
    return artifacts_root(cfg) / "io02" / "gt_rays_flags.bin"


def cameras_json_path(cfg: dict) -> Path:
    """Return ``geometry/cameras.json`` from the frozen bundle."""
    return Path(cfg["paths"]["bundle"]) / "geometry" / "cameras.json"


def convention_path(cfg: dict) -> Path:
    """Return the frozen ``depth_convention.json`` path."""
    return artifacts_root(cfg) / "depth_convention.json"


def _config_digest(context: dict, cfg: dict) -> str:
    digest = context.get("config_sha256")
    if digest:
        return str(digest)
    from .config import config_sha256  # noqa: PLC0415 - avoid an import cycle

    return config_sha256(cfg)


def locate_render_dir(cfg: dict, method: str) -> Path:
    """Return the non-smoke render directory for one method.

    The canonical directory is ``renders/<method>-<run_id>``.  Its manifest must
    declare the method and the full frame window, so a 2-frame smoke render can
    never be mistaken for the production run.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    root = artifacts_root(cfg) / "renders" / f"{method}-{cfg['run_id']}"
    manifest_path = root / "render_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"render manifest missing: {manifest_path}; run --render-{method} first"
        )
    manifest = read_json(manifest_path)
    if manifest.get("method") != method:
        raise RuntimeError(
            f"{manifest_path}: method {manifest.get('method')!r} != {method!r}"
        )
    per_frame = manifest.get("per_frame")
    expected_frames = [int(frame) for frame in cfg["frames"]]
    if not isinstance(per_frame, list):
        raise RuntimeError(f"{manifest_path}: per_frame must be a list")
    recorded = [int(entry.get("frame", -1)) for entry in per_frame]
    if recorded != expected_frames:
        raise RuntimeError(
            f"{manifest_path}: rendered frames {recorded} != expected "
            f"{expected_frames}; refusing a smoke or stale render"
        )
    resolution = manifest.get("resolution")
    if resolution is not None:
        values = [int(value) for value in resolution]
        if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
            raise RuntimeError(f"{manifest_path}: invalid resolution {resolution!r}")
    return root


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def _load_map(path: Path, name: str, width: int, height: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path}: {name} must be a 2-D array")
    if array.dtype not in FLOAT_DTYPES:
        raise ValueError(f"{path}: {name} dtype must be float32/float64")
    if tuple(int(value) for value in array.shape) != (height, width):
        raise ValueError(
            f"{path}: {name} shape {array.shape} does not match {(height, width)}"
        )
    return array


def _ratio(direct: float, z_depth: float) -> float | None:
    if z_depth > 0.0:
        return direct / z_depth
    return None if direct > 0.0 else 1.0


def run_probe(cfg: dict, methods: dict[str, str | Path], out_path: str | Path) -> dict:
    """Sample one frame and freeze the depth convention to ``out_path``.

    ``methods`` maps ``3dgs``/``2dgs`` to that method's render directory.  Both
    methods must select the same convention; otherwise the probe raises instead
    of guessing.
    """
    protocol = cfg["protocol"]
    frame = int(protocol["depth_probe_frame"])
    stride = int(protocol["depth_probe_stride"])
    alpha_threshold = float(protocol["alpha_threshold"])
    if stride < 1:
        raise ValueError("protocol.depth_probe_stride must be >= 1")

    gt_rays = read_gt_rays(gt_rays_path(cfg))
    selected = [
        ray
        for ray in gt_rays
        if int(ray["camera"]) == 0 and int(ray["frame"]) == frame
    ][::stride]
    if not selected:
        raise RuntimeError(
            f"probe frame {frame} has no camera-0 rays in {gt_rays_path(cfg)}"
        )

    cameras = load_cameras(cameras_json_path(cfg))
    K = cameras["K"]
    width = int(cameras["width"])
    height = int(cameras["height"])
    # Production resolution is enforced by ``locate_render_dir`` (render manifest)
    # and by the shared protocol's 1408x376 checks in ``build_hits``; the probe
    # itself follows the camera metadata so a small synthetic fixture is usable.
    T_C_W = cameras["frames"][frame]["T_C_W"]
    cx = float(K[0][2])
    cy = float(K[1][2])

    results: dict[str, dict] = {}
    decisions: dict[str, str] = {}
    for method in METHODS:
        if method not in methods:
            raise ValueError(f"probe is missing render directory for {method}")
        render_dir = Path(methods[method])
        depth = _load_map(
            render_dir / f"{frame:010d}_depth.npy", "depth", width, height
        )
        alpha = _load_map(
            render_dir / f"{frame:010d}_alpha.npy", "alpha", width, height
        )

        errors_z: list[float] = []
        errors_direct: list[float] = []
        n_projected = 0
        n_valid = 0
        n_off_axis = 0
        for ray in selected:
            projected = project_to_pixel(ray["dir_W"], T_C_W, K)
            if projected is None:
                continue
            n_projected += 1
            u, v = projected
            if abs(float(u) - cx) > OFF_AXIS_PIXELS:
                n_off_axis += 1
            pixel = nearest_pixel(u, v, width, height)
            if pixel is None:
                continue
            iu, iv = pixel

            alpha_value = float(alpha[iv, iu])
            if not (
                math.isfinite(alpha_value) and alpha_value >= alpha_threshold
            ):
                continue
            z = float(depth[iv, iu])
            if not (math.isfinite(z) and z > 0.0):
                continue

            n_valid += 1
            t_gt = float(ray["t_gt"])
            errors_z.append(abs(z_to_range(z, iu, iv, K) - t_gt))
            errors_direct.append(abs(z - t_gt))

        if n_valid == 0:
            raise RuntimeError(
                f"{method}: no probe ray passed alpha/depth checks at frame {frame}"
            )

        median_z = float(np.median(errors_z))
        median_direct = float(np.median(errors_direct))
        decision = "z_depth" if median_z < median_direct else "range"
        decisions[method] = decision
        results[method] = {
            "render_dir": str(render_dir),
            "n_valid": n_valid,
            "n_projected": n_projected,
            "off_axis_fraction": (
                n_off_axis / n_projected if n_projected else 0.0
            ),
            "median_abs_error_z": median_z,
            "median_abs_error_direct": median_direct,
            "decision": decision,
        }

    unique = set(decisions.values())
    if len(unique) != 1:
        raise RuntimeError(
            "the two methods disagree on the depth convention: "
            f"{decisions}; refusing to freeze a mixed protocol"
        )
    decision = decisions[METHODS[0]]
    if decision not in SUPPORTED_CONVENTIONS:
        raise RuntimeError(f"unsupported frozen decision: {decision!r}")

    worst_z = max(results[method]["median_abs_error_z"] for method in METHODS)
    worst_direct = max(
        results[method]["median_abs_error_direct"] for method in METHODS
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "frame": frame,
        "stride": stride,
        "n_selected": len(selected),
        "methods": results,
        "decision": decision,
        "evidence": {
            "median_abs_error_z": worst_z,
            "median_abs_error_direct": worst_direct,
            "ratio": _ratio(worst_direct, worst_z),
            "ratio_definition": (
                "median_abs_error_direct / median_abs_error_z"
            ),
            "aggregation": "max_across_methods",
            "per_method": {
                method: {
                    "median_abs_error_z": results[method]["median_abs_error_z"],
                    "median_abs_error_direct": results[method][
                        "median_abs_error_direct"
                    ],
                    "ratio": _ratio(
                        results[method]["median_abs_error_direct"],
                        results[method]["median_abs_error_z"],
                    ),
                }
                for method in METHODS
            },
        },
        "frozen_at_utc": utc_now(),
        "rationale": RATIONALE,
    }
    write_json_atomic(out_path, document)
    return document


def run(context: dict) -> dict:
    """Pipeline stage: freeze the depth convention for both methods."""
    cfg = context.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("probe stage requires context['cfg']")

    methods = {method: locate_render_dir(cfg, method) for method in METHODS}
    out_path = convention_path(cfg)
    document = run_probe(cfg, methods, out_path)

    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": "probe",
        "method": None,
        "generated_at_utc": utc_now(),
        "config_sha256": _config_digest(context, cfg),
        "depth_convention": str(out_path),
        "depth_convention_sha256": sha256_file(out_path),
        "frame": document["frame"],
        "stride": document["stride"],
        "n_selected": document["n_selected"],
        "decision": document["decision"],
        "methods": document["methods"],
        "evidence": document["evidence"],
    }
    write_json_atomic(artifacts_root(cfg) / "stages" / "probe.json", record)
    return record


__all__ = [
    "METHODS",
    "OFF_AXIS_PIXELS",
    "RATIONALE",
    "SCHEMA_VERSION",
    "artifacts_root",
    "cameras_json_path",
    "convention_path",
    "flags_path",
    "gt_rays_path",
    "locate_render_dir",
    "run",
    "run_probe",
]
