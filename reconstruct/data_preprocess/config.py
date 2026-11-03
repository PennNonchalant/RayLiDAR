"""Phase-2 configuration loading.

The YAML file is the only place the dataset roots, window, mask bounds, ray
domain, and GT-cache thresholds are chosen. Loading is intentionally strict
enough to catch typos while remaining tolerant of the legacy keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import PreprocessError, sha256_file

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "preprocess_1820.yaml"
DATA_ROOT = Path("/data-20T/hybrid-lidar-sim")
DEFAULT_SEQUENCE = "2013_05_28_drive_0000_sync"


@dataclass(frozen=True)
class SelectionOptions:
    window_frames: int = 60
    step_frames: int = 10
    min_usable_frames: int = 60
    min_scene_points_per_frame: int = 20000


@dataclass(frozen=True)
class MaskOptions:
    dynamic_classes: tuple[str, ...] = ()
    dynamic_label_ids: tuple[int, ...] = (26, 27, 28, 32, 33, 24, 25)
    ego_roi_x: tuple[float, float] = (-2.5, 2.5)
    ego_roi_y: tuple[float, float] = (-1.2, 1.2)
    ego_roi_z: tuple[float, float] = (-2.0, 0.3)
    bbox_max_time_gap_frames: int = 2
    residual_dynamic_radius_m: float = 1.0


@dataclass(frozen=True)
class RaysOptions:
    t_min_m: float = 1.0e-4
    r_min_m: float = 1.5
    r_max_m: float = 60.0
    image_margin_px: int = 5
    occlusion_neighbor_px: int = 3
    occlusion_sigma_lidar_m: float = 0.02
    occlusion_sigma_multiple: float = 3.0
    occlusion_depth_threshold_m: float = 0.06


@dataclass(frozen=True)
class GtCacheOptions:
    min_points_per_frame: int = 20000
    expected_points_range: tuple[int, int] = (50000, 150000)
    camera_unknown: int = 255
    class_unknown: int = 65535


@dataclass(frozen=True)
class Phase2Config:
    schema_version: int
    sequence: str
    dataset_root: Path
    lidar_root: Path
    bboxes_xml: Path
    cameras: tuple[str, ...]
    phase_dir: Path
    bundle_dir: Path
    selection: SelectionOptions
    mask: MaskOptions
    rays: RaysOptions
    gt_cache: GtCacheOptions
    frames: tuple[int, ...]
    config_sha256: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def image_width(self) -> int:
        return 1408

    @property
    def image_height(self) -> int:
        return 376


def _as_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    return (float(value[0]), float(value[1]))


def load_config(path: str | Path | None = None) -> Phase2Config:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_file():
        raise PreprocessError(f"missing Phase-2 config: {config_path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment error
        raise PreprocessError("PyYAML is required to read the Phase-2 config") from exc

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PreprocessError(f"{config_path}: expected a YAML mapping")

    selection_node = loaded.get("selection") or {}
    mask_node = loaded.get("mask") or {}
    rays_node = loaded.get("rays") or {}
    occlusion = rays_node.get("occlusion") or {}
    gt_node = loaded.get("gt_cache") or {}

    dataset_root = Path(loaded.get("dataset_root", DATA_ROOT / "raw" / "kitti360"))
    lidar_root = Path(
        loaded.get("lidar_root", "/data-20T/kitti/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points")
    )
    bboxes_xml = Path(
        loaded.get(
            "bboxes_xml",
            "/data-20T/kitti/data_3d_bboxes/train/2013_05_28_drive_0000_sync.xml",
        )
    )
    artifacts = loaded.get("artifacts") or {}
    phase_dir = Path(artifacts.get("phase_dir", DATA_ROOT / "artifacts" / "02"))
    bundle_dir = Path(artifacts.get("bundle_dir", DATA_ROOT / "exports" / "reconstruct-bundle"))

    selection = SelectionOptions(
        window_frames=int(selection_node.get("window_frames", 60)),
        step_frames=int(selection_node.get("step_frames", 10)),
        min_usable_frames=int(selection_node.get("min_usable_frames", 60)),
        min_scene_points_per_frame=int(selection_node.get("min_scene_points_per_frame", 20000)),
    )
    mask = MaskOptions(
        dynamic_classes=tuple(mask_node.get("dynamic_classes", ()) or ()),
        dynamic_label_ids=tuple(int(v) for v in mask_node.get("dynamic_label_ids", (26, 27, 28, 32, 33, 24, 25))),
        ego_roi_x=_as_float_pair((mask_node.get("ego_roi_velodyne_m") or {}).get("x"), (-2.5, 2.5)),
        ego_roi_y=_as_float_pair((mask_node.get("ego_roi_velodyne_m") or {}).get("y"), (-1.2, 1.2)),
        ego_roi_z=_as_float_pair((mask_node.get("ego_roi_velodyne_m") or {}).get("z"), (-2.0, 0.3)),
        bbox_max_time_gap_frames=int(mask_node.get("bbox_max_time_gap_frames", 2)),
        residual_dynamic_radius_m=float(mask_node.get("residual_dynamic_radius_m", 1.0)),
    )
    rays = RaysOptions(
        t_min_m=float(rays_node.get("t_min_m", 1.0e-4)),
        r_min_m=float(rays_node.get("r_min_m", 1.5)),
        r_max_m=float(rays_node.get("r_max_m", 60.0)),
        image_margin_px=int(rays_node.get("image_margin_px", 5)),
        occlusion_neighbor_px=int(occlusion.get("neighbor_px", 3)),
        occlusion_sigma_lidar_m=float(occlusion.get("sigma_lidar_m", 0.02)),
        occlusion_sigma_multiple=float(occlusion.get("sigma_multiple", 3.0)),
        occlusion_depth_threshold_m=float(occlusion.get("depth_threshold_m", 0.06)),
    )
    expected_range = gt_node.get("expected_points_range", [50000, 150000])
    gt_cache = GtCacheOptions(
        min_points_per_frame=int(gt_node.get("min_points_per_frame", 20000)),
        expected_points_range=(int(expected_range[0]), int(expected_range[1])),
        camera_unknown=int(gt_node.get("camera_unknown", 255)),
        class_unknown=int(gt_node.get("class_unknown", 65535)),
    )

    frames_value = loaded.get("frames")
    frames = tuple(int(v) for v in frames_value) if frames_value else ()

    return Phase2Config(
        schema_version=int(loaded.get("schema_version", 1)),
        sequence=str(loaded.get("sequence", DEFAULT_SEQUENCE)),
        dataset_root=dataset_root,
        lidar_root=lidar_root,
        bboxes_xml=bboxes_xml,
        cameras=tuple(str(v) for v in (loaded.get("cameras") or ("image_00", "image_01"))),
        phase_dir=phase_dir,
        bundle_dir=bundle_dir,
        selection=selection,
        mask=mask,
        rays=rays,
        gt_cache=gt_cache,
        frames=frames,
        config_sha256=sha256_file(config_path),
        raw=loaded,
    )
