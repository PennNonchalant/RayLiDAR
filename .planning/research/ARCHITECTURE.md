# Architecture Research

**Domain:** Mesh-only KITTI-360 LiDAR simulation baseline with GT-defined-ray evaluation
**Researched:** 2026-10-30
**Confidence:** HIGH for KITTI-360 official data layout/frames and the stage/artifact contracts; MEDIUM for the concrete mesh-reconstruction route, the locally available dataset variant, and P2 GPU implementation details (all are flagged below).

## Architecture in One Paragraph

Build a stage-oriented offline pipeline with five hard boundaries: **prepare → reconstruct → build BVH → cast GT rays → evaluate/report**. Every boundary is a versioned artifact on disk with a manifest and content hash. The GT ray set is generated once and frozen; every representation (mesh now, Gaussian/router later) is cast against the same rays and judged by the same evaluator. The mesh is reconstructed by a route-specific adapter (COLMAP MVS / OpenMVS / learning-based) and normalized into one canonical `MeshArtifact` in a segment-local metric frame (`SEG`). The CPU BVH implements a flat, cache-friendly array layout and a Möller–Trumbore intersector behind a `SceneBackend` protocol so P2 can add a GPU backend and a routed hybrid backend without touching data prep, ray generation, metrics, or reporting.

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        INPUT / ARTIFACT LAYER                                │
│  KITTI-360 raw (read-only)          processed/<seq>/        artifacts/<run>/  │
│  calibration, poses, images,        calib, frames, images,  mesh, bvh, pred  │
│  velodyne, 2D/3D semantics,         masks, gt rays,         metrics, plots   │
│  3D bboxes                          colmap model                            │
└───────────────┬──────────────────────────────┬───────────────────────────────┘
                │                              │
┌───────────────▼──────────────────────────────▼───────────────────────────────┐
│                        CORE LIBRARY LAYER                                    │
│  hls.common.frames   hls.common.calib   hls.common.io   hls.common.geometry  │
│  (T_AB algebra, SEG) (KITTI loaders)    (PLY/NPZ/JSON)  (BVH, MT, KD-tree)   │
└───────────────┬──────────────────────────────┬───────────────────────────────┘
                │                              │
┌───────────────▼──────────────┐  ┌────────────▼───────────────────────────────┐
│  DATA PREP                   │  │  RECONSTRUCTION                            │
│  segment select              │  │  route adapter: COLMAP MVS / OpenMVS /     │
│  dynamic/ego masks           │  │  learning-based                            │
│  static GT + fixed GT rays   │  │  ingest → canonical mesh in SEG            │
│  COLMAP-format export        │  │  mesh manifest                             │
└───────────────┬──────────────┘  └────────────┬───────────────────────────────┘
                │                              │
                │                     ┌────────▼────────┐
                │                     │  BVH BUILD      │
                │                     │  flat CPU BVH   │
                │                     │  .bvh + manifest│
                │                     └────────┬────────┘
                │                              │
┌───────────────▼──────────────────────────────▼───────────────────────────────┐
│  CAST                                                                        │
│  GT rays (fixed) + SceneBackend (CPU Möller–Trumbore now; GPU/router later)  │
│  → PredictionBundle (t_pred, hit, prim_id, bary, normal, served_by)          │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  EVALUATE / REPORT                                                           │
│  range MAE/RMSE/MedAE · CD-L1/L2 · F@5/20/50 cm · P/R · miss · buckets     │
│  FPS/rays/s/memory · metrics.json/csv · plots · report.md                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `hls.common.frames` | Frame definitions, `T_AB` algebra, pose lookup, rectification, SEG anchor, transform validation | NumPy float64 4×4 matrices; explicit `T_dst_src` naming |
| `hls.common.calib` | Load KITTI-360 `calib_cam_to_pose.txt`, `calib_cam_to_velo.txt`, `perspective.txt`; build per-frame `T_C_V`; assert against `cam0_to_world.txt` | Official kitti360Scripts-compatible parsing |
| `hls.common.io` | Deterministic PLY/OBJ/NPZ/JSON/CSV read-write, SHA-256, manifest helpers | `numpy`, `open3d` for PLY/OBJ, stdlib `json`/`csv` |
| `hls.data.kitti360` | Locate sequences; read images, velodyne `.bin`, timestamps, poses, semantics; layout-variant detection | Config-driven root; fail loudly on missing files |
| `hls.data.segment` | Score candidate windows for low dynamic content, camera coverage, parallax; freeze selection | Semantic/bbox counts; no metric-based cherry-picking |
| `hls.data.masks` | Dynamic and ego filtering for GT; optional 2D semantic masks for recon | 3D bbox + 2D semantic + ego crop; counts logged |
| `hls.data.gt_rays` | Static GT extraction, camera eligibility, primary-camera assignment, fixed ray set, ray manifest/hash | Per-frame/per-camera NPZ; deterministic ordering |
| `hls.recon.base` | `MeshReconstructor` protocol; `MeshArtifact` dataclass; canonicalization contract | Route-agnostic interface |
| `hls.recon.export_colmap` | Convert KITTI-360 known poses + rectified images to COLMAP TXT model in SEG | PINHOLE cameras; world-to-camera `images.txt` |
| `hls.recon.routes.colmap_mvs` | Known-pose feature matching/triangulation + COLMAP dense MVS + meshing | `feature_extractor`, `sequential_matcher`, `point_triangulator`, `patch_match_stereo`, `stereo_fusion`, `poisson_mesher` |
| `hls.recon.routes.openmvs` | Optional alternative MVS/mesh route | OpenMVS CLI behind the same adapter |
| `hls.recon.ingest` | Validate metric scale, bbox, orientation, triangle count; transform to SEG; decimate; write manifest | Open3D + NumPy; reject non-metric or empty meshes |
| `hls.cast.rays` | Load fixed GT rays; transform camera-frame rays to SEG; batch/chunk | Pure NumPy; no Python per-ray loop |
| `hls.cast.bvh` | Build/serialize flat BVH; precompute triangle edges; AABB traversal | C++17 + pybind11; SoA arrays; deterministic build |
| `hls.cast.backends.cpu` | Möller–Trumbore closest-hit kernel; batch API; timings | C++/OpenMP; same routine as future CUDA |
| `hls.cast.pipeline` | Orchestrate cast; produce `PredictionBundle`; map hits back to LiDAR frame | Backend-agnostic |
| `hls.eval.metrics` | Range errors, CD, F-score, P/R, miss rate, buckets, per-frame/per-camera aggregation | NumPy + `scipy.spatial.cKDTree` |
| `hls.eval.report` | Metrics tables, JSON/CSV, plots, generated report | Matplotlib; deterministic output |
| `hls.pipeline` | Run IDs, config resolution, manifests, caching, stage orchestration | CLI; idempotent stages |

### Recommended Project Structure

```
hybrid-lidar-sim/
├── README.md
├── pyproject.toml                  # package + pybind11 extension build
├── cpp/
│   ├── include/hls/bvh.hpp         # flat BVH layout
│   ├── include/hls/moller_trumbore.hpp
│   ├── src/bvh_cpu.cpp
│   ├── src/bindings.cpp
│   └── CMakeLists.txt
├── configs/
│   ├── dataset.yaml                # roots, sequence, frame window, cameras
│   ├── recon_colmap.yaml           # route, matcher, MVS/meshing params
│   ├── cast_cpu.yaml               # backend, t_min/t_max, threads, two_sided
│   └── eval.yaml                   # range, buckets, tau, metric definitions
├── src/hls/
│   ├── common/
│   │   ├── frames.py               # T_AB, SEG, pose lookup
│   │   ├── calib.py                # KITTI-360 calibration loading
│   │   ├── io.py                   # hashes, manifests, PLY/NPZ/JSON/CSV
│   │   └── geometry.py             # projections, AABB, quaternion, transforms
│   ├── data/
│   │   ├── layout.py               # official vs variant path resolution
│   │   ├── kitti360.py             # sequence/frame discovery
│   │   ├── segment.py              # candidate scoring + selection
│   │   ├── masks.py                # dynamic/ego masks
│   │   └── gt_rays.py              # fixed GT ray set
│   ├── recon/
│   │   ├── base.py                 # MeshReconstructor protocol
│   │   ├── export_colmap.py        # known-pose COLMAP model
│   │   ├── routes/
│   │   │   ├── colmap_mvs.py
│   │   │   └── openmvs.py
│   │   ├── ingest.py               # canonical mesh + validation
│   │   └── postprocess.py          # decimate, clean, normals, bbox
│   ├── cast/
│   │   ├── rays.py
│   │   ├── bvh.py
│   │   ├── backend.py              # SceneBackend protocol, RayBatch/HitBatch
│   │   ├── backends/
│   │   │   ├── cpu.py
│   │   │   └── open3d_oracle.py    # tests only, not a result backend
│   │   └── pipeline.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── pointcloud.py
│   │   ├── buckets.py
│   │   └── report.py
│   └── pipeline/
│       ├── run.py
│       ├── manifest.py
│       └── cache.py
├── scripts/
│   ├── 00_check_dataset.py
│   ├── 01_select_segment.py
│   ├── 02_prepare.py
│   ├── 03_reconstruct.py
│   ├── 04_cast.py
│   ├── 05_evaluate.py
│   ├── 06_report.py
│   └── run_all.sh
├── tests/
│   ├── test_frames.py
│   ├── test_gt_rays.py
│   ├── test_bvh.py
│   ├── test_metrics.py
│   └── test_end_to_end_synthetic.py
├── docs/
│   ├── frames.md                   # normative coordinate-frame reference
│   ├── protocol.md                 # GT-ray protocol + metric definitions
│   └── repro.md                    # exact rerun commands and hashes
├── artifacts/                      # large caches; gitignored
└── results/                        # small committed outputs: metrics, plots, report
```

### Structure Rationale

- **`src/hls/` as one installable package:** avoids top-level packages named `common`, `data`, `eval` that can collide; imports are explicit (`from hls.common.frames import ...`). The README §8 skeleton (`src/recon`, `src/infer`, `src/eval`, `src/common`) is preserved in spirit but nested under one package.
- **`cpp/` outside the Python package:** the CPU BVH and the future CUDA BVH share headers (`bvh.hpp`, `moller_trumbore.hpp`). P2 adds `bvh_cuda.cu`; it does not restructure Python.
- **`artifacts/` vs `results/`:** processed data, meshes, BVHs, and per-ray predictions are large and cacheable; metrics JSON/CSV, plots, and the report are small and should be committed for review.
- **Per-stage configs:** dataset, reconstruction, casting, and evaluation change at different rates. A resolved config is dumped into every run manifest.
- **`tests/` with a synthetic end-to-end test:** the evaluation and cast correctness can be validated before any real mesh exists.

---

## Architectural Patterns

### Pattern 1: Stage contracts with canonical artifacts

**What:** Each stage reads named artifacts and writes named artifacts plus a manifest. No stage reaches into another stage's internals or mutable state.
**When to use:** Always in this project; reconstruction and evaluation have different failure modes and need independent reruns.
**Trade-offs:** A little more disk I/O and schema boilerplate; huge payoff in debuggability and reproducibility.

```python
# Every stage writes a manifest with the same fields.
manifest = {
    "schema_version": 1,
    "stage": "gt_rays",
    "run_id": "0000_colmap_20261030_ab12cd3",
    "config_hash": "...",
    "input_hashes": {...},
    "output_hashes": {...},
    "git_commit": "...",
    "dirty": False,
    "created_utc": "...",
}
```

### Pattern 2: Backend-agnostic `SceneBackend` protocol

**What:** Casting is expressed as a batch operation over rays in SEG. Mesh, Gaussian, and routed-hybrid backends all implement the same protocol.
**When to use:** From day one; the CPU backend is the only implementation in v1.0.
**Trade-offs:** The protocol forces a copy of ray data into SEG and a hit result with a few extra fields. That is cheap and is exactly what P2 needs.

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class RayBatch:
    ray_id: np.ndarray        # (N,) int64, stable ID from GT prep
    origin: np.ndarray        # (N, 3) float32, SEG
    direction: np.ndarray     # (N, 3) float32, unit length, SEG
    t_min: float
    t_max: float

@dataclass
class HitBatch:
    ray_id: np.ndarray        # (N,)
    t: np.ndarray             # (N,) float32, +inf on miss
    hit: np.ndarray           # (N,) bool
    prim_id: np.ndarray       # (N,) int32, -1 on miss
    bary: np.ndarray          # (N, 2) float32
    normal: np.ndarray        # (N, 3) float32, SEG
    served_by: np.ndarray     # (N,) uint8: 0=mesh, 1=gaussian, 2=router fallback

class SceneBackend(Protocol):
    name: str
    def intersect(self, rays: RayBatch) -> HitBatch: ...
```

`served_by` is a one-field P2 seam: v1.0 always writes `0`; P2 fills it per ray and the evaluator can report fallback rate without schema changes.

### Pattern 3: Fixed GT-ray set as the evaluation unit

**What:** Generate the ray set once from static LiDAR GT + calibration + camera eligibility. Freeze it with a content hash. Every representation is evaluated on exactly this set.
**When to use:** Always; this is the primary protocol in README §1.3 and the milestone.
**Trade-offs:** Changing eligibility rules requires regenerating the hash and re-running all comparisons; that is intentional.

```python
# ray_id is stable and human-debuggable.
ray_id = encode(frame_id, cam_id, lidar_point_index)
# Frozen arrays per frame/camera:
#   dir_C (N,3) float32 unit, t_gt (N,) float64, p_L (N,3) float32,
#   bucket (N,) int8, eligible_cams (N,) uint8 bitmask
```

### Pattern 4: Reconstruction-route adapter

**What:** Each mesh route is a `MeshReconstructor` that consumes images + known poses + optional 2D masks and returns a raw mesh. A single `ingest` step validates and canonicalizes it.
**When to use:** v1.0 must choose COLMAP MVS / OpenMVS / learning-based without letting that choice leak into cast/eval.
**Trade-offs:** Route-specific quirks (COLMAP image IDs, OpenMVS CLI flags, monocular scale) stay in the adapter. The downstream contract is stable.

```python
class MeshReconstructor(Protocol):
    name: str
    def reconstruct(self, ctx: ReconContext) -> Path: ...  # raw mesh path

# recon.ingest validates and writes:
#   artifacts/<run>/mesh/scene.ply
#   artifacts/<run>/mesh/mesh_manifest.json  (frame=SEG, units=m, hashes, provenance)
```

### Pattern 5: Segment-local metric frame (SEG)

**What:** A per-segment frame with the same orientation as KITTI world `W` and origin at the median camera center of the selected window. Mesh and BVH live in SEG.
**When to use:** Always; KITTI-360 world coordinates are global and can be kilometric, which wastes float32 precision.
**Trade-offs:** Every frame/camera needs `T_S_C` and every mesh needs `T_S_W`. This is a small, explicit cost and makes artifacts portable across sequences.

### Pattern 6: Manifest + content hashes

**What:** Every artifact has a SHA-256 and a manifest with config, code version, environment, and input hashes.
**When to use:** Always. Reconstruction itself may not be bitwise reproducible (multi-threaded MVS), but evaluation must be exactly reproducible from an archived mesh.
**Trade-offs:** Hashing large meshes/rays costs seconds; worth it.

### Pattern 7: Flat-array BVH shared by CPU and GPU

**What:** Build a BVH once as flat arrays (nodes, triangle indices, precomputed triangle edges) and traverse it with a Möller–Trumbore kernel. The same layout can be uploaded to a GPU in P2.
**When to use:** v1.0 CPU reference; P2 GPU.
**Trade-offs:** A flat BVH is slightly more code than `trimesh`/Open3D one-liners, but it is the artifact P2 needs and it makes correctness testing tractable.

---

## Coordinate Frames & Calibration (Critical)

This section is the normative reference for `docs/frames.md`. Every transform in code must follow it.

### Notation

Use `T_AB` as a 4×4 homogeneous transform mapping points from frame **B** to frame **A**:

```
p_A = T_AB @ p_B
```

The README writes `p_C = T_CL p_L`. Interpret that as `T_CL ≡ T_C←L` ("camera from LiDAR"). Do **not** use bare `T_CL`, `Tr(cam→velo)` and `T_V_C` interchangeably in code; KITTI's textual names use the opposite word order in places.

### Frame Definitions

| Symbol | Frame | Axes | Source | Notes |
|--------|-------|------|--------|-------|
| `V` | Velodyne HDL-64E | x forward, y left, z up | `data_3d_raw/.../velodyne_points/data/*.bin` | Raw LiDAR points are in `V`; 4 floats per point `(x,y,z,intensity)` |
| `Ck_raw` | Perspective camera k (k=0,1) | x right, y down, z forward | `calib_cam_to_pose.txt` row k | Unrectified camera frame; `calib_cam_to_velo.txt` is only k=0 |
| `Ck_rect` | Rectified perspective camera k | same as `Ck_raw` | `R_rect_k` from `perspective.txt` | `p_Ck_rect = R_rect_k @ p_Ck_raw` |
| `Ck_raw` | Fisheye camera k (k=2,3) | k=2: x forward, y down, z left; k=3: x backward, y down, z right | `calib_cam_to_pose.txt`, `image_0k.yaml` | Different axes and projection model; not used by default in v1.0 |
| `I` | GPS/IMU | x forward, y right, z down | `poses.txt` | The official docs occasionally say "GPU/IMU"; it is GPS/IMU |
| `W` | KITTI-360 world | x forward, y left, z up | Global origin shared across sequences | Used by 3D semantics/bboxes and `cam0_to_world.txt` |
| `S` | Segment frame (SEG) | same orientation as `W` | Derived: translation-only from `W` | Origin at median camera center; mesh/BVH live here |
| `Ek` | Evaluation camera | `Ck_rect` for perspective; `Ck_raw` for fisheye | Derived | Ray origin is the camera optical center; directions are in `Ek` |

### Transform Chain

Let `R_rect4(k)` be the 4×4 matrix with `R_rect_k` in the top-left and identity elsewhere, and `inv()` be a 4×4 inverse. Load KITTI-360 files as 3×4 matrices, append `[0,0,0,1]`.

```python
# Primitives from the official KITTI-360 calibration files.
T_V_C0   = load_calib_cam_to_velo()        # p_V = T_V_C0 @ p_C0_raw
T_I_Ck   = load_calib_cam_to_pose()        # p_I = T_I_Ck[k] @ p_Ck_raw  (k=0..3)
T_I_C0   = T_I_Ck[0]                       # camera 0 raw -> IMU
P_rect_k = load_perspective()[f"P_rect_{k:02d}"]
R_rect_k = load_perspective()[f"R_rect_{k:02d}"]   # p_Ck_rect = R_rect_k @ p_Ck_raw (k=0,1)
K_rect_k = P_rect_k[:3, :3]                # single-camera intrinsics; ignore P[:,3] baseline
T_W_I    = load_poses()[frame]             # p_W = T_W_I @ p_I

# Camera k -> Velodyne, composed through camera 0.
T_C0_Ck  = inv(T_I_C0) @ T_I_Ck[k]         # p_C0 = T_C0_Ck @ p_Ck
T_V_Ck   = T_V_C0 @ T_C0_Ck                # p_V  = T_V_Ck  @ p_Ck
T_Ck_V   = inv(T_V_Ck)

# Rectification applies on the camera side.
T_Ck_rect_V = R_rect4(k) @ T_Ck_V          # k=0,1 only

# World <- Velodyne and World <- rectified camera.
T_W_V       = T_W_I @ T_I_C0 @ inv(T_V_C0)
T_W_Ck_rect = T_W_I @ T_I_Ck[k] @ inv(R_rect4(k))   # k=0,1
# Equivalent: T_Ck_rect_V = inv(T_W_Ck_rect) @ T_W_V == R_rect4(k) @ T_Ck_V

# Segment frame: translation only, orientation identical to W.
a_W = median([T_W_Ck_rect[:3, 3] for selected frames/cameras])
T_S_W = np.eye(4); T_S_W[:3, 3] = -a_W
T_W_S = np.eye(4); T_W_S[:3, 3] =  a_W
T_S_V       = T_S_W @ T_W_V
T_S_Ck_rect = T_S_W @ T_W_Ck_rect
```

**Per-frame, per-camera evaluation transform:**

```python
T_Ck_rect_V = inv(T_W_Ck_rect) @ T_W_V
```

**Cross-check against the dataset's direct file:**

```python
# cam0_to_world.txt is T_W_C0_rect, for the rectified camera 0.
assert_allclose(T_W_C0_rect, load_cam0_to_world()[frame], atol=1e-3)
```

Use `poses.txt` + `calib_cam_to_pose.txt` + `R_rect` as the source of truth for all cameras; use `cam0_to_world.txt` only as an independent assertion. This mirrors the official `kitti360Scripts` behavior.

### Rectification: The Most Common Integration Bug

- `calib_cam_to_pose.txt` rows for `image_00`/`image_01` refer to **unrectified** perspective cameras.
- `perspective.txt` provides `R_rect_00`/`R_rect_01`; `data_2d_raw/.../image_00/data_rect` contains **rectified** images.
- Therefore:
  - Projection: `p_Crect = R_rect @ (T_Ck_V @ p_V)`.
  - World pose: `T_W_Crect = T_W_I @ T_I_Craw @ inv(R_rect)`.
  - Never mix `data_rgb` (raw) with `P_rect`/`R_rect`.
- `P_rect` is a 3×4 projection matrix. Its fourth column encodes the stereo baseline; for a single-camera projection use `K = P_rect[:3,:3]`, not the full 3×4 with the baseline term.
- If only raw images are available (e.g., the local tree's `image_00/data`), rectify/undistort them before MVS using `R_rect`/`P_rect` and any distortion coefficients present in the calibration file. Do not feed raw distorted images to a PINHOLE MVS model.

### Pose Lookup, Missing Frames, and Timestamps

- `poses.txt` contains only frames where the platform moved more than a threshold; frame indices may be **non-contiguous**. Build `pose_by_frame`; for an image frame `f` missing from the pose table, choose the nearest valid `f'` and record `pose_frame=f'`, `pose_delta=f-f'`. Do not silently use frame 0.
- `cam0_to_world.txt` has its own frame list; use it for assertions only.
- `timestamps.txt` files are per sensor; `_sync` sequences are nominally aligned but sensor timestamps differ. Record the image and LiDAR timestamps for every frame.
- KITTI-360 raw velodyne scans are captured over a rotation period; the official toolkit provides `curlVelodyneData` to compensate ego motion. v1.0 should treat raw points as GT and record the limitation; optionally evaluate a motion-compensated variant as an ablation. Do not mix raw and curled points across representations.
- If a frame is missing an image, a velodyne scan, or a pose, drop the frame from both recon and eval and record the reason. Never let stages silently use different frame sets.

### Ray Construction and the Metric Frame

For each static LiDAR GT point `p_L` in `V`, an eligible camera `k`, and frame `f`:

```python
p_C = T_Ck_rect_V @ [p_L, 1]          # camera-frame GT point
r_gt = norm(p_C[:3])                  # camera-origin GT range (NOT raw LiDAR range)
d_C = p_C[:3] / r_gt                  # unit direction in Ek
o_C = [0, 0, 0]                       # camera optical center

# Cast in SEG.
T_S_C = T_S_W @ T_W_Ck_rect
o_S = T_S_C[:3, 3]
d_S = T_S_C[:3, :3] @ d_C             # unit length preserved
# t is metric range because d_S is unit.
p_pred_S = o_S + t_pred * d_S
p_pred_V = inv(T_S_V) @ [p_pred_S, 1] # unified LiDAR frame for CD/F-score
```

Protocol details:

- `r_gt = ||p_C||` is the **camera-origin range** along the GT-defined ray. It differs from the raw LiDAR range `||p_L||` by at most the camera–LiDAR baseline (roughly 0.1–0.8 m). Use `r_gt` for range errors and distance buckets; record `range_lidar = ||p_L||` alongside for diagnostics.
- `t_pred = +inf` is a miss. Do not include misses in range MAE/RMSE/MedAE; count them in miss rate and ray recall.
- `p_pred_V` is used for Chamfer distance, F-score, precision, and recall so that predicted and GT points share one coordinate system.
- If a GT point is eligible in multiple cameras, assign a deterministic `primary_cam` (e.g., smallest normalized image radius, tie → lower camera ID) for the main metric. Store the full `eligible_cams` bitmask and report per-camera counts. This avoids double-counting one LiDAR point.
- Distance buckets use `r_gt`: `[0,20)`, `[20,40)`, `[40,50]` for the primary 0–50 m protocol; `[40,60]` only if `t_max` is extended to 60 m. Bucket edges are fixed in `eval.yaml` and recorded in the metrics manifest.

### Calibration Validation Checklist (Run Before Any Reconstruction)

1. `T_AB @ T_BA ≈ I` for every composed transform.
2. `T_W_C0_rect` from poses+calib matches `cam0_to_world.txt` within `atol=1e-3`.
3. Project a sample of velodyne points into `image_00/data_rect` and visually confirm alignment (road edges, poles, building corners).
4. `R_rect` is orthonormal (`R @ R.T ≈ I`, `det ≈ 1`); `K` has positive focal lengths and a principal point inside the image.
5. Camera centers form a continuous trajectory in `W`; no frame jumps by more than a few meters at 10 Hz.
6. `abs(r_gt - ||p_L||) <= ||t_C_V|| + 1e-3` for all rays.
7. Bucket counts (`0–20`, `20–40`, `40–50`) are non-zero and roughly plausible for a driving scene.
8. Compare `T_W_V` to the OXTS trajectory (if available) for gross errors; do not use OXTS to replace official poses.

### Frame Pitfalls

- **Ambiguous `T_CL`:** Define it once as `T_C←L`; in code use `T_dst_src` or explicit names.
- **Raw vs rectified:** The most likely source of a 5–20 px systematic offset.
- **KITTI raw vs KITTI-360 calibration:** `calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt` are KITTI-raw-style names, not the KITTI-360 files (`calib_cam_to_pose.txt`, `calib_cam_to_velo.txt`, `perspective.txt`). Do not mix them without proving equivalence.
- **World-frame float32:** KITTI world coordinates can be kilometric; store mesh in SEG, transforms in float64.
- **Fisheye axes:** `image_02` and `image_03` do not share the perspective camera axes; do not reuse a pinhole `K` or the same ray formula.
- **`P_rect` baseline:** Using the full 3×4 matrix for a single camera adds a spurious shift.
- **Missing poses:** Silent fallback to frame 0 or nearest frame without logging invalidates metrics.

---

## Data Flow

**v1.0 scope:** one low-dynamic KITTI-360 segment, static geometry only, one mesh representation, CPU cast, 0–50 m primary range. Dynamic objects are filtered from GT and, where the route supports it, from reconstruction; no dynamic scene graph, no Gaussian representation.

**Default camera set:** `image_00` and `image_01` (rectified perspective). `image_02`/`image_03` fisheye cameras are deferred; the `CameraModel` seam exists but is not implemented in v1.0. The reconstruction and evaluation camera sets must be identical.

### End-to-End Pipeline

```
[0] check_dataset
    raw KITTI-360 → readiness report (missing files, sequence list, frame counts)

[1] prepare
    raw + config → processed/<seq>/
      calib.npz / calib_manifest.json
      frames.csv
      images/ (hardlinks or symlinks to rectified image_00/01)
      masks/dynamic/, masks/ego/
      gt/rays/<frame>_<cam>.npz, gt/rays_manifest.json
      recon/colmap/{cameras.txt,images.txt,points3D.txt}, recon/image_list.txt

[2] reconstruct
    processed images + poses + optional 2D masks
      → recon route (COLMAP MVS / OpenMVS / learning)
      → raw mesh
      → ingest/postprocess
      → artifacts/<run>/mesh/scene.ply + mesh_manifest.json

[3] build_bvh
    mesh → artifacts/<run>/bvh/scene.bvh + bvh_manifest.json

[4] cast
    gt rays + mesh + bvh + backend
      → artifacts/<run>/pred/<frame>.npz + cast_manifest.json

[5] evaluate
    gt rays + predictions
      → artifacts/<run>/metrics/metrics.json
      → artifacts/<run>/metrics/per_frame.csv
      → artifacts/<run>/metrics/per_bucket.csv
      → artifacts/<run>/metrics/run_manifest.json

[6] report
    metrics + manifests → results/plots/*.png, results/report.md, results/tables/*.csv
```

### Artifact Contracts

| Artifact | Format | Required Keys / Fields | Producer → Consumer |
|----------|--------|------------------------|---------------------|
| `CalibrationBundle` | `calib.npz` + `calib_manifest.json` | `T_V_C0`, `T_I_Ck[4]`, `R_rect_k[2]`, `K_rect_k[2]`, `image_size_k[4]`, `T_W_I[F]`, `T_Ck_rect_V[F,K]`, `T_S_W`, `T_S_Ck_eval[F,K]` (rect for perspective, raw for fisheye), `T_S_V[F]` | prepare → recon/cast/eval |
| `FrameIndex` | `frames.csv` | `frame_id`, image/LiDAR timestamps, `pose_frame`, `pose_delta`, `dynamic_fraction`, `in_recon`, `in_eval` | prepare → all |
| `GtRaySet` | `gt/rays/<frame>_<cam>.npz` + `gt/rays_manifest.json` | `ray_id`, `lidar_idx`, `dir_C`, `t_gt`, `p_L`, `p_C`, `bucket`, `eligible_cams`, `primary_cam`, `semantic_id`, `range_lidar`; manifest: counts, filters, hash, assignment policy | prepare → cast/eval |
| `ColmapModel` | `cameras.txt`, `images.txt`, `points3D.txt` | PINHOLE cameras; world-to-camera poses in SEG; empty `points3D` before triangulation | prepare → recon |
| `MeshArtifact` | `scene.ply` + `mesh_manifest.json` | frame=`SEG`, units=`m`, vertices/faces, bbox, `n_vertices`, `n_triangles`, route/params, `T_S_W`, input hashes | recon → cast |
| `BvhArtifact` | `scene.bvh` + `bvh_manifest.json` | version, node count, triangle order, `t_min/t_max`, mesh hash, build time | cast |
| `PredictionBundle` | `pred/<frame>.npz` + `cast_manifest.json` | `ray_id`, `t_pred` (inf=miss), `hit`, `prim_id`, `bary`, `normal_S`, `served_by`, `backend`; timings | cast → eval |
| `MetricsArtifact` | `metrics.json`, `per_frame.csv`, `per_bucket.csv` | definitions, counts, aggregate, buckets, per-camera, per-frame, timings, hashes | eval → report |
| `RunManifest` | `run_manifest.json` | schema version, run ID, config hash, git commit, dirty flag, environment, wall time, peak RSS | pipeline → all |

Use `schema_version: 1` in every JSON manifest. JSON must not contain `NaN`/`Inf`; use `null` or strings. NPZ may contain `+inf` for misses.

### State and Caching

- A run ID is `<seq>_<route>_<date>_<gitsha>`; the resolved config is dumped verbatim.
- Each stage is idempotent: if the output manifest's `config_hash` and input hashes match, skip; `--force` rebuilds.
- The GT ray set is generated once and never regenerated during a comparison; its hash is the key for all evaluation runs.
- The mesh artifact is the cache boundary: evaluation reproducibility is **bitwise from the archived mesh**; reconstruction reproducibility is **approximate** (same route/params, possibly different MVS threading/meshing output). State this explicitly in the report.
- Raw KITTI-360 is read-only. Never write into the dataset root.

### Key Data Flows

1. **Calibration → everything:** `T_Ck_rect_V`, `T_S_Ck_rect`, `K` are used by GT ray generation, cast, and evaluation. One wrong transform corrupts every metric.
2. **GT rays → cast → eval:** The ray set is immutable; `t_gt`, `p_L`, `bucket`, `ray_id` flow through predictions and metrics.
3. **Mesh → BVH → cast:** The BVH is built once from the canonical mesh; all frames reuse it.
4. **Predictions + GT → metrics:** Range errors use `t_pred - t_gt`; CD/F-score use `p_pred_V` vs `p_L` per frame.
5. **Future router:** The router is a `SceneBackend` that runs mesh first and dispatches misses/low-confidence rays to a Gaussian backend; the `PredictionBundle` already carries `served_by` and `ray_id`, so eval/report do not change.

### Evaluation Contract (freeze in `docs/protocol.md` before running the real mesh)

- **Range (hits only):** `e_i = |t_pred_i - t_gt_i|`; `MAE = mean(e)`, `RMSE = sqrt(mean(e²))`, `MedAE = median(e)`; also report signed bias and p95.
- **Miss / ray recall:** `t_pred = +inf` on miss; `miss_rate = n_miss / n_eligible`; `ray_recall = n_hit / n_eligible`. Misses never enter range errors.
- **Point cloud (per frame in `V`):** `P = {p_pred_V for hits}`, `G = {p_L}`.
  - Primary definition: `CD-L1 = 0.5 · (mean_{p∈P} min_{g∈G} ||p−g||_1 + mean_{g∈G} min_{p∈P} ||g−p||_1)` (coordinate L1).
  - Primary definition: `CD-L2 = 0.5 · (mean_{p∈P} min_{g∈G} ||p−g||_2² + mean_{g∈G} min_{p∈P} ||g−p||_2²)` (squared Euclidean).
  - Some papers use Euclidean distance for "CD-L1". If comparing to them, label that variant `CD-E` and report it separately; do not silently change the primary definition.
- **F-score@τ (τ = 0.05 / 0.20 / 0.50 m):** `precision = |{p∈P : min_g ||p−g||_2 ≤ τ}| / |P|`; `recall = |{g∈G : min_p ||g−p||_2 ≤ τ}| / |G|`; `F = 2PR/(P+R)`.
- **Buckets:** by `r_gt`: `[0,20)`, `[20,40)`, `[40,50]` primary; `[40,60]` only for the 60 m extension. Report micro (pooled rays) and per-frame macro; per-camera and per-semantic-class splits are optional.
- **Throughput:** rays/s, per-frame latency, FPS, peak RSS, BVH build time, `#triangles`, `#rays`; fixed thread count and hardware; median of 3 runs after warm-up.

**Report deliverables (minimum):** per-bucket range RMSE + ray recall; error CDF/histogram; CD/F-score/precision/recall table at τ = 5/20/50 cm; throughput/memory/BVH table; per-frame error timeline; qualitative mesh-vs-GT overlay. v1.0 has one representation, so the README's CD-vs-FPS Pareto is a single point; still emit the same plot schema so P2/P3 can add points without changing the plotting code.

---

## New vs Modified Components

No source tree exists yet, so every code path is **new**. The table states how each path relates to the README §8 skeleton.

| Path | Status | Purpose |
|------|--------|---------|
| `src/hls/common/frames.py` | New (absorbs README `common/camera.py`, `common/lidar.py` frame logic) | Normative transforms, SEG, pose lookup |
| `src/hls/common/calib.py` | New | KITTI-360 calibration loaders + assertions |
| `src/hls/common/io.py`, `geometry.py` | New | PLY/NPZ/JSON/CSV, hashes, projections, BVH helpers |
| `src/hls/data/layout.py`, `kitti360.py` | New | Dataset discovery, official/variant layouts |
| `src/hls/data/segment.py` | New | Low-dynamic segment scoring and freeze |
| `src/hls/data/masks.py` | New | Dynamic/ego filtering and 2D recon masks |
| `src/hls/data/gt_rays.py` | New | Fixed GT ray set + manifest |
| `src/hls/recon/base.py` | New | `MeshReconstructor` protocol |
| `src/hls/recon/export_colmap.py` | New | Known-pose COLMAP TXT model in SEG |
| `src/hls/recon/routes/colmap_mvs.py` | New | COLMAP MVS + meshing route |
| `src/hls/recon/routes/openmvs.py` | New (optional fallback) | OpenMVS route |
| `src/hls/recon/ingest.py`, `postprocess.py` | New | Canonical mesh, validation, decimation |
| `src/hls/recon/train_gs.py`, `render_depth.py`, `tsdf_mesh.py`, `lod.py`, `residual_gs.py` | **Deferred** (README §8 entries) | 3DGS/2DGS/TSDF/residual paths; not in v1.0 |
| `src/hls/cast/rays.py` | New (replaces README `infer/rays.py`) | Load/transform/batch GT rays |
| `src/hls/cast/bvh.py` | New | Flat BVH build/serialize |
| `src/hls/cast/backend.py` | New | `SceneBackend`, `RayBatch`, `HitBatch` |
| `src/hls/cast/backends/cpu.py` + `cpp/` | New | CPU Möller–Trumbore reference |
| `src/hls/cast/pipeline.py` | New (replaces README `infer/pipeline.py`) | Cast orchestration |
| `src/hls/cast/mesh_raster.cu`, `gs_splat.cu`, `fused_raster.cu`, `bvh.cu` | **Deferred** (README §8 entries) | P2 GPU/raster/router kernels |
| `src/hls/eval/metrics.py`, `pointcloud.py`, `buckets.py` | New (replaces README `eval/metrics.cu` for CPU) | Full metric suite |
| `src/hls/eval/report.py` | New (README has `report.py`) | Tables, plots, report |
| `src/hls/eval/metrics.cu` | **Deferred** (README §8) | P2 GPU metrics |
| `src/hls/pipeline/` | New | Run IDs, manifests, caching, CLI |
| `configs/dataset.yaml`, `recon_colmap.yaml`, `cast_cpu.yaml`, `eval.yaml` | New/expanded from README `configs/kitti360.yaml` | Per-stage config |
| `scripts/00_check_dataset.py` … `scripts/06_report.py`, `run_all.sh` | New/expanded from README `scripts/` | Stage entry points |
| `tests/` | New | Unit + synthetic end-to-end tests |
| `docs/frames.md`, `protocol.md`, `repro.md` | New | Normative reference and rerun instructions |
| `.gitignore`, `pyproject.toml` | New | Ignore `artifacts/`, data roots; build package |
| `README.md` | Modify | Add v1.0 commands, artifact paths, and architecture pointer after implementation |

**Deferred means interface-only or not present.** Do not implement Gaussian training, TSDF, GPU kernels, or a router in v1.0. Leave the interfaces and artifact fields described above; that is the seam.

---

## Integration Points

### External Services / Tools

| Tool / Data | Integration Pattern | Notes |
|-------------|---------------------|-------|
| KITTI-360 raw dataset | Read-only mount; config `dataset.root` | Official layout expected: `calibration/`, `data_poses/`, `data_2d_raw/`, `data_3d_raw/`, `data_2d_semantics/`, `data_3d_semantics/`, `data_3d_bboxes/` |
| COLMAP CLI / pycolmap | Subprocess with pinned version; adapter writes TXT model in SEG | Known-pose path: `feature_extractor` → `sequential_matcher` → `point_triangulator` → `image_undistorter` → `patch_match_stereo` → `stereo_fusion` → `poisson_mesher`/`delaunay_mesher`. Do **not** run `mapper`/SfM. |
| OpenMVS CLI (optional) | Subprocess; `InterfaceCOLMAP` → `DensifyPointCloud` → `ReconstructMesh` → `RefineMesh` | Uses COLMAP workspace/sparse model; preserves SEG if exported there |
| Open3D | Python API for PLY/OBJ I/O, decimation, optional TSDF future; `RaycastingScene` as test oracle | **Not** the production cast backend in v1.0; use it to cross-check the custom BVH |
| SciPy `cKDTree` | In-process | CD/F-score/P/R nearest-neighbor queries |
| Matplotlib | In-process | Plots; deterministic settings |
| pybind11 / C++17 | Build extension via `pyproject.toml`/CMake | CPU BVH; headers shared with future CUDA |
| CUDA 12.x | Future P2 | Not required for v1.0 cast; may be required for COLMAP PatchMatch Stereo depending on build |
| Git | Commit small `results/`; ignore `artifacts/` | Record commit + dirty flag in run manifests |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `common.frames` ↔ `data` | Direct function calls; matrices | Single source of truth for `T_AB` |
| `data.prepare` ↔ `recon` | COLMAP TXT model + image list + optional masks | No raw KITTI paths inside `recon` |
| `recon` ↔ `cast` | `MeshArtifact` in SEG + manifest | Route-specific code stops at `ingest` |
| `data.gt_rays` ↔ `cast` | `GtRaySet` NPZ + manifest/hash | Fixed rays; no regeneration during comparison |
| `cast` ↔ `eval` | `PredictionBundle` + manifest | Evaluator never sees mesh/BVH internals |
| `eval` ↔ `report` | Metrics JSON/CSV + run manifest | Report is pure presentation |
| `cast.backends.cpu` ↔ future `gpu` | `SceneBackend` protocol; `RayBatch`/`HitBatch`; optional `scene.bvh` | No Python per-ray loops |
| `cast` ↔ future router | `HitBatch` cues: `hit`, `normal`, `prim_id`, optional per-face confidence/semantic; subset casting; `served_by` | Router must not use GT |
| `recon.mvs` ↔ future `2dgs_tsdf` | Same `MeshReconstructor`/`MeshArtifact` contract | Future route writes the same canonical mesh |
| `eval` ↔ future router | `PredictionBundle.served_by`; per-backend metric split | Fallback rate/oracle gap become report fields, not schema changes |
| `pipeline` ↔ all | Config + manifests + run ID | Idempotent stages; no hidden global state |

### Future Seams Explicitly Preserved

**P2 — GPU BVH and routed hybrid casting**

- `SceneBackend` is the only cast interface; `cast/backends/gpu.py` can be added without changing `ray gen`, `eval`, or `report`.
- `RayBatch`/`HitBatch` are batch-oriented and SoA; no Python per-ray loop.
- `cpp/include/hls/bvh.hpp` and `moller_trumbore.hpp` use flat arrays that can be uploaded to CUDA. P2 can also use a different GPU BVH (LBVH); the protocol only requires `intersect`.
- `HitBatch.served_by` and `PredictionBundle.served_by` reserve the per-ray backend label. v1.0 writes `0` (mesh).
- `HitBatch` carries `normal`, `prim_id`, `bary`; the future router can compute grazing angle, hit confidence, and normal discontinuity without a second mesh query.
- Optional `mesh_attrs.npz` (per-face confidence/semantic) is not required for v1.0 but is a natural place for P2 cues. If added, hash it in the mesh manifest.
- Tiling/occupancy/Morton ordering are P2 implementation details behind the backend. Do not add them to `cast/rays.py`.

**Later — 2DGS → TSDF → Marching Cubes**

- Add `hls/recon/routes/depth_tsdf.py` implementing `MeshReconstructor`.
- Define a future `DepthMap` artifact: `depth`, `confidence`, `K`, `T_C_W`, frame/cam. A `DepthSource` can be COLMAP MVS, monocular depth, or 2DGS rendered depth.
- The route produces the same `MeshArtifact` in SEG; `ingest` validates it; cast/eval are unchanged.
- Do not implement TSDF/MC or 2DGS training now.

**P3 — True hybrid representation**

- `mesh_manifest.json` can later grow a `representations: ["mesh", "gaussians"]` field (or a sibling `hybrid_manifest.json`).
- A `HybridScene` can implement `SceneBackend` as a single scene; eval is unchanged.
- Do not implement a hybrid representation now.

**Fisheye / multi-camera**

- Add a `CameraModel` protocol (`project`, `unproject`) in `common.frames`; pinhole is the only v1.0 implementation.
- GT rays carry `cam_id`; cast/eval do not assume a specific camera.
- Fisheye `image_02`/`image_03` axes differ; a new model must not reuse the perspective formula.

**Multi-segment**

- `processed/<seq>/` and `artifacts/<run_id>/` are keyed; config selects the segment.
- A future aggregator can read multiple `metrics.json` files; no global path assumptions.

---

## Build Order (Risk-Minimizing)

The order follows a **correctness-first, evaluation-first** principle: validate frames and metrics on synthetic data before spending time on reconstruction, then cast the real mesh.

| Phase | Work | Deliverables | Gate (must pass before next) | Depends On |
|-------|------|--------------|-----------------------------|------------|
| **0. Preflight & scaffold** (1–2 d) | Dataset readiness check; package + C++ extension skeleton; configs; `docs/frames.md`; test harness; `.gitignore` | `check_dataset` report; `pip install -e .`; `pytest` smoke; `hls --help` | All required KITTI-360 files present **or** a concrete acquisition task; build works | — |
| **1. Frames & calibration** (2–3 d) | Load official calib/poses; compose `T_AB`; SEG; pose lookup; rectification; validation overlay | `calib.npz`, `frames.csv`, overlay images, unit tests | Round-trip; `T_W_C0_rect` vs `cam0_to_world` < 1e-3; LiDAR→image overlay aligns | 0 |
| **2. Data prep + fixed GT rays** (3–5 d) | Segment selection; dynamic/ego filtering; static GT; camera eligibility; primary camera; ray NPZ + manifest | `processed/<seq>/gt/rays/*.npz`, `rays_manifest.json`, overlays | Counts stable across reruns; no ego/dynamic; coverage/bucket counts reported; ray-set hash frozen | 1 |
| **3. Eval harness on synthetic predictions** (2–3 d) | Range/CD/F/P/R/miss/buckets/timing; JSON schema; synthetic tests | `metrics.json`/CSV schema; passing tests | Identity prediction = 0 error; empty scene = 100% miss; analytic plane/box matches; definitions frozen | 2 (rays) |
| **4. Mesh route spike + ingest** (3–7 d, parallel) | Choose COLMAP MVS / OpenMVS / learning; export COLMAP; reconstruct a short window; canonicalize | `scene.ply` + `mesh_manifest.json`; route decision note | Metric scale; bbox overlaps trajectory; visual alignment; triangle count/runtime acceptable | 1 (calib/images) |
| **5. CPU BVH + cast** (3–5 d) | Flat BVH; Möller–Trumbore; batching/threads; `PredictionBundle` | `scene.bvh` + `pred/*.npz` + `cast_manifest.json` | Matches brute force/Open3D on 1k rays; no misses for analytic inside-mesh rays; throughput measured | 4 (mesh), 2 (rays) |
| **6. End-to-end + metrics/report** (3–5 d) | Full selected segment; buckets; CD/F/P/R; timings/memory; plots; report | `metrics.json`, CSV, plots, `results/report.md` | All required metrics present; per-bucket/per-frame counts consistent; rerun from archived mesh identical | 3, 5 |
| **7. Reproducibility hardening** (1–2 d) | Clean-room rerun; manifests/hashes; docs; fallback notes | `docs/repro.md`; final artifacts | Cast/eval rerun bitwise identical; raw→processed reproducible; missing-data and route limitations documented | 6 |

### Critical Path and Parallel Tracks

```
0 → 1 → 2 → 3 ─┐
       │        ├→ 5 → 6 → 7
       └→ 4 ────┘   (4 is parallel and the long pole)
```

- **Critical path:** dataset → frames → GT rays → eval harness → cast → end-to-end → hardening.
- **Long pole:** mesh reconstruction (Phase 4). Start its spike as soon as calibration and a short image window exist; do not wait for the full GT ray set.
- **Parallel track:** Phase 2 (GT rays) and Phase 4 (mesh) are independent after Phase 1. Phase 3 only needs the ray schema, not the real mesh.
- **Why eval before reconstruction:** if metrics are wrong, every mesh result is meaningless. Synthetic tests are cheap and catch sign/axis/rectification errors early.
- If official poses/calibration are missing, dataset acquisition is a blocking external dependency inside Phase 0; code scaffolding can proceed in parallel, but no real-data gate can pass until the files are present and validated.

### Fallbacks and De-Scope Rules

- **Dataset missing official poses/calibration:** stop and acquire them. Do not substitute KITTI-raw calibration or OXTS-derived poses without labeling the run as a different protocol. A silent substitution invalidates the milestone.
- **COLMAP MVS fails or is too slow:** try OpenMVS on the same known-pose model; if still failing, use a learning-based metric-depth/TSDF route. If none works, reduce the window/decimate aggressively and document the limitation. Do **not** build a mesh from LiDAR points and report it as an RGB reconstruction.
- **Custom BVH slips:** promote Open3D `RaycastingScene` to a temporary CPU backend behind `SceneBackend`; keep the custom BVH as the reference/oracle and document the substitution. This still gives a CPU BVH result, but the milestone's Möller–Trumbore requirement should be met before final reporting.
- **COLMAP CUDA unavailable:** use OpenMVS CPU or a learning route; record the route.
- **Coverage too low with two front cameras:** add `image_01` (already front stereo) or fisheye cameras; report coverage explicitly rather than silently evaluating a narrow sector.
- **Time pressure:** reduce the number of frames/triangles and drop optional per-class metrics; never drop the fixed ray set, calibration validation, or the full metric suite.

---

## Scaling Considerations

This is an offline research pipeline, so "scale" means frames, rays, triangles, and cameras—not users.

| Scale | Rays | Triangles | Architecture Adjustment |
|-------|------|-----------|-------------------------|
| 1 frame, 1 camera | ~1e5 | ~1e5 | In-memory; no chunking; trivial |
| 50 frames, 2 cameras | ~1e7 | ~1e6 | Build BVH once; chunk rays; OpenMP; per-frame NPZ |
| Full segment (1e4 frames) | ~1e9 | 1e6–1e7 | GPU BVH + tiling (P2); frame streaming; LOD; do not hold all predictions in memory |
| Multi-camera (4) | 2–4× rays | same | Primary-camera assignment or dedupe by `lidar_idx`; report per-camera |
| Multi-segment | scales linearly | per-segment meshes | Keyed artifacts; aggregate metrics later |

### Scaling Priorities

1. **First bottleneck: mesh quality, not cast speed.** MVS holes/vegetation/road texture cause misses and high errors. Improve reconstruction route/params or choose a better segment before optimizing the BVH.
2. **Second bottleneck: CPU cast throughput.** 1e7 rays against 1e6 triangles is slow in Python; use C++/OpenMP, chunking, and BVH quality. This is exactly why P2 moves to GPU.
3. **Third bottleneck: metrics nearest-neighbor.** `cKDTree` per frame is fast, but storing/writing per-ray data for 1e9 rays is not. Store per-frame summaries by default; per-ray dumps only for debugging.
4. **Memory:** process frame-by-frame; keep the mesh/BVH resident; write predictions incrementally. Do not accumulate all `p_pred` arrays in RAM.

---

## Anti-Patterns

### Anti-Pattern 1: Casting from the LiDAR origin

**What people do:** Use the LiDAR origin and beam directions for the main protocol.
**Why it's wrong:** The camera and LiDAR are not co-located; this reintroduces parallax and mismatches the reconstruction camera. The project's primary protocol is camera-origin GT-defined rays.
**Do this instead:** For each GT point, transform to the camera frame, cast from the camera optical center along `p_C/||p_C||`, and map hits back to `V`.

### Anti-Pattern 2: Using the accumulated static PLY as per-frame GT

**What people do:** Take `data_3d_semantics/.../static/*.ply` and evaluate against it.
**Why it's wrong:** It is a fused world-frame cloud containing points from many frames, not the per-frame LiDAR measurement; it double-counts and includes points not visible in the evaluated frame.
**Do this instead:** Use raw per-frame `velodyne_points/data/{frame}.bin`, filter dynamic/ego, generate per-frame GT rays.

### Anti-Pattern 3: Mixing raw images with rectified intrinsics

**What people do:** Use `data_rgb` with `P_rect`/`R_rect`, or `data_rect` without `R_rect`.
**Why it's wrong:** A systematic pixel/projection offset that looks like a reconstruction error.
**Do this instead:** Use `data_rect` + `P_rect` + `R_rect` consistently for perspective cameras; apply `R_rect` on the camera side in every transform.

### Anti-Pattern 4: World-frame float32 mesh

**What people do:** Store the mesh in KITTI world coordinates with float32.
**Why it's wrong:** Kilometric coordinates lose centimeter precision; BVH bounds and intersections suffer.
**Do this instead:** Store mesh/BVH in SEG; keep transforms in float64; record `T_S_W`.

### Anti-Pattern 5: Treating misses as large range errors

**What people do:** Set `r_pred` to a large number or 0 for misses and include them in MAE/RMSE.
**Why it's wrong:** Misses are a coverage/recall failure, not a range measurement; including them corrupts the range metric.
**Do this instead:** `t_pred = inf`, exclude from range errors, count in miss rate and ray recall.

### Anti-Pattern 6: Rebuilding the BVH per frame

**What people do:** Load the mesh and build a BVH inside the per-frame loop.
**Why it's wrong:** Wastes minutes per run and dwarfs cast time.
**Do this instead:** Build once per mesh, cache `scene.bvh` with a mesh hash, reuse across frames.

### Anti-Pattern 7: Evaluating points outside the reconstructed cameras

**What people do:** Evaluate all 360° LiDAR points even though the mesh was reconstructed from two front cameras.
**Why it's wrong:** The representation never saw those surfaces; metrics are meaningless and dominated by misses.
**Do this instead:** Define eligible GT rays by projection into the reconstruction camera set; report coverage and per-camera counts.

### Anti-Pattern 8: Using GT to tune the pipeline or select the segment

**What people do:** Try thresholds/segments and keep the one with the best metrics.
**Why it's wrong:** Leakage and cherry-picking; the result is not a fair baseline.
**Do this instead:** Freeze segment selection and eligibility before reconstruction; use GT only for evaluation (and for the sealed ray set). P2 router training must use a different sequence from evaluation.

### Anti-Pattern 9: Running COLMAP SfM despite metric poses

**What people do:** Run `colmap mapper` and then align to metric poses.
**Why it's wrong:** Violates the locked decision; adds scale/pose drift and unnecessary work.
**Do this instead:** Build a known-pose COLMAP model (empty `points3D`, poses in `images.txt`), run `point_triangulator` + MVS. No `mapper`, no Umeyama.

### Anti-Pattern 10: Coupling reconstruction output to evaluation code

**What people do:** Let the MVS script write directly into the eval or metrics code path.
**Why it's wrong:** Replacing the mesh route (e.g., with 2DGS→TSDF later) requires touching eval; results become hard to compare.
**Do this instead:** `MeshArtifact` + manifest is the only contract; `ingest` validates; eval reads predictions only.

### Anti-Pattern 11: Non-deterministic discovery and ordering

**What people do:** `glob()` images/velodyne files, rely on filesystem order, or use unseeded sampling.
**Why it's wrong:** Different frame sets/ray sets between runs, unreproducible metrics.
**Do this instead:** Sort frame IDs numerically, seed any sampling, write a frame list/manifest, hash the ray set.

### Anti-Pattern 12: Assuming the MVS mesh is watertight

**What people do:** Use front-face-only intersection or assume every ray hits.
**Why it's wrong:** MVS meshes are open, have holes, and can have inconsistent normals; front-face culling creates false misses.
**Do this instead:** Use two-sided Möller–Trumbore for v1.0; report miss rate; do not require watertightness. A later TSDF/MC route can be watertight, but the interface stays the same.

### Anti-Pattern 13: Ignoring camera-origin occlusion in GT rays

**What people do:** Assume every GT point visible to the LiDAR is also visible from the displaced camera origin.
**Why it's wrong:** Parallax can put an occluder between the camera and the GT point, producing a short hit that is not a reconstruction error.
**Do this instead:** Keep the README protocol, but add a diagnostic `occlusion_suspect` flag (first hit significantly before `t_gt` with a front-facing normal) and report metrics with/without suspects as an ablation. Do not filter the main metric silently.

---

## Open Questions / Research Flags

1. **Dataset readiness (blocking).** On 2026-10-30, the local `/data-20T/kitti` tree contains KITTI-360-style sequences (`2013_05_28_drive_0000_sync`, …), raw images (1408×376) under `raw/2013_05_28/<seq>/image_00/data` (not the official `data_rect`/`data_rgb` subfolders), velodyne `.bin` files under `data_3d_raw/`, and 3D bbox XML, but no official `data_poses/` (`poses.txt`/`cam0_to_world.txt`), no KITTI-360 `calibration/` (`calib_cam_to_pose.txt`, `calib_cam_to_velo.txt`, `perspective.txt`), and no `data_2d_semantics/`. It has KITTI-raw-style `calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt` at `raw/2013_05_28/` instead. **Do not assume this is a complete official KITTI-360 release.** Run a preflight, acquire the missing official files, and validate any repackaged calibration against image size and projection overlays. If only the raw `image_XX/data` images are available, add an explicit rectification/undistortion step (or acquire `data_rect`). The `data.layout` adapter should support both the official tree and this split/repackaged variant, but the protocol still requires official metric poses. The existing empty `processed/` folders under `/data-20T/kitti` are scaffolding, not data; do not treat them as a ready-made preprocessing output. OXTS-derived poses are a last-resort fallback, not a substitute for official poses.
2. **Mesh route.** COLMAP MVS with known poses requires a sparse model; the FAQ-approved workaround is an empty `points3D.txt` + `point_triangulator` after feature extraction/matching. Validate that this works on KITTI-360 rectified stereo and that `patch_match_stereo` runs in the available CUDA build. OpenMVS may produce a better mesh; compare on a short window. A learning-based route must preserve metric scale (monocular depth is scale-ambiguous).
3. **COLMAP image IDs.** `images.txt` image IDs must match the database created by `feature_extractor`. Use `pycolmap` or run feature extraction first and read IDs; do not hand-assign IDs blindly.
4. **MVS dynamic objects.** COLMAP MVS may reconstruct dynamic objects that remain in the images. Low-dynamic selection mitigates; per-pixel masking support in the chosen MVS route must be verified. If 3D bboxes/GT are used to mask images, that is LiDAR-informed reconstruction and should be avoided or disclosed.
5. **Pose/timestamp gaps.** `poses.txt` is non-contiguous; LiDAR and images have separate timestamps and motion distortion. Decide and document nearest-pose lookup and whether raw vs curled velodyne points are used.
6. **CPU BVH implementation cost.** A custom C++/pybind11 flat BVH is recommended for correctness and the P2 seam; Open3D `RaycastingScene` is the independent oracle. Validate build effort early; fall back to Open3D only with explicit labeling.
7. **Camera coverage.** The two front perspective cameras (`image_00`, `image_01`) cover a limited sector. Decide whether they suffice or whether fisheye cameras are needed; report coverage rather than implying 360° evaluation.
8. **Evaluation split.** Decide whether reconstruction uses all selected frames and evaluation is on a held-out subset, or both use the same frames. Holding out frames is a stronger novel-view protocol; on-trajectory evaluation is simpler. Record the split in `frames.csv`.
9. **Metric definitions.** Freeze the definitions in `docs/protocol.md` before the real mesh run. This document proposes coordinate L1 + squared Euclidean for CD, bidirectional nearest-neighbor F-score at τ, and both micro (pooled) and macro (per-frame) aggregation. Do not change definitions between representations.
10. **Ground/textureless surfaces.** MVS may produce poor road/ground geometry; this is the expected mesh-only weakness. Do not "fix" it with LiDAR. Report per-class metrics if semantic IDs are available.
11. **Reconstruction reproducibility.** Bitwise reproducibility of multi-threaded MVS is unlikely. Archive the mesh and hash it; define reproducibility as bitwise cast/eval from the archived mesh plus approximate recon reproduction.
12. **Licenses.** Check the chosen mesh route's license before distributing code/binaries (OpenMVS is AGPLv3; COLMAP and learning-based routes vary).

---

## Sources

- KITTI-360 official documentation — folder structure, 2D/3D formats, calibrations, poses, sensor axes: https://www.cvlibs.net/datasets/kitti-360/documentation.php (HIGH)
- `kitti360Scripts` `project.py` — camera-to-world composition, rectification, pose loading: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py (HIGH)
- `kitti360Scripts` `loadCalibration.py` — calibration file parsing: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/commons/loadCalibration.py (HIGH)
- `kitti360Scripts` `kitti360Viewer3DRaw.py` — `TrVeloToCam`, `R_rect` application, velodyne uncurling: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/viewer/kitti360Viewer3DRaw.py (HIGH)
- COLMAP documentation, model format (`images.txt` world-to-camera, quaternion convention, PINHOLE): https://colmap.github.io/format.html (HIGH)
- COLMAP FAQ, reconstruct from known camera poses (empty `points3D`, `point_triangulator`): https://colmap.github.io/faq.html (HIGH)
- COLMAP CLI, dense reconstruction commands (`image_undistorter`, `patch_match_stereo`, `stereo_fusion`, `poisson_mesher`): https://colmap.github.io/cli.html (HIGH)
- Open3D `RaycastingScene.cast_rays` API — `t_hit`, `primitive_ids`, `primitive_uvs`, `primitive_normals`, `inf` on miss: https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html (HIGH)
- Möller, T., Trumbore, B. "Fast, Minimum Storage Ray/Triangle Intersection," *Journal of Graphics Tools*, 1997. (HIGH)
- Project documents: `README.md` §0.5, §1.3, §6, §8; `.planning/PROJECT.md`; `docs/proposal_V2.tex`. (HIGH for project intent)
- Local dataset preflight, `/data-20T/kitti`, observed 2026-10-30 (MEDIUM; machine-specific and may change).

---

*Architecture research for: mesh-only KITTI-360 LiDAR simulation + GT-ray evaluation*
*Researched: 2026-10-30*
