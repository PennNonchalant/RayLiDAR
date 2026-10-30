# Stack Research

**Domain:** Mesh-only KITTI-360 reconstruction + GT-defined-ray LiDAR evaluation (CPU BVH reference)
**Researched:** 2026-10-30
**Confidence:** HIGH for versions, tool availability, and environment fit; MEDIUM for mesh-quality ranking on KITTI-360 driving scenes (extrapolated from published MVS/mesh benchmarks, not a KITTI-360 head-to-head).

---

## Decision (read this first)

**Use COLMAP for pose ingestion + sparse triangulation, and OpenMVS 2.4.0 for densification + meshing.** Concretely: write KITTI-360's provided metric poses into a known-pose COLMAP model (PINHOLE, rectified images), triangulate sparse points while freezing poses/intrinsics, import that model with `InterfaceCOLMAP`, then run OpenMVS `DensifyPointCloud → ReconstructMesh → RefineMesh`. Defer all learning-based reconstruction (VGGT / MASt3R) out of this milestone.

```
KITTI-360 rectified perspective images (image_00/01) + cam0_to_world.txt + P_rect_00/01
        │  our converter writes a known-pose COLMAP model (cameras.txt / images.txt / empty points3D.txt)
        ▼
colmap feature_extractor → exhaustive_matcher → point_triangulator   # poses fixed, intrinsics fixed
        ▼
InterfaceCOLMAP -i <workspace> -o scene.mvs --image-folder <rect images>
        ▼
DensifyPointCloud scene.mvs -m masks --ignore-mask-label 1            # CPU MVS fusion
        ▼
ReconstructMesh scene_dense.mvs -p scene_dense.ply -o scene_dense_mesh.ply   # visibility-aware Delaunay surface
        ▼
RefineMesh scene_dense.mvs -m scene_dense_mesh.ply --scales 1 --max-face-area 16
        ▼
scene_dense_mesh_refined.ply  →  Open3D cleanup/decimation  →  Numba BVH (Möller–Trumbore)
```

Why this route wins for a course-timeline, single-segment baseline:

1. **KITTI-360 already gives metric poses, so SfM is pure waste.** COLMAP is used only as a *pose container + triangulator*: features and matches are needed to seed the sparse cloud for MVS, but the mapper never re-estimates poses. Poses and intrinsics are frozen.
2. **The prebuilt OpenMVS bundle does not ship `CreateStructure`.** Verified locally on the v2.4.0 Ubuntu bundle (2026-10-30): the archive contains `InterfaceCOLMAP`, `InterfaceMetashape`, `InterfaceMVSNet`, `InterfacePolycam`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, `TextureMesh`, `TransformScene`, `Viewer`, `Tests` — but **no `CreateStructure` / `ExtractKeyframes`**. The wiki's `--import-poses-file` route is therefore unavailable in the shipped binary; COLMAP is the supported pose-import front end (`InterfaceCOLMAP` explicitly reads COLMAP TXT/BIN models).
3. **OpenMVS is the mesh-oriented tool.** In the UseGeo dense-reconstruction evaluation, COLMAP produced the best *depth*, but **OpenMVS produced triangle meshes with the lowest error** (Hermann et al., ISPRS Open J. Photogramm. Remote Sens., 2024, doi:10.1016/j.ophoto.2024.100065). We need a mesh, not just depth.
4. **It works on CPU.** The host's COLMAP 4.0.4 is a CPU-only build (can do features/matching/triangulation, not `patch_match_stereo`), and the OpenMVS binaries run without CUDA. The host has **no `nvcc`** (verified), so this matters.
5. **The OpenMVS bundle is self-contained and verified to execute on this host.** The 165 MB release asset runs directly (`-h` succeeded, no `LD_LIBRARY_PATH` needed), so there is no CGAL/VCGLib/vcpkg build on the critical path — the historic reason people avoided OpenMVS.
6. **Its Delaunay + visibility mesher produces visibility-consistent surfaces** and does not automatically close unobserved regions the way Poisson does. That is the correct prior for a hit/miss LiDAR protocol: unobserved space tends to stay empty (honest misses) instead of being sealed with hallucinated geometry (false hits that inflate recall).

**Quality-upgrade path (only if the acceptance bar is missed):** install COLMAP 4.2.1 with the `cuda_129` variant, run `patch_match_stereo → stereo_fusion` for better depth, then either mesh with COLMAP's `delaunay_mesher` or feed the depth maps to OpenMVS (DMAP import) and keep the OpenMVS mesh stage. This is the best-of-both route: COLMAP depth, OpenMVS mesh.

---

## Recommended Stack

### Core Technologies (Python, data + evaluation)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11.x (pin; host has 3.12.3) | Runtime for preprocessing, eval, BVH | PROJECT.md targets 3.11; every selected wheel supports 3.11 and 3.12. 3.11 is the safe pin because NumPy ≥2.5 / SciPy ≥1.18 already require 3.12+. |
| NumPy | 2.4.4 (3.11 line) / 2.5.4 (3.12) | All geometry, ray arrays, point clouds | NumPy 2.4.x is the last line installable on 3.11; NumPy 2.5.4 requires ≥3.12. Both are fine; let the interpreter choose. |
| SciPy | 1.17.1 (3.11 line) / 1.18.1 (3.12) | `cKDTree` for Chamfer / F-score / NN matching | `cKDTree.query(..., workers=-1)` handles 1e5×1e5 point sets in seconds; no need for PyTorch3D or GPU metrics in P1. |
| Numba | 0.68.1 | JIT-compiled CPU BVH + Möller–Trumbore | The locked "CPU BVH (Möller–Trumbore) first" decision. Numba compiles the traversal loop without a C++/pybind11 build. 0.68.1 requires `numpy<2.6`, so NumPy 2.4/2.5 are compatible. |
| Open3D (CPU) | open3d-cpu 0.20.0 | Mesh I/O, cleanup, decimation, TSDF later, Embree oracle | `open3d-cpu` avoids dragging a CUDA runtime into a CPU-first project. `RaycastingScene` (Embree) is the independent cross-check for the hand-written BVH. Also the future 2DGS→TSDF→MC bridge. |
| trimesh | 5.2.0 | Watertight/repair checks, PLY scene handling, triangle utilities | Complements Open3D: better repair/containment checks and cheap PLY/triangle introspection. Do not duplicate Open3D's TSDF/raycasting. |
| pandas | 3.0.2 (host) / 3.1.x | Metrics tables → CSV/JSON | One row per (frame, camera, bucket, metric); trivial pivot for the report. |
| Matplotlib | 3.10.9 (host) / 3.11.x | Range-vs-bucket plots, CD–FPS Pareto, coverage plots | No dashboard/experiment tracker needed for one segment. |
| Pillow | 12.2.0 | PNG image + mask I/O | Sufficient for KITTI-360 PNGs and generating black/white masks; avoids an OpenCV dependency. |
| PyYAML | 6.0.1 | `configs/kitti360.yaml` | Config-driven frame window, paths, thresholds, bucket edges. |

### Reconstruction Toolchain (external binaries)

| Technology | Version | Purpose | When to Use |
|------------|---------|---------|-------------|
| **OpenMVS** | **2.4.0** | Dense MVS, mesh reconstruction, mesh refinement | **Primary mesh stage.** Prebuilt `OpenMVS_Ubuntu_x64.zip` (flat bundle, verified executable on this host); `InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → RefineMesh`. `CreateStructure` is **not** in the bundle. |
| **COLMAP** | **4.0.4 (installed)**; 4.2.1 optional | Pose container + sparse triangulation (`feature_extractor`, `exhaustive_matcher`, `point_triangulator`, `model_converter`); optional dense `patch_match_stereo` | **Primary pose-ingestion stage.** The installed `/data-20T/conda_envs/colmap` build is CPU-only: fine for features/matching/triangulation, **cannot** run `patch_match_stereo`. Install 4.2.1 (`cuda_129` variant) only for the dense quality upgrade. |
| pycolmap | 4.2.1 (optional) | Read/write COLMAP models; easier than parsing `database.db` | Optional convenience for writing `cameras.txt`/`images.txt` with database-consistent image IDs. |
| KITTI-360 devkit (`kitti360Scripts`) | git master (MIT) | Calibration, pose, label and Velodyne helpers | Use as a **reference implementation**, not as a runtime dependency: parts are Python 2 (`print '...'`, `.astype(np.int)`), so port/vendor the loaders you need. |

### Mesh Processing & Ray Casting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `open3d.t.geometry.RaycastingScene` | 0.20.0 (Embree, CPU) | Independent ray-cast oracle; `cast_rays` returns `t_hit`, `primitive_ids`, `primitive_uvs` | Validate the hand-written BVH on a 1e4-ray sample before trusting full-suite numbers. Also useful for quick mesh/ray sanity plots. |
| NumPy + Numba BVH (in-repo) | numba 0.68.1 | Primary GT-ray caster: median-split BVH + branchless-capable Möller–Trumbore, `prange` over rays | Every metric number. Keep `fastmath=False` for the reference; optionally add a `fastmath=True` variant for a perf comparison. |
| Open3D legacy geometry | 0.20.0 | `read_triangle_mesh`, `remove_degenerate_triangles`, `remove_duplicated_vertices`, `cluster_connected_triangles`, `simplify_quadric_decimation`, `filter_smooth_laplacian`, `compute_vertex_normals` | Post-OpenMVS cleanup and LOD experiments. |
| Open3D TSDF (deferred) | 0.20.0 | `ScalableTSDFVolume` / `VoxelBlockGrid` + `extract_triangle_mesh` | Only for the deferred 2DGS→TSDF→Marching Cubes route. Do not use for the v1.0 OpenMVS mesh. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| psutil | 7.2.2 | Peak RSS for the memory column of the metric suite | Every evaluation run; GPU memory only becomes relevant in P2. |
| tqdm | 4.70.1 | Progress for frame loops and BVH casting | Long preprocessing/casting loops. |
| pytest | 9.2.0 | Unit tests for calibration transforms, uncurling, BVH-vs-Embree, metric math | Phase 1 gate: transforms and metrics must be tested before any reconstruction result is interpreted. |
| ruff | 0.17.3 | Lint/format | Repo hygiene. |
| opencv-python-headless | 5.0.x (optional) | Fisheye undistortion, PNG mask rasterization if Pillow is insufficient | Only if fisheye `image_02/03` coverage is added later. Not required for the 00/01 path. |
| pymeshlab | 2025.7 (optional) | Heavy mesh cleanup (hole filling, Laplacian, Hausdorff) | Only if Open3D/trimesh cleanup proves insufficient. Do not install pre-emptively. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `uv` 0.11.8 (already installed) | Create/pin the Python 3.11 env and resolve wheels | `uv venv --python 3.11`; no conda needed for the Python side. |
| Git + DVC or plain manifests | Track frame windows, configs, and metric JSONs | Keep large artifacts out of git; store `metrics.json`, `metrics.csv`, `figs/`. |
| Nsight / nvcc | Deferred | **Not installed** on this host; do not put any CUDA compile step in v1.0. |

---

## Reconstruction Route Decision (required comparison)

The three options are not perfectly symmetric: COLMAP and OpenMVS are complementary (SfM/dense vs. surface), while VGGT/MASt3R replace the whole geometric pipeline with a learned feed-forward model. Compare them as **end-to-end mesh routes**.

| Criterion | COLMAP MVS (4.2.1) | OpenMVS (2.4.0) | Learning-based (VGGT / MASt3R) |
|-----------|--------------------|-----------------|-------------------------------|
| Maturity | Very high; de-facto reference for SfM/MVS | High (since 2014; active 2.4.0 in Jan 2026); standard COLMAP companion | High research velocity, low operational maturity; APIs/checkpoints move |
| Mesh output | Built-in `poisson_mesher`, `delaunay_mesher`, new advancing-front mesher, `mesh_texturer`, `mesh_simplifier` | Purpose-built `ReconstructMesh` (Delaunay + visibility) + `RefineMesh` | **No native mesh**; must chain depth → TSDF/poisson → MC yourself |
| Setup cost here | **Low for pose ingest:** installed 4.0.4 CPU does features/matching/`point_triangulator`; **medium for dense:** a CUDA build is needed for `patch_match_stereo` | **Low:** prebuilt Ubuntu x64 bundle (verified runnable); needs a COLMAP model as input, then 4 commands; CPU only | Low install, **high integration**: depth/point-map export, scale alignment, chunking, TSDF, MC, plus GPU memory tuning |
| Quality on driving scenes | Best depth accuracy in the UseGeo evaluation; Poisson tends to close holes and hallucinate surfaces; Delaunay/advancing-front better but need clean points | **Lowest mesh error** in the UseGeo dense-MVS evaluation (UAV scenes, not driving); visibility-aware open surfaces; detail recovered by `RefineMesh` | Depth is smooth and plausible, but loses thin structures; long-range driving geometry is weak (limited parallax); no metric scale |
| GPU need | Features/matching/triangulation run on CPU; **`patch_match_stereo` requires CUDA** | Optional (CUDA builds exist); the Ubuntu bundle is verified to run without CUDA | Required; 1B-class transformer + attention over frames; memory grows with frame count |
| Metric scale | Metric (poses given) | Metric (poses given, stays in input frame) | **Up-to-scale**; needs Umeyama alignment (cameras-only is possible, but it is another failure mode) |
| Coordinate/pose handling | **We write the known-pose model; freeze intrinsics and BA**; the mapper is never run | Reads the COLMAP model; keeps the given coordinate frame; only PINHOLE supported | Predicts its own poses; must be aligned to the provided metric frame |
| License | New BSD (permissive) | **AGPL-3.0** (fine for coursework; matters if binaries are redistributed) | VGGT: original checkpoint non-commercial, new commercial checkpoint gated; MASt3R: **CC BY-NC-SA 4.0** |
| Verdict for v1.0 | **Primary (pose ingest; optional dense upgrade)** | **Primary (densify + mesh)** | **Out of scope** |

**The recommended route is COLMAP + OpenMVS, not either one alone:** COLMAP supplies the pose-consistent sparse cloud (the shipped OpenMVS bundle has no `CreateStructure`), and OpenMVS supplies the surface stage where it is strongest.

### Why OpenMVS for the mesh stage, with COLMAP only for pose ingestion

- **The shipped bundle has no SfM front end.** `CreateStructure`/`ExtractKeyframes` are absent from the prebuilt 2.4.0 Ubuntu archive (verified locally), so a pose-import tool is required anyway. The installed CPU COLMAP fills that role using a documented, frozen-pose workflow.
- **No CUDA toolchain needed for the primary path.** COLMAP's CPU feature/matching/triangulation stages plus OpenMVS's CPU densify/mesh run on this host. A CUDA COLMAP build is only needed for the optional `patch_match_stereo` quality upgrade.
- **Mesh-first, not depth-first.** The deliverable is a mesh; OpenMVS's `ReconstructMesh` + `RefineMesh` are the surface stages, and visibility-aware Delaunay avoids Poisson's false closure (important for honest miss rates). OpenMVS also exposes cleanup (`--remove-spurious`, `--remove-spikes`, `--close-holes`, `--smooth`, `--target-face-num`) so the mesh is presentable without a second tool.
- **Prebuilt binary changes the cost calculus.** The historic objection ("OpenMVS is a 1-hour build with CGAL/VCGLib") no longer holds — the bundle ran on this host with no dependency setup.

### Why not learning-based (VGGT / MASt3R) in v1.0

- **No mesh.** You still need depth → TSDF → Marching Cubes, i.e. the deferred route plus a learned-depth dependency.
- **Scale ambiguity.** VGGT/MASt3R outputs are up-to-scale; making them metric requires aligning predicted cameras to KITTI-360 poses (cameras-only Umeyama is legitimate but adds a failure mode). Any LiDAR-based scale alignment would leak GT into reconstruction and violate the protocol.
- **Driving-scene geometry.** Forward motion gives weak parallax at 40–60 m; learned depth is smooth and loses thin structures, which is the opposite of what the baseline needs to isolate.
- **Licensing and ops.** Non-commercial checkpoints and frame-count-dependent GPU memory are avoidable complexity for a course timeline.
- **It would confound the baseline.** v1.0 is supposed to validate the *evaluation harness* and a classical mesh; adding a learned geometric prior makes any error impossible to attribute.

### When each alternative is correct

- **Accuracy upgrade (only if needed):** install `colmap=4.2.1` with the `cuda_129` variant, run `image_undistorter → patch_match_stereo → stereo_fusion`, and either mesh with COLMAP's `delaunay_mesher` or feed the depth maps to OpenMVS's DMAP densification. This is the best-of-both route: COLMAP depth, OpenMVS mesh.
- **Single-toolchain alternative:** if the COLMAP↔OpenMVS import proves brittle, mesh directly with COLMAP 4.2.1's `delaunay_mesher` / `poisson_mesher` / advancing-front mesher. Lower setup variety, but Poisson tends to close holes (inflating recall) and Delaunay needs cleaner points.
- **Use VGGT/MASt3R** only as a *diagnostic* escape hatch on a low-texture block, or later as an additional baseline — never as the metric reference.

---

## End-to-End Pipeline Contract

### 1. Data preparation (Python)

- Use **rectified perspective images** `image_00/data_rect`, `image_01/data_rect` and the matching `P_rect_00` / `P_rect_01` / `S_rect_00` from `calibration/perspective.txt`. KITTI-360 provides 2D semantic/instance labels on these rectified images, so masks and dynamic filtering line up pixel-exactly. Do not use `data_rgb` for the first pass (it is unrectified).
- **Sensor-count check:** the official KITTI-360 documentation describes **2 perspective cameras (`image_00/01`) + 2 fisheye cameras (`image_02/03`)**, while README §6 says "4 perspective + 2 fisheye". Plan v1.0 around the 2 rectified perspective cameras and confirm the actual folder inventory after download; do not budget reconstruction compute for 4 perspective cameras.
- Poses come from `data_poses/.../cam0_to_world.txt` (4×4, rectified cam0 → world, per frame). Do **not** run COLMAP SfM and do **not** run Umeyama alignment.
- Derive the second camera's pose from `P_rect_00`/`P_rect_01` (stereo baseline) or from `calib_cam_to_pose` + `R_rect`; **unit-test by projecting static LiDAR points into both rectified images** before trusting any ray.
- Build the Velodyne→camera chain once. With the convention `T_A_B` = "transforms B into A" (`p_A = T_A_B · p_B`):
  `T_velo_cam0rect = T_velo_cam0raw · R_rect_00⁻¹` (because `p_cam0rect = R_rect_00 · p_cam0raw`), then `T_world_velo = T_world_cam0rect · T_cam0rect_velo` with `T_cam0rect_velo = inv(T_velo_cam0rect)`. Verify by overlay, not by inspection.
- **Uncurl the Velodyne scans.** KITTI-360 raw scans are ego-motion distorted; the devkit's `curlVelodyneData` (a Cython/C extension) applies an azimuth-interpolated rigid correction using `poses.txt` + `calib_cam_to_pose` + `calib_cam_to_velo`. Port that ~30-line routine to NumPy (vectorized Rodrigues) and unit-test against the devkit on a handful of frames. At driving speed the distortion is at the metre scale — it is not optional for 0–50 m metrics.
- Velodyne `.bin` is `N×4 float32` (`x, y, z, intensity`).
- Choose a **60–120 frame window** (≈ 50–100 m of driving) with low dynamic-pixel fraction and low speed. A whole KITTI-360 sequence is thousands of frames and is out of scope.
- Masks: dynamic classes from `labels.py` (`car 26`, `person 24`, `rider 25`, `truck 27`, `bus 28`, `train 31`, `motorcycle 32`, `bicycle 33`, plus `dynamic 5`). Write binary masks named `<image>.mask.png` (e.g., `00_0000000000.png.mask.png`) with `1` = ignore. Pass `-m work/masks --ignore-mask-label 1` to `DensifyPointCloud`, and `--mask-path work/masks --ignore-mask-label 1` to `RefineMesh` (the short `-m` means `--mesh-file` there). For a low-dynamic window this is a small correction, but it prevents ghost geometry.

Artifacts: `work/images/*.png`, `work/masks/*.png`, `work/colmap_ws/sparse/{cameras,images,points3D}.txt`, `work/frames.txt`.

### 2. Pose ingestion + reconstruction (COLMAP → OpenMVS)

**Step 2a — known-pose COLMAP model (CPU, installed 4.0.4):**

The installed binary is `/data-20T/conda_envs/colmap/bin/colmap` (not on `PATH`); alias it or use the full path below.

```bash
# 2a.1 Extract features first: the database assigns the image IDs that images.txt must reuse.
colmap feature_extractor \
  --database_path work/database.db --image_path work/images \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params "<fx,fy,cx,cy from P_rect_00>" \
  --FeatureExtraction.use_gpu 0

# 2a.2 Write work/colmap_ws/sparse/{cameras.txt,images.txt,points3D.txt}
#        - PINHOLE cameras with P_rect_00/01 values
#        - image_id values copied from database.db (names must match)
#        - world-to-camera quaternion/translation from cam0_to_world.txt (+ stereo baseline for cam1)
#        - points3D.txt EMPTY

colmap exhaustive_matcher --database_path work/database.db --FeatureMatching.use_gpu 0

colmap point_triangulator \
  --database_path work/database.db --image_path work/images \
  --input_path work/colmap_ws/sparse --output_path work/colmap_tri \
  --refine_intrinsics 0 \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0 \
  --Mapper.ba_refine_sensor_from_rig 0

# 2a.3 InterfaceCOLMAP reads <input>/sparse/*; export TXT explicitly (verified readable).
colmap model_converter --input_path work/colmap_tri \
  --output_path work/colmap_ws/sparse --output_type TXT
```

**Step 2b — OpenMVS densify + mesh:**

```bash
# The prebuilt bundle is FLAT: executables sit directly in the archive root (no bin/ or lib/).
# github.com is unreachable on this host; use the API asset endpoint (verified).
mkdir -p third_party && cd third_party
AID=$(curl -sL https://api.github.com/repos/cdcseacave/openMVS/releases/latest \
  | python3 -c "import sys,json;print([a['id'] for a in json.load(sys.stdin)['assets'] if a['name']=='OpenMVS_Ubuntu_x64.zip'][0])")
curl -sL -H "Accept: application/octet-stream" -o OpenMVS_Ubuntu_x64.zip \
  "https://api.github.com/repos/cdcseacave/openMVS/releases/assets/$AID"
unzip -q OpenMVS_Ubuntu_x64.zip -d openmvs-2.4.0
cd ..
export PATH="$PWD/third_party/openmvs-2.4.0:$PATH"

InterfaceCOLMAP -i work/colmap_ws -o work/scene.mvs --image-folder work/images
DensifyPointCloud work/scene.mvs -m work/masks --ignore-mask-label 1
ReconstructMesh   work/scene_dense.mvs -p work/scene_dense.ply -o work/scene_dense_mesh.ply
RefineMesh        work/scene_dense.mvs -m work/scene_dense_mesh.ply \
  -o work/scene_dense_mesh_refined.ply --scales 1 --max-face-area 16 \
  --mask-path work/masks --ignore-mask-label 1
```

Note the short-option asymmetry: in `DensifyPointCloud`, `-m` is `--mask-path`; in `RefineMesh`, `-m` is `--mesh-file` and the mask flag is long-only (`--mask-path`). Mixing them silently overwrites the wrong argument.

Pipeline gotchas (verified against the 2.4.0 binaries):

- `InterfaceCOLMAP` expects a folder containing `sparse/{cameras,images,points3D}.txt` (or `.bin`) — **not** just a bare model directory. A 2-image PINHOLE TXT model imported successfully on this host (2026-10-30), so the format is compatible.
- **Only the PINHOLE camera model is supported** by `InterfaceCOLMAP` (stated in its `-h`). `data_rect` + `P_rect` is exactly PINHOLE; do not use `SIMPLE_RADIAL`/`OPENCV`.
- `InterfaceCOLMAP`'s help says "undistort the images". Because KITTI-360 `data_rect` is already rectified and the model is already in that frame, `--image-folder work/images` is normally sufficient. If it complains, run `colmap image_undistorter` (CPU) to produce the exact expected workspace layout — with zero distortion it is effectively a copy.
- `image_id` consistency between `database.db` and `images.txt` is mandatory (COLMAP FAQ). Run `feature_extractor` first and read the IDs back (`sqlite3 database.db` or `pycolmap`).
- If you use `--ImageReader.single_camera 1` (valid: rectified stereo shares `fx,fy,cx,cy`), write exactly **one** PINHOLE camera (`camera_id 1`) to `cameras.txt` and reference it from every image row; otherwise `point_triangulator` sees a camera-ID mismatch.
- Freeze BA: `point_triangulator` exposes `--refine_intrinsics 0` (default) and inherits the `--Mapper.ba_refine_*` flags. Turn off focal/extra-parameter/rig refinement and verify the poses in the output `images.txt` are unchanged before evaluating.
- `DensifyPointCloud` defaults crop to an estimated ROI (`--estimate-roi 1.1 --crop-to-roi 1`). For a street corridor, inspect the result and set `--estimate-roi 0 --crop-to-roi 0` if it crops away valid geometry.
- `ReconstructMesh` already exposes cleanup (`--remove-spurious`, `--remove-spikes`, `--close-holes`, `--smooth`, `--decimate`, `--target-face-num`, `--crop-to-roi`); use it before Open3D cleanup. `RefineMesh` also accepts `--mask-path` and `--ignore-mask-label` (long forms; `-m` is the mesh there).
- Fast smoke test: `ReconstructMesh work/scene.mvs` from the sparse cloud → `RefineMesh`; only then run the dense path for the reported result.

### 2c. What about `CreateStructure`?

OpenMVS's wiki documents a `CreateStructure --import-poses-file` workflow, but **the prebuilt v2.4.0 Ubuntu bundle does not include `CreateStructure`** (verified by listing the archive). Do not make it the critical path. If a source build is ever needed, that CLI is the alternative pose-import route; otherwise use COLMAP (`InterfaceCOLMAP`) as above.

### 3. Mesh cleanup (OpenMVS built-ins + Open3D + trimesh)

- First use OpenMVS's own clean options: `ReconstructMesh --remove-spurious 20 --remove-spikes 1 --close-holes 30 --smooth 2 --target-face-num <N>` (and `--crop-to-roi` as appropriate). They are already tuned defaults, so only change them with a recorded rationale.
- Then a second pass in Open3D: `read_triangle_mesh` → `remove_duplicated_vertices` → `remove_degenerate_triangles` → `cluster_connected_triangles` (drop tiny components) → optional `simplify_quadric_decimation` to ≤2M triangles.
- Keep the **full-resolution** mesh as the reported baseline; if decimating for BVH build time, report both full and decimated metrics.
- `compute_vertex_normals` (needed later for grazing-angle features and normals-based diagnostics).
- Sanity checks: triangle count, bbox, `trimesh` volume/watertight flag (expect **not** watertight — the mesh is a street corridor; that is correct, not a bug).

### 4. GT-defined rays (Python)

For each frame, for each camera (00/01), for each static GT LiDAR point `p_L` in the chosen range:

```
p_C   = T_cam_velo · p_L            # camera frame
d     = p_C / ||p_C||
r_gt  = ||p_C||
o     = (0, 0, 0)
```

Keep a point only if it projects inside the image (optionally with a small margin) for that camera; merge cameras and dedupe overlapping rays. Save per-frame `.npz`:

`origin (N,3) float32`, `dir (N,3) float32`, `r_gt (N,) float32`, `p_gt_velo (N,3) float32`, `cam_id (N,) int8`, `bucket (N,) int8`.

This makes the ray set fixed and reproducible (a hard requirement in PROJECT.md).

### 5. CPU BVH cast (Numba)

- Build a median-split BVH over triangles (leaf ≤ 8; flat `node_min/node_max/left/right/start/count` + `tri_idx` arrays). Use SoA float32 for bounds/vertices; accumulate `t` in float64.
- Traverse with an explicit per-ray stack; slab test; **two-sided** Möller–Trumbore (mesh normals may be inconsistent); `t > eps`; return nearest hit plus `tri_id` and barycentrics.
- `@njit(parallel=True)` + `prange` over rays; `cache=True`; `fastmath=False` for the reference.
- Miss → `r_pred = +inf`, `tri_id = -1`.
- **Validation gate:** compare 1e4 random rays against `open3d.t.geometry.RaycastingScene.cast_rays`; require identical hit/miss and `max|Δt| < 1e-4` m before running the suite.

### 6. Metrics and artifacts

- Range (hits only): MAE, RMSE, MedAE, P90; misses excluded from range error but counted in recall/miss rate.
- Chamfer L1/L2 between `p_pred` (hits only) and the GT set. Define the GT set as **all static GT points in the camera FOV** so missed rays are penalised through CD as well as recall; additionally report CD restricted to GT points of hit rays for a like-for-like accuracy view. F-score@{0.05, 0.2, 0.5} m; precision/recall; miss rate.
- Distance buckets 0–20 / 20–40 / 40–60 m (primary 0–50 m; report 50–60 m separately).
- Efficiency: rays/s, frames/s, BVH build time, peak RSS, triangle count. Fixed hardware, fixed ray set, mean over runs.
- Emit `metrics.json` (nested), `metrics.csv` (tidy), and figures (`CD–FPS` later, range-RMSE-by-bucket now, coverage/evaluable-GT fraction).
- Also record **evaluable GT fraction** (points inside the 00/01 FOV ÷ all static GT points). With only 2 forward perspective cameras this will be well below 100%; that is a protocol property, not a failure.

---

## Integration With the Existing Environment (verified on this machine)

| Fact | Verified value | Implication |
|------|----------------|-------------|
| OS / glibc | Ubuntu 24.04.2 LTS, glibc 2.39 | The OpenMVS bundle was executed on this exact host, and `open3d-cpu` manylinux_2_35 (glibc ≥2.35) is compatible. |
| GPU / driver | RTX 4090, 48 GB, driver 580.82.09 | Plenty for later GPU work; irrelevant to the CPU-first v1.0 path. |
| CUDA toolkit | **No `nvcc` in PATH** | Do not require any CUDA compile (no gsplat JIT, no custom CUDA) in v1.0. |
| CPU / RAM | 32 cores, 125 GB | OpenMVS CPU densify/refine is feasible; no need for sub-scene splitting at 100–200 images. |
| Python | System 3.12.3; `uv` 0.11.8; no conda on PATH | Build the project env with `uv venv --python 3.11`; put the flat OpenMVS directory on `PATH` (no `LD_LIBRARY_PATH` needed). |
| Installed Python libs | numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, matplotlib 3.10.9, PyYAML 6.0.1, Pillow 12.2.0 | Missing: numba, trimesh, open3d, torch, cv2, pycolmap. Nothing blocking. |
| Existing COLMAP | `/data-20T/conda_envs/colmap` = **COLMAP 4.0.4, built without CUDA** (Python 3.13) | Good for `feature_extractor`/`exhaustive_matcher`/`point_triangulator`/`model_converter`; **cannot** run `patch_match_stereo`. Use `--FeatureExtraction.use_gpu 0` / `--FeatureMatching.use_gpu 0`. |
| Existing MASt3R env | `/data-20T/conda_envs/mast3r` with **torch 2.5.1 + CUDA 12.1** | Available if you ever need a learned-depth diagnostic; not part of v1.0. |
| Network to GitHub | `github.com` (release download URL) is unreachable from this host, but `api.github.com` and `release-assets.githubusercontent.com` work | Download the OpenMVS asset via the API asset endpoint (`/repos/.../releases/assets/<id>` with `Accept: application/octet-stream`), not the `github.com/.../releases/download/...` URL. Verified. |
| OpenMVS bundle | `OpenMVS_Ubuntu_x64.zip` (165 MB, v2.4.0) → flat directory of ~40 MB executables | **Verified on this host:** `-h` runs for `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`; no `LD_LIBRARY_PATH` needed; **no `CreateStructure`**. |
| KITTI-360 data | `/data-20T/kitti/data_3d_raw` (136 GB, all sequences, Velodyne + SICK) and `data_3d_bboxes` (419 MB) | **LiDAR GT is already local.** But there is **no** `calibration/`, `data_poses/`, `data_2d_raw/`, or `data_2d_semantics/` — those must be downloaded for the chosen sequence. |
| Disk | `/` 59 GB free; `/data-20T` 44 GB free | Tight. Download only: calibration + `data_poses` + `data_2d_raw` for image_00/01 + `data_2d_semantics` for one sequence (expect ~10–20 GB). Do not fetch the full dataset. |

**Note on the KITTI-360 devkit:** install path is `pip install git+https://github.com/autonomousvision/kitti360Scripts.git` (MIT), but several helpers are Python 2. Vendor/port only `data.py`, `annotation.py`, `labels.py`, `ply.py` and the `curlVelodyneData` math; do not make the whole package a runtime dependency.

---

## Installation

```bash
# 1. Python environment (3.11 per PROJECT.md; 3.12 also works)
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install \
  "numpy==2.4.4" "scipy==1.17.1" "numba==0.68.1" \
  "open3d-cpu==0.20.0" "trimesh==5.2.0" \
  "pandas==3.0.2" "matplotlib==3.10.9" "pyyaml==6.0.1" "pillow==12.2.0" \
  "psutil==7.2.2" "tqdm==4.70.1" \
  "pytest==9.2.0" "ruff==0.17.3"

# Optional fallback / later diagnostics
# uv pip install "pycolmap==4.2.1"
# uv pip install "opencv-python-headless==5.0.*"

# 2. OpenMVS 2.4.0 (prebuilt; no build). The bundle is FLAT (no bin/ or lib/).
#    github.com is unreachable on this host, so use the API asset endpoint (verified):
mkdir -p third_party && cd third_party
AID=$(curl -sL https://api.github.com/repos/cdcseacave/openMVS/releases/latest \
  | python3 -c "import sys,json;print([a['id'] for a in json.load(sys.stdin)['assets'] if a['name']=='OpenMVS_Ubuntu_x64.zip'][0])")
curl -sL -H "Accept: application/octet-stream" -o OpenMVS_Ubuntu_x64.zip \
  "https://api.github.com/repos/cdcseacave/openMVS/releases/assets/$AID"
unzip -q OpenMVS_Ubuntu_x64.zip -d openmvs-2.4.0
ls openmvs-2.4.0    # InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, RefineMesh, ...
```

```bash
# 3. OPTIONAL: COLMAP 4.2.1 with CUDA, only if the fallback is activated.
#    (conda-forge publishes cpu, cuda_129 and cuda_134 variants; pick one.)
# conda create -n colmap-cuda -c conda-forge "colmap=4.2.1" "cuda-version=12.9"

# 4. OPTIONAL: KITTI-360 devkit for reference/validation
# pip install "git+https://github.com/autonomousvision/kitti360Scripts.git"
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| COLMAP pose ingest + OpenMVS mesh (recommended) | COLMAP 4.2.1 dense + `poisson_mesher` / `delaunay_mesher` | If the COLMAP→OpenMVS handoff proves brittle, or you want a single toolchain and already have a CUDA COLMAP build. Add `--PoissonMeshing.trim` tuning and expect hole-closing. |
| OpenMVS densify | COLMAP `patch_match_stereo` depth → OpenMVS `ReconstructMesh` (via DMAP) | When 0–20 m depth accuracy is the bottleneck; COLMAP has the better depth stage. |
| Own Numba BVH | Open3D `RaycastingScene` as the primary caster | Only as a temporary performance reference; the milestone explicitly requires Möller–Trumbore for the P2 GPU port. |
| `open3d-cpu` | full `open3d` (CUDA wheel) | If you later need Open3D's CUDA/SYCL features; not for this milestone. |
| OpenMVS `RefineMesh` cleanup | `pymeshlab` / Meshlab GUI | If Open3D/trimesh cleanup cannot remove spikes/holes acceptably. |
| `image_00/01` rectified | add fisheye `image_02/03` (undistort to pinhole) | If evaluable-GT coverage is unacceptably low and extra integration time exists. Fisheye has no 2D semantics and needs `cv2.fisheye` undistortion. |
| VGGT/MASt3R (deferred) | Depth Anything V2 / Metric3D v2 as depth prior | Later, only as an auxiliary depth prior for the 2DGS route — never as the metric reconstruction. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Running COLMAP SfM / `mapper` on KITTI-360 | Poses are already metric; SfM re-estimates them, changes the frame, and breaks GT-ray alignment | Import provided poses (`cam0_to_world.txt`) and use `point_triangulator` with frozen poses/intrinsics (the primary route). |
| The installed COLMAP 4.0.4 for dense MVS | It is built **without CUDA**; `patch_match_stereo` is effectively unavailable | OpenMVS CPU densify, or install `colmap=4.2.1` with the `cuda_129` variant. |
| **VGGT / MASt3R as the v1.0 reconstruction** | No mesh, up-to-scale, heavy GPU, non-commercial licenses; confounds the baseline | COLMAP pose ingest + OpenMVS mesh; keep learned methods as later diagnostics. |
| `pyembree` / `trimesh` Embree backend | Unmaintained, incompatible with NumPy 2.x builds | Open3D `RaycastingScene` (bundled Embree) as oracle; own Numba BVH as primary. |
| PyTorch3D for Chamfer/F-score | Painful build matrix; unnecessary at 1e5 points | `scipy.spatial.cKDTree` (NumPy), `torch`/`gsplat` only later. |
| `vcpkg install openmvs` (2.3.0) or a source build on day 1 | Adds CGAL/VCGLib/Boost build time for no benefit | The 2.4.0 prebuilt Ubuntu bundle; build only if the bundle fails. |
| Poisson-only meshing of raw MVS points | Closes holes and hallucinates surfaces → false hits, inflated recall | OpenMVS visibility-aware Delaunay; or COLMAP Delaunay/advancing-front; tune `--PoissonMeshing.trim` if Poisson is used. |
| 5 cm TSDF over a 0–50 m cube | ~1e9 voxels; memory/time blow-up | OpenMVS mesh for v1.0; if TSDF later, use 10–20 cm voxels or `VoxelBlockGrid` on a bounded ROI. |
| The whole KITTI-360 sequence | Thousands of frames, hundreds of GB; disk is already 94–100% full | 60–120 frame window; download only the needed cameras/poses/semantics. |
| Treating the KITTI-360 devkit as a clean Python package | Parts are Python 2 and the Cython uncurl extension needs a build | Vendor/port the specific loaders and the uncurl math into the project. |
| Installing gsplat/torch for v1.0 | No `nvcc`; gsplat JIT needs a CUDA toolkit; out of scope | Defer to the 2DGS→TSDF milestone; use the existing mast3r env only for diagnostics. |
| Decimating the mesh before the reported baseline | Changes the geometry being evaluated | Report full-resolution metrics; decimation is an ablation. |

---

## Stack Patterns by Variant

**If the COLMAP → OpenMVS import fails (the primary path):**
- Debug in this order: (1) confirm the layout `<workspace>/sparse/{cameras,images,points3D}.txt`; (2) confirm PINHOLE only (no `SIMPLE_RADIAL`/`OPENCV`); (3) confirm database `image_id`s match `images.txt`; (4) re-export with `model_converter --output_type TXT`.
- If the sparse model has too few points, lower `--Mapper.min_num_matches` or use denser features (`--SiftExtraction.max_num_features 16384`, `--SiftExtraction.estimate_affine_shape 1`), and re-run `point_triangulator`.
- Since `InterfaceCOLMAP` was verified to import a minimal PINHOLE TXT model on this host, failures are almost always layout/model/ID mismatches, not a format incompatibility.
- Budget: CPU SIFT on ~200 images is the slow part (expect tens of minutes with exhaustive matching). If that dominates the phase, install COLMAP 4.2.1's CUDA build for feature extraction/matching only, keeping the same model/import flow.

**If 0–20 m mesh accuracy misses the acceptance bar (quality upgrade):**
- Install COLMAP 4.2.1 with the `cuda_129` variant and run the full dense chain (`image_undistorter → patch_match_stereo → stereo_fusion`), then either mesh with COLMAP's `delaunay_mesher` or feed the PatchMatch depth to OpenMVS (DMAP import) and keep OpenMVS's `ReconstructMesh` + `RefineMesh`.
- Because COLMAP has the stronger depth stage while OpenMVS has the stronger mesher; compare both point clouds before committing to one mesh.

**If the mesh has too many triangles for a fast BVH:**
- Open3D quadric decimation to ≤2M triangles, rebuild the BVH, report both full and decimated metrics as an ablation.
- Because BVH build/traversal cost is the P1 bottleneck to isolate before P2.

**If low-texture road/facade regions produce large holes:**
- Accept them first (honest misses), then try OpenMVS's thin-structure/ROI options and `RefineMesh --max-face-area`; only then consider a learned depth prior as a diagnostic.
- Because hole-filling changes the hit/miss accounting you are trying to measure.

**If evaluable GT coverage from 00/01 is too low:**
- Add fisheye 02/03 with `cv2.fisheye` undistortion to a pinhole model, reconstruct all four cameras, and keep the per-camera FOV assignment. Expect extra calibration work.
- Because the GT-ray protocol only evaluates points observed by a reconstructed camera.

**If the deferred 2DGS→TSDF→MC route is activated later:**
- Add `torch` 2.x + `gsplat` 1.5.3 (it has `rasterization_2dgs`), render depth, fuse with Open3D TSDF, extract with Marching Cubes at 10–20 cm, then reuse the **same** BVH, ray `.npz` schema, and metric code.
- Because the evaluation stack is representation-agnostic by design.

---

## Milestone Fit (suggested phasing)

| Phase | Stack actually used | Notes |
|-------|---------------------|-------|
| 0 — Data prep | Python 3.11, NumPy, Pillow, PyYAML, vendored KITTI-360 loaders | Download the minimal sequence subset; port the uncurl routine; write masks and the known-pose COLMAP model. |
| 1 — GT rays + eval harness | NumPy, SciPy `cKDTree`, pandas, Matplotlib, pytest | Build the fixed ray `.npz` set and metric functions before any mesh exists; test transforms by projection. |
| 2 — Pose ingest + mesh | COLMAP 4.0.4 CPU (`feature_extractor`/`exhaustive_matcher`/`point_triangulator`), OpenMVS 2.4.0 (`InterfaceCOLMAP`/densify/mesh), Open3D + trimesh (cleanup) | Accept an open, non-watertight street-corridor mesh. |
| 3 — CPU BVH cast | Numba 0.68.1, Open3D `RaycastingScene` (oracle) | Validate on 1e4 rays against Embree, then run the full suite. |
| 4 — Metrics + artifacts | pandas, Matplotlib, psutil | `metrics.json` / `metrics.csv` / figures; fixed hardware and ray set. |
| Deferred | torch 2.x, gsplat 1.5.3, Open3D TSDF/MC | Only after v1.0 is validated; requires a CUDA toolkit (`nvcc`). |

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| Python 3.11 | numpy 2.4.4, scipy 1.17.1 | NumPy ≥2.5 and SciPy ≥1.18 require Python ≥3.12. Pin the 2.4/1.17 line on 3.11. |
| Python 3.12 (host) | numpy 2.5.4, scipy 1.18.1 | Current host interpreter; all other selected wheels support it. |
| numba 0.68.1 | numpy <2.6, Python 3.10–3.15 | Declared by PyPI metadata; NumPy 2.4/2.5 are in range. |
| open3d-cpu 0.20.0 | Python 3.10–3.14, glibc ≥2.35 (manylinux_2_35) | Host glibc 2.39. `open3d-cpu` cp311/cp312 x86_64 wheels verified on PyPI. |
| OpenMVS 2.4.0 bundle | Ubuntu 24.04.2 x64 | **Verified to execute on this host** (`InterfaceCOLMAP`/`DensifyPointCloud`/`ReconstructMesh`/`RefineMesh -h`), flat layout, no `LD_LIBRARY_PATH`; no `CreateStructure`. |
| COLMAP 4.2.1 `cuda_129` | CUDA 12.9 runtime; host driver 580.82.09 is newer than the 12.9 minimum | conda-forge variants: `cpu`, `cuda_129`, `cuda_134`. Do not mix with the CPU-only 4.0.4 env. |
| trimesh 5.2.0 | Python ≥3.10, NumPy 2.x | Major-version API changes vs. trimesh 3/4; check any copied snippets. |
| gsplat 1.5.3 (deferred) | torch 2.x + CUDA toolkit (`nvcc`) | JIT-compiles CUDA kernels at first use; host has no `nvcc`, so do not activate in v1.0. |
| VGGT (deferred) | torch 2.x, Ampere+ for bf16 | Original checkpoint non-commercial; commercial checkpoint is gated. |
| MASt3R (deferred) | torch 2.x, CUDA | CC BY-NC-SA 4.0 (non-commercial, share-alike). |

---

## Open Risks To Validate In Phase 1

1. **OpenMVS binaries run on this host — RESOLVED.** Verified 2026-10-30: `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh` all execute; the bundle is flat and self-contained. Remaining unknown is end-to-end output quality, not installability.
2. **COLMAP → OpenMVS handoff with a real triangulated model.** `InterfaceCOLMAP` imported a minimal 2-image PINHOLE TXT model, but a model with thousands of points and real image paths still needs a smoke test. Validate layout (`<workspace>/sparse/*.txt`), PINHOLE cameras, and DB image-ID consistency first.
3. **Pose/intrinsics drift during `point_triangulator`.** It inherits `--Mapper.ba_refine_*`; freeze focal length, extra params, and rig refinement, and diff the output `images.txt` against the input. If poses move, the GT-ray origins no longer match the reconstruction frame.
4. **CPU SIFT cost.** Exhaustive matching of ~200 images on CPU is the likely time sink. Measure on 10 images first; if prohibitive, install COLMAP 4.2.1 CUDA for feature extraction/matching only.
5. **Evaluable GT coverage** with only `image_00/01`. Measure it early; if it is very low, decide whether to add fisheye cameras before investing in metrics.
6. **40–60 m MVS accuracy.** Forward-motion parallax is weak; expect RMSE to grow sharply with range. Treat this as an expected baseline property, not a bug, and keep the bucketed reporting.
7. **Disk budget.** `/` has 59 GB and `/data-20T` 44 GB free. Download the minimal sequence subset and clean intermediate OpenMVS scenes (`scene_dense.mvs` can be large).
8. **ROI auto-crop.** `DensifyPointCloud`/`ReconstructMesh` crop to an estimated ROI by default. Verify the first mesh's extent against the GT point cloud; disable cropping if it removes valid corridor geometry.

---

## Sources

- COLMAP releases page — 4.2.1 latest; built-in `poisson_mesher` / `delaunay_mesher` / advancing-front mesher; `mesh_texturer`; PoissonRecon fixes — https://github.com/colmap/colmap/releases (HIGH)
- COLMAP FAQ, "Reconstruct sparse/dense model from known camera poses", "Surface mesh reconstruction", "Available functionality without GPU/CUDA" — https://colmap.readthedocs.io/en/latest/faq.html (HIGH)
- conda-forge package API — `colmap 4.2.1` (cpu/cuda_129/cuda_134 builds), `pycolmap 4.2.1`, `open3d 0.20.0` — https://api.anaconda.org/package/conda-forge/colmap (HIGH)
- OpenMVS v2.4.0 release notes (Jan 20, 2026) — prebuilt Ubuntu x64 asset, DMAP depth import, mask segmentation, ROI, Python utilities — https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0 (HIGH)
- OpenMVS Usage wiki (2.4.0 docs) — documents a `CreateStructure --import-poses-file/--import-poses-mode` workflow, pose CSV/JSON schemas, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, `MvsScalablePipeline.py` — https://github.com/cdcseacave/openMVS/wiki/Usage (HIGH for documented behavior; **note the shipped Ubuntu bundle omits `CreateStructure`**, verified locally)
- OpenMVS Building wiki — vcpkg/dependency list, prebuilt binaries, Python API — https://github.com/cdcseacave/openMVS/wiki/Building (HIGH)
- OpenMVS 2.4.0 Ubuntu bundle, verified locally (2026-10-30): flat archive (no `bin/`/`lib/`), executables run without `LD_LIBRARY_PATH`; contains `InterfaceCOLMAP`/`DensifyPointCloud`/`ReconstructMesh`/`RefineMesh`/`TextureMesh`/`TransformScene`/`Viewer`/`Tests`; **no `CreateStructure`/`ExtractKeyframes`**; `InterfaceCOLMAP` imported a 2-image PINHOLE TXT model; `DensifyPointCloud -h` documents `.mask.png` masks + `--ignore-mask-label`; `RefineMesh -m` is `--mesh-file` (mask flag is long-only) — local command output (HIGH)
- COLMAP 4.0.4 CPU build, verified locally: `feature_extractor`/`exhaustive_matcher`/`point_triangulator`/`model_converter` available; GPU flags are `--FeatureExtraction.use_gpu` / `--FeatureMatching.use_gpu` — local command output (HIGH)
- Hermann et al., "Depth estimation and 3D reconstruction from UAV-borne imagery: Evaluation on the UseGeo dataset", ISPRS Open J. Photogramm. Remote Sens. 13:100065, 2024 — COLMAP best depth, OpenMVS lowest mesh error — https://doi.org/10.1016/j.ophoto.2024.100065 (MEDIUM; UAV scenes, not KITTI-360)
- Waymo Open Dataset static-scene surface-reconstruction leaderboard — OpenMVS listed as a competitive explicit-MVS baseline — https://sota2.com/research/sota/surface-reconstruction-on-waymo-open-dataset-static-scenes (LOW–MEDIUM; aggregator)
- Open3D 0.20.0 docs — Python 3.10–3.14 support; `RaycastingScene.cast_rays` returning `t_hit` / `primitive_ids` / `primitive_uvs` — https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html (HIGH)
- PyPI JSON API — current versions and `requires_python` for numpy, scipy, numba, open3d-cpu, trimesh, pycolmap, gsplat, pytest, ruff, psutil, tqdm — https://pypi.org/pypi/<package>/json (HIGH)
- KITTI-360 documentation — folder structure, rectified images, `perspective.txt` (`P_rect_00/01`, `R_rect_00/01`), `cam0_to_world.txt`, Velodyne format, 2 perspective + 2 fisheye cameras — https://www.cvlibs.net/datasets/kitti-360/documentation.php (HIGH)
- KITTI-360 devkit (`kitti360Scripts`) — MIT; `KITTI360` loader uses `data_rect` + `P_rect_00`; `curlVelodyneData` ego-motion compensation; `labels.py` dynamic classes — https://github.com/autonomousvision/kitti360Scripts (HIGH)
- VGGT README — CVPR 2025 best paper; outputs poses/depth/point maps; license update; frame-memory fix; COLMAP-format export — https://github.com/facebookresearch/vggt (HIGH for capabilities, MEDIUM for driving-scene quality)
- MASt3R LICENSE — CC BY-NC-SA 4.0 — https://github.com/naver/mast3r/blob/main/LICENSE (HIGH)
- gsplat source — `rasterization_2dgs`, Apache-2.0, 1.5.3 — https://github.com/nerfstudio-project/gsplat (HIGH)
- Local environment inventory (verified by command): Ubuntu 24.04.2 / glibc 2.39, RTX 4090 48 GB driver 580.82.09, no `nvcc`, 32 cores / 125 GB RAM, Python 3.12.3, `uv` 0.11.8, COLMAP 4.0.4 CPU-only env, MASt3R env with torch 2.5.1+cu121, KITTI-360 `data_3d_raw` (136 GB) + `data_3d_bboxes` present but no `calibration/`/`data_poses/`/`data_2d_raw`/`data_2d_semantics`, disk 59 GB free on `/` and 44 GB on `/data-20T` — (HIGH)

---

*Stack research for: mesh-only KITTI-360 reconstruction + GT-ray LiDAR evaluation*
*Researched: 2026-10-30*
