# Pitfalls Research

**Domain:** Mesh-only reconstruction + GT-defined-ray LiDAR evaluation on KITTI-360
**Project:** Hybrid Mesh–Gaussian LiDAR Simulation — Milestone v1.0
**Researched:** 2026-10-30
**Confidence:** HIGH for KITTI-360 data/calibration conventions (official docs + devkit source); HIGH for COLMAP/OpenMVS/Open3D behaviours (official docs/source); MEDIUM for derived metric-degeneracy analysis (first-principles, not a single published source); LOW where flagged.

**Scope of this file:** mistakes specific to *adding* a mesh-only reconstruction and a GT-defined-ray evaluator to an RGB-reconstruction + LiDAR-evaluation system. Generic ML/software pitfalls are deliberately excluded.

**How the roadmap should use this file:** every critical pitfall carries a **Phase** and a **Gate**. The phase names below are the recommended shape; the roadmap may renumber, but the *gate contents* should survive renumbering. The final mapping table is the compact checklist.

---

## Proposed Phase & Gate Scheme

| Phase | Name | Deliverable | Gate |
|---|---|---|---|
| **P1** | Data foundation | Sub-segment chosen; calibration chain implemented; dynamic masks; per-frame static GT; frozen ray set | **G1 Calibration**, **G2 Ray-set** |
| **P2** | Mesh reconstruction | COLMAP model from known poses → dense MVS → fused cloud → mesh | — |
| **P3** | Mesh validation & cleanup | Scale/frame assertions, Poisson/TSDF parameters, decimation, camera-outside test | **G3 Mesh** |
| **P4** | GT-ray casting engine | Backend-agnostic `RaySet` → `cast()` contract; CPU BVH; hit/miss accounting | **G4 Raycast** |
| **P5** | Metrics harness | Range, CD, F-score, P/R, buckets, per-class, throughput; machine-readable artifacts | **G5 Metrics** |
| **P6** | Reproducibility & report | Manifest, frozen artifacts, reruns, plots, claims audit | **G5** re-run |

**Gate definitions (recommended pass/fail thresholds — tune once, then freeze):**

- **G1 Calibration gate**
  - Transform round-trip `inv(T) @ T` error < 1e-9; every `R` satisfies `det(R) = +1`, `RᵀR = I` (1e-6).
  - Projected static LiDAR edges align with image edges to **< 2 px** at 10–30 m on ≥ 3 frames (manual or semi-automated overlay).
  - Fitted camera height above the local ground plane is **1.4–1.9 m** (KITTI-360 rig is car-mounted).
  - Camera height is positive and stable across the frame range; no sign flips.
- **G2 Ray-set gate**
  - Every ray satisfies `z_cam > 0`, projects inside `[0,W)×[0,H)`, `r_min ≤ r_gt ≤ r_max` (e.g. 1.5–50 m), and has exactly one `camera_id`.
  - The GT waterfall counts (raw → ego → dynamic → out-of-FOV → occluded → evaluated) sum exactly; percentages are logged.
  - Ray set has a SHA-256 and is frozen before reconstruction starts.
- **G3 Mesh gate**
  - Mesh bbox vs GT extent ratio in **0.8–1.2** on all axes; units are metres.
  - **Camera-outside test passes:** a ray from each camera origin toward the scene must not hit within 1 m on all 6 axis directions; hit rate in the forward cone is 0.3–0.9 (not ≈ 1.0).
  - Degenerate (zero-area) triangles < 0.1%; one connected component holds > 95% of triangles.
  - Median absolute range error against GT on one frame **< 1 m** before parameter tuning; if > 5 m, stop and debug frames, not parameters.
- **G4 Raycast gate**
  - Analytic sphere/plane/box raycast tests match closed-form `t` to **< 1e-6 m**.
  - Explicit tests for: miss, ray parallel to triangle, ray hitting an edge, origin inside a closed mesh, negative `t`, `t < t_min`.
  - `t_hit` is a metric distance (normalized direction or `‖hit − origin‖`), verified by construction.
  - Cross-check the production BVH against a simple reference Möller–Trumbore on 10k random rays; disagreements only within numerical epsilon of edges.
- **G5 Metrics gate**
  - Formulas for CD, F-score, P/R, buckets, and miss handling are written in code *and* in the report, with units.
  - All thresholds (F@5/20/50 cm) and all buckets (0–20/20–40/40–50, plus 50–60 as out-of-primary) reported with counts.
  - `inf`/`NaN` never enters a mean; misses are counted and reported separately.
  - Metrics regenerate byte-identically from `ray_set.parquet` + `mesh.ply` + manifest alone.

---

## Top Six If You Only Fix Six Things

1. **Get the calibration chain right once, and gate it with a projection overlay** (C1, C2). Every downstream number is wrong otherwise.
2. **Motion-compensate the Velodyne scan before using it as GT** (C3). At 40 km/h the sweep spans ~1.1 m — larger than the entire expected error budget.
3. **Never use the accumulated `data_3d_semantics` clouds as per-frame GT** (C4). They are multi-frame, world-frame, and include occluded surfaces.
4. **Report accuracy *and* coverage together** (C16). Range MAE on hits alone rewards holes.
5. **Kill Poisson ballooning with density trimming + a camera-outside test** (C10). A closed shell can make every ray "hit" fake geometry.
6. **Freeze the ray set before reconstruction** (C20). Otherwise the evaluator silently drifts and the results are not reproducible.

**Plus one scope decision:** build the v1.0 mesh from an open-source MVS route, not from 2DGS depth (C21). The README's 2DGS→TSDF recipe belongs to the later ablation, and using it now makes the "mesh-only" baseline neither independent nor mesh-only.

---

## Critical Pitfalls

### C1: Inverting or mis-composing the KITTI-360 calibration chain

**What goes wrong:**
The mesh, the GT points, and the rays end up in three different frames. Typical variants: treating `calib_cam_to_pose.txt` as pose→camera; treating `poses.txt` as camera→world; treating `calib_cam_to_velo.txt` as velo→camera. The scene is rotated ~180° about the forward axis, mirrored, or offset by the sensor lever arm. Because street scenes are locally symmetric, the result can look plausible while every metric is wrong.

**Why it happens:**
The file names read "camera to pose" but people assume "pose to camera". KITTI-360 is *not* KITTI raw: `poses.txt` stores **IMU→world**, not cam0→world, and `calib_cam_to_velo.txt` stores **cam0→Velodyne**, not Velodyne→cam0. The subscript convention `T_CL` in the project README ("maps LiDAR to camera") contradicts the usual robotics reading "C←L", inviting an inversion.

**How to avoid:**
Write the chain down once, in one module, and never re-derive it inline:

```text
T_world_imu   = poses.txt                      # IMU -> world (3x4)
T_imu_camk    = calib_cam_to_pose.txt[k]       # camera -> IMU (3x4)
T_world_camk  = T_world_imu @ inv(T_imu_camk)  # camera -> world
T_cam0_velo   = inv(calib_cam_to_velo.txt)     # file is cam0 -> velo
T_world_velo  = T_world_cam0 @ T_cam0_velo

# Rectified image_00 convenience file:
T_world_cam0_rect = cam0_to_world.txt          # rectified cam0 -> world (4x4)
```

- Use `kitti360Scripts` loaders (`loadCalibrationCameraToPose`, `loadCalibrationRigid`) rather than hand-parsing.
- Pad 3×4 to 4×4 with `[0,0,0,1]` before inverting; never call `inv` on a 3×4.
- Unit-test the round trip and assert `det(R) > 0`.
- Gate with the G1 overlay and camera-height checks.

**Warning signs:**
- Camera height negative, ~0.3 m, or > 3 m above the local ground.
- Mesh rendered upside down or mirrored (check a road sign or building corner, not symmetric geometry).
- LiDAR projected onto images is offset by more than ~2 px everywhere.
- `np.linalg.inv` on a reshaped 3×4 raises or silently produces garbage.
- Depth along the optical axis (`z_cam`) is negative for most front-facing points.

**Phase to address:**
P1 — Data foundation. **Gate G1.**

---

### C2: Mixing rectified and unrectified streams (and the image_01 baseline)

**What goes wrong:**
`data_rect` images are rectified; `data_rgb` images are raw. Using `P_rect_*` with raw images, applying `R_rect_*` twice, or using the rectified pose with unrectified intrinsics shifts rays by a fraction of a degree. At 50 m, 0.1° is 8.7 cm — the same order as the 5–20 cm error the project is trying to measure. For image_01, `P_rect_01` already encodes the stereo baseline in its `Tx`; using the physical image_01 pose *and* keeping the `Tx` term double-counts it, or using image_00's pose for image_01 rays omits it (≈0.5–0.6 m lateral error).

**Why it happens:**
KITTI's projection convention (`P_rect @ R_rect @ T_cam_velo @ p_velo`) is often copied without understanding which matrix belongs to which image stream. The `P_rect_01` translation term is easy to overlook because it lives inside the 3×4 matrix rather than in an extrinsic.

**How to avoid:**
- Decide at P1: use `data_rect` + `P_rect_*` + rectified poses (`cam0_to_world.txt` for image_00), **or** `data_rgb` + `R_rect` + unrectified poses. Never mix.
- For ray generation from image_i, use `K_i = P_rect_i[:, :3]` (the 3×3 block) and the **physical pose of camera i**; do not add the `Tx` baseline again.
- For image_01 rectified pose: `T_world_cam1_rect = T_world_cam1_unrect @ inv(R_rect_01)`.
- Assert that projected GT points land on the correct image content (lane markings, sign edges) for both cameras.

**Warning signs:**
- image_00 metrics look fine but image_01 has a constant lateral/range bias.
- Projection overlay is good on image_00, off by tens of pixels on image_01.
- Ray count and FOV allocation for image_01 look implausible.
- Re-running with `data_rgb` instead of `data_rect` changes results by more than a few cm.

**Phase to address:**
P1 — Data foundation. **Gate G1.**

---

### C3: Ignoring Velodyne motion distortion ("curled" scans)

**What goes wrong:**
The HDL-64E measures continuously during a ~0.1 s 360° sweep. The `.bin` stores all points in one Velodyne frame, so points measured at different azimuths correspond to different ego poses. At 40 km/h (~11 m/s), the sweep spans ~1.1 m; rear-facing points can be offset by ~0.5 m relative to the camera trigger time. Using these points as GT produces a range error that correlates with azimuth and grows with speed — a systematic bias larger than the reconstruction error being measured.

**Why it happens:**
The KITTI-360 docs say only "defined in Velodyne coordinates"; the KITTI raw readme warns about "untwisting", but most KITTI-360 tutorials skip it. The devkit ships a Cython motion-compensation helper (`curlVelodyneData`) that is easy to miss and requires a build step.

**How to avoid:**
- Build `kitti360Scripts` with Cython (`CYTHONIZE_EVAL= python setup.py build_ext --inplace`) and use `curlVelodyneData` (or reimplement the same azimuth-to-time linear interpolation between sweep-start and sweep-end poses) to unwrap each scan to the camera timestamp before transforming to world.
- Determine the sweep-start timestamp convention and, if needed, calibrate a constant time offset by minimizing LiDAR-to-image reprojection error over ±50 ms.
- Alternatively, restrict to low-speed segments (< 20 km/h) and **report** the residual bias rather than pretending it is absent.
- Apply the same compensated points everywhere: GT rays, dynamic masks, projection overlays, and mesh validation.

**Warning signs:**
- Mean **signed** range error plotted against azimuth is sinusoidal.
- Error scales with ego speed across frames.
- A straight wall becomes a "banana" in BEV when accumulated without compensation.
- LiDAR/image overlay is sharp at one side of the image and blurred at the other.

**Phase to address:**
P1 — Data foundation (GT extraction). Re-check in P3 if LiDAR is projected for validation. **Gate G2.**

---

### C4: Using accumulated `data_3d_semantics` clouds as per-frame GT

**What goes wrong:**
`data_3d_semantics/.../static/{start}_{end}.ply` are **accumulated over a window of frames**, in the **world** frame, with semantic labels. They are the only labelled clouds, which makes them tempting. Using them as GT for a single frame breaks the protocol: they contain points from other times and viewpoints, surfaces occluded from the camera, and (in the `dynamic` files) moving objects. Recall becomes meaningless and range errors are often *better* than reality because the accumulated cloud densifies surfaces the camera never saw.

**Why it happens:**
The semantic labels live only on the accumulated clouds, and "GT" colloquially means "the labelled thing". The per-frame `.bin` files have no semantic labels, so the path of least resistance is to use the PLY.

**How to avoid:**
- GT rays come **only** from `data_3d_raw/.../velodyne_points/data/{frame}.bin`, motion-compensated (C3), transformed by the per-frame pose.
- Use the accumulated semantic clouds only to transfer labels to per-frame points (nearest neighbour within a small radius, e.g. 5 cm, with a confidence threshold) or to build dynamic masks.
- Keep `raw/` and `semantics/` in separate directories and make the ray-set builder assert it is reading the raw `.bin`.
- Sanity-check the per-frame point count: ~0.17M for HDL-64E (64 × ~2650), not millions.

**Warning signs:**
- GT points per frame are far above ~0.17M.
- GT contains points behind walls or on both sides of a thin surface.
- The ray-set builder never reads a timestamp or pose per frame.
- Metrics are suspiciously good on occluded facades.

**Phase to address:**
P1 — Data foundation. **Gate G2.**

---

### C5: Wrong sensor-count and FOV assumption ("4 perspective cameras")

**What goes wrong:**
The project plan describes KITTI-360 as "4 pinhole + 2 fisheye" (README §6) and allocates GT points across four perspective cameras. KITTI-360 actually has **two perspective cameras** (`image_00`, `image_01`, a stereo pair, ≈80–90° HFOV) and **two fisheye cameras** (`image_02`, `image_03`, side coverage). A four-camera loader fails or silently duplicates the stereo pair; including fisheye without a fisheye model makes projection and ray allocation wrong near the edges; including fisheye in MVS requires undistortion/remapping and loses the wide FOV that motivated it.

**Why it happens:**
KITTI-360 is often described alongside KITTI raw (4 cameras), and both have `image_00..03`. The naming overlaps, the sensor types do not.

**How to avoid:**
- Correct the sensor model in P1: **2 perspective + 2 fisheye**, per the official documentation.
- For v1.0, use `image_00`/`image_01` only and restrict GT rays to their union FOV. Document the evaluated angular domain explicitly.
- Use `P_rect_00`/`P_rect_01` and the stereo baseline; do not invent a third/fourth perspective camera.
- If fisheye is later added: use `image_02.yaml`/`image_03.yaml` intrinsics and a fisheye/omnidirectional model, and note that 2D semantic labels are **not** provided for fisheye images, so dynamic masking cannot be transferred directly.

**Warning signs:**
- `calib_cam_to_pose.txt` has 4 rows but `image_02/03` come with YAML intrinsics, not `P_rect`.
- HFOV of the "four cameras" is ≈80–90°, not a 360° surround.
- Fisheye projections match only near the image centre.
- Dynamic masks exist for image_00/01 and are missing for 02/03.

**Phase to address:**
P1 — Data foundation. **Gate G2.**

---

### C6: Asymmetric or partial dynamic masking

**What goes wrong:**
Three classic variants: (a) mask dynamic pixels in training images but keep dynamic GT points → guaranteed misses and multi-metre errors; (b) remove dynamic GT points but leave moving objects in images → ghost blobs in the mesh that occlude static GT points behind them; (c) mask `image_00` but not `image_01` → one clean and one contaminated view, and the stereo pair becomes inconsistent. Ego-vehicle points are a fourth variant: they are static relative to the sensor but not reconstructible from images, and no semantic label marks them.

**Why it happens:**
Images, LiDAR, and semantic labels have different availability and formats (2D panoptic PNGs vs. 3D bboxes vs. per-point timestamps). It is easy to apply each mask in only one place.

**How to avoid:**
- Build one `dynamic_mask` artifact per frame and apply it to **all** consumers: images (before MVS), GT points, LiDAR projections, and mesh validation.
- Combine sources: 2D panoptic labels (image_00/01 only, train/val frames only), 3D bboxes with per-object timestamps, and the accumulated `dynamic` cloud with per-point timestamps.
- Remove ego points by a 3D ROI in Velodyne coordinates (e.g. a box around the sensor, below z ≈ 0.2 m, within ±2.5 m), **not** by a blanket range threshold (see M1).
- Log the residual dynamic fraction (GT points still within 1 m of a dynamic bbox) after masking.

**Warning signs:**
- Car-shaped or pedestrian-shaped blobs in the mesh.
- GT waterfall shows 0% dynamic but the range error histogram has isolated 1–10 m spikes.
- A dynamic bbox intersects the mesh.
- Metrics differ strongly between image_00 and image_01 on the same frames.

**Phase to address:**
P1 — Data foundation; re-audit at P3. **Gate G2.**

---

### C7: MVS failure on textureless road, sky, and glass — exactly where the metric is densest

**What goes wrong:**
PatchMatch-style MVS depends on photometric consistency. Asphalt at grazing incidence, sky, and glass are ambiguous. The road is the largest and nearest surface and dominates the 0–20 m bucket, so its failure is maximally damaging: either holes exactly where the evaluation is densest, or "orange-peel" depth. Sky produces spurious far points and mesh caps that later become false occluders.

**Why it happens:**
MVS benchmarks reward facade/building accuracy; street-scene tutorials rarely discuss the road. Sky is not a surface, but nothing in the MVS pipeline knows that.

**How to avoid:**
- Mask sky with the semantic label **before** dense stereo or before fusion.
- Use COLMAP geometric consistency (`--PatchMatchStereo.geom_consistency 1`) and filter by support in ≥ 2–3 views; weight fusion by photometric confidence.
- For the road, prefer a slanted-plane/plane-sweep prior or a robust local ground plane fit; if you fill road holes with a plane, **declare it as a prior**, not as observation.
- Evaluate the road as its own semantic bucket so a bad road does not hide behind facades (and vice versa).

**Warning signs:**
- Depth coverage on the road is far lower than on facades.
- BEV mesh shows stripes or holes along the driving direction.
- Triangles beyond ~150 m with no semantic support (sky/sky-dome).
- Road height noise of > 10 cm at 10–20 m.

**Phase to address:**
P2 — Reconstruction; validate at P3. **Gate G3.**

---

### C8: MVS on moving objects and exposure changes → ghost geometry

**What goes wrong:**
A car present in frame *t* but not *t+1* produces inconsistent depth and a ghost blob. Moving shadows and auto-exposure changes break NCC. Wind moves vegetation between frames. The fused cloud contains floating components, double walls, and spikes. These are not just ugly: they become false occluders that change range errors and recall.

**Why it happens:**
Dense MVS assumes a static Lambertian scene; street sequences violate all three assumptions. Masking is partial (C6), and the remaining dynamic content is exactly the high-contrast content that attracts PatchMatch.

**How to avoid:**
- Mask dynamic pixels before MVS (C6); prefer overcast, consistent-exposure frames.
- Filter the fused cloud by multi-view support (≥ 2–3 views), reprojection error, and connected-component size.
- Remove fused points inside dynamic 3D bboxes (transformed per frame) before meshing.
- Keep the largest connected component; discard components smaller than a threshold in m³.
- Inspect the mesh from a bird's-eye view; ghost blobs are obvious there and invisible in a forward render.

**Warning signs:**
- Floating connected components in mid-air.
- Double facades / "double wall" surfaces.
- Mesh density spikes where a car was present in only some frames.
- Per-view depth maps disagree by > 0.5 m at the same 3D point.

**Phase to address:**
P2 — Reconstruction; validate at P3. **Gate G3.**

---

### C9: Mesh in a different frame or scale than the evaluation rays

**What goes wrong:**
If MVS is driven through COLMAP SfM (because OpenMVS wants a COLMAP model), the mesh lives in COLMAP's arbitrary similarity frame — arbitrary origin, orientation, and scale. A 180° yaw or a 1.3× scale still yields a "plausible" reconstruction. Rays defined by KITTI poses then hit the wrong place; or worse, the mesh is rescaled/aligned to the LiDAR to "fix" it, which leaks GT into reconstruction.

**Why it happens:**
COLMAP's default workflow is SfM → MVS, and most tutorials start there. The project's constraint is "use provided metric poses, do not run COLMAP SfM", but the MVS tooling still expects COLMAP's sparse model format.

**How to avoid:**
- **Fabricate the COLMAP sparse model from KITTI poses**: write `cameras.txt`, `images.txt` (remember: COLMAP stores **world→camera**, `qvec,tvec`), and `points3D.txt`; skip the mapper entirely. If the MVS stage needs sparse points for depth ranges, triangulate them from image matches with the fixed poses (**never from LiDAR**) or set explicit `--PatchMatchStereo.depth_min/depth_max`.
- Alternatively use OpenMVS with an importer that accepts known poses, or build the TSDF directly from MVS depth maps expressed in the KITTI world frame.
- Assert: COLMAP camera centres match `cam0_to_world.txt` to < 1e-6, mesh bbox matches GT extent within a few percent, and `det(R) > 0`.
- Check pose continuity across the **whole** window (frame-to-frame translation/rotation and reprojection consistency at 10 m, 30 m, 50 m), not just one frame. Provided SfM poses can drift or jump locally even when they are globally metric.
- If an SfM route is unavoidable, solve **one** similarity transform, record it in the manifest, and never tune it against metrics. (Better: don't.)
- Do not use LiDAR to fix scale, orientation, or hole filling. Validate only after freezing the mesh.

**Warning signs:**
- Mesh bbox coordinates in the hundreds (normalized COLMAP units) instead of metres.
- Camera centres do not coincide with the trajectory file.
- A single global similarity transform "fixes" the metrics.
- Scale factor ≈ 0.3–3 between mesh and LiDAR.
- The reconstructed scene is rotated relative to the road.

**Phase to address:**
P2 — Reconstruction; enforced at P3. **Gate G3.**

---

### C10: Poisson ballooning, over-smoothing, and the camera inside the mesh

**What goes wrong:**
Poisson reconstruction **always** produces a closed surface; it stretches a low-density "skin" across unobserved regions (behind buildings, above the road, behind the camera). Without a density trim, the camera can end up **inside** a closed shell. Forward rays then hit the observed facade (plausible), but unobserved directions hit a fake inner surface at small `t`, and holes are filled with smooth geometry that hides reconstruction failures. Over-smoothing also erases thin structures and rounds corners.

**Why it happens:**
"Watertight" sounds like a quality guarantee, and Poisson's default output looks complete. The density field is returned as a second output and is routinely discarded.

**How to avoid:**
- Keep the density field and trim vertices below a quantile (commonly 5–10%, e.g. `densities < np.quantile(densities, 0.10)`); raise the quantile until the unwanted fill disappears.
- Crop the mesh to the observed corridor and the camera frustum envelope; do not mesh a 1 km box for a 200 m segment.
- Store per-vertex density/confidence for later analysis and for the hybrid router.
- Prefer a **visible-surface** mesh (TSDF + Marching Cubes, or OpenMVS `ReconstructMesh`) when the goal is a ray-castable surface rather than a solid. Holes are more honest than balloons.
- Run the camera-outside test (G3) and an all-directions hit-rate test.

**Warning signs:**
- Hit rate ≈ 100% in every direction, including behind the camera.
- Camera-outside test fails (hits within 1 m on multiple axes).
- Mesh volume >> expected volume; surfaces behind the trajectory.
- Silhouettes rounded; poles/wires disappear.
- A large low-density mode in the density histogram.

**Phase to address:**
P3 — Mesh validation & cleanup. **Gate G3.**

---

### C11: Mesh resolution and TSDF truncation inconsistent with the metric thresholds

**What goes wrong:**
5 cm voxels + 10 cm decimation + Poisson smoothing give an error floor above 5 cm, so **F@5 cm measures the meshing, not the reconstruction**. Conversely, `sdf_trunc = 0.04` with MVS depth noise of 0.1–0.3 m rejects most valid observations and produces holes; raising truncation too far fuses unrelated surfaces. Decimation error is often larger than the claimed tolerance and is never reported.

**Why it happens:**
Defaults are tuned for clean indoor RGB-D or small objects, not 50 m street scenes. The metric thresholds are chosen independently of the mesh pipeline's resolution.

**How to avoid:**
- Set `voxel_length ≤ τ_min / 2` (e.g. 2–2.5 cm if you intend to report F@5 cm).
- Set `sdf_trunc ≈ 3–5×` the measured MVS depth noise; estimate noise from cross-view depth agreement, not from a default.
- Measure and report decimation Hausdorff error; if it exceeds τ_min, say so and demote F@5 to a secondary metric.
- Report mesh resolution, `#triangles`, and `#vertices` next to every F-score.
- If compute forces 10 cm voxels, drop F@5 and lead with F@20/50 — do not report a metric the pipeline cannot resolve.

**Warning signs:**
- F@5 ≈ 0 for every method (metric floor, not method difference).
- Error histogram quantized at the voxel size.
- Holes appear when truncation is lowered; double surfaces when raised.
- Decimation changes CD by > 10%.

**Phase to address:**
P3 — Mesh validation & cleanup; metric lock at P5. **Gate G3/G5.**

---

### C12: Coordinate-frame and ray-direction convention errors

**What goes wrong:**
`T_CL` is read as LiDAR→camera in one file and camera→LiDAR in another. Renderers disagree on forward axis (OpenGL/pyrender look −z; OpenCV looks +z). Camera is y-down, Velodyne is z-up, IMU is y-right/z-down, world is y-left/z-up. A rotation is inverted as `Rᵀ` but the translation is left as `t` instead of `−Rᵀt`. Outcomes range from total failure to a mirrored scene that "looks fine" and yields plausible-looking numbers.

**Why it happens:**
Every library picks a different convention, and the project's own README uses `p_C = T_CL p_L` (LiDAR→camera) while the subscript conventionally means "C←L". Silent transposes are the norm in copied code.

**How to avoid:**
- One `frames.py` with named 4×4 transforms and docstrings; `T_world_cam` is the only source of truth, everything else is derived by explicit `inv`.
- Write the ray equation once: `p(t) = o + t·d`, `d` unit, `t ≥ t_min`.
- Assert `det(R) = +1`, `RᵀR = I`, and round-trip error < 1e-9 at load time.
- Unit-test with a known point: `inv(T) @ (T @ p) == p`.
- If a renderer is involved, add an axis-conversion shim and test it with a non-symmetric object.
- Name variables `T_world_cam`, never `T` or `pose`.

**Warning signs:**
- Hit rate ≈ 0 or ≈ 1.
- Scene rotated 90°/180°, or mirrored text on signs.
- Camera height negative while the overlay looks "almost right".
- Distances plausible but CD/F-score nonsense.
- Different results between two backends on the same mesh.

**Phase to address:**
P1 (data) and P4 (ray casting). **Gate G1/G4.**

---

### C13: Miss, FOV, and range-denominator errors

**What goes wrong:**
Five distinct bugs that all surface as "miss rate is weird":
1. Rays to GT points **behind the camera** (`z_cam ≤ 0`) are cast anyway and hit geometry behind the image plane.
2. Rays to points **outside the image** are cast and counted as misses, so recall includes directions no camera observed.
3. `r_gt` is computed as `‖p_L‖` (LiDAR range) while `r_pred` is a camera-origin distance — reintroducing the 0.5–0.8 m parallax bias the protocol was designed to remove.
4. `inf` misses are fed into `np.mean` → `NaN`, or into `np.median` → `inf`.
5. `t_max = 50` makes 50.1 m points "misses" while the bucket is labelled 40–60.

**Why it happens:**
The protocol is mathematically clean, so implementation shortcuts look harmless. FOV filtering and miss accounting are easy to forget because they do not appear in the equations.

**How to avoid:**
- Compute `p_C = T_cam_world @ p_world`, `r_gt = ‖p_C‖`, `d = p_C / r_gt` — always from the **camera** origin.
- Pre-filter the ray set: `z_cam > 0`, projection inside `[0,W)×[0,H)`, `r_min ≤ r_gt ≤ r_max`; record counts at each step (GT waterfall).
- Use masked statistics: `valid = np.isfinite(r_pred)`; never average raw arrays.
- Define `t_max` once as the protocol truncation and use the same value for miss detection and bucketing; state it.
- Report `miss_rate + hit_rate = 1` with the same denominator.

**Warning signs:**
- `NaN`/`Inf` in the metrics JSON.
- Ray count per frame ≠ in-FOV GT point count.
- Miss rate > 50% and strongly correlated with image borders.
- A constant range offset that varies with azimuth (the `r_gt = ‖p_L‖` bug).
- Bucket counts do not sum to the evaluated total.

**Phase to address:**
P1 (ray-set construction) and P4 (casting). **Gate G2/G4.**

---

### C14: Camera-origin occlusion misattributed to reconstruction

**What goes wrong:**
A GT point visible to the LiDAR may be **occluded from the camera** 0.5–0.8 m away — behind a pole, a car edge, or a curb. The ray hits the occluder first, so `r_pred << r_gt`, producing a multi-metre "error" that is not a reconstruction error at all. GT-defined rays remove *parallax* bias but not *occlusion*. The effect concentrates at depth discontinuities, exactly where thin structures live, and it dominates RMSE tails.

**Why it happens:**
The protocol is usually justified as "no parallax", and people stop there. Occlusion is a separate geometric fact and is invisible unless you look for it.

**How to avoid:**
- Compute a **camera-visibility flag using GT alone**: project all GT points into the camera, build a sparse z-buffer, and mark a point as "possibly occluded" if another GT point projects within k pixels (e.g. 2–3) and is closer by > 3σ_LiDAR.
- Exclude flagged points from the primary range/CD metrics, but report them as a separate bucket and also report a "full" variant.
- Keep the flag in the frozen ray set (`occluded: bool`) so every backend uses the same domain.
- Do not use the reconstructed depth map for this test: that would let reconstruction errors define the evaluation domain.

**Warning signs:**
- The error tail is systematically `r_pred < r_gt`.
- Errors cluster at object silhouettes and curbs.
- Excluding occluded points changes RMSE by 2× or more.
- Error is independent of reconstruction quality but tracks depth discontinuities.

**Phase to address:**
P4 — Ray-casting engine (flag construction), consumed at P5. **Gate G2.**

---

### C15: Multi-camera overlap double counting

**What goes wrong:**
`image_00` and `image_01` overlap almost completely. If both cameras generate rays for every GT point in their shared FOV, the same GT point contributes two different errors and twice the weight. Pooling the overlap inflates one camera's influence, makes aggregates depend on camera ordering, and — if fisheye views are later added — makes the overlap geometry much worse. Deduplicating by direction hash fails because the origins differ.

**Why it happens:**
"Allocate GT points by FOV" sounds like a per-camera filter, not a partition. Nothing forces a point to belong to exactly one camera.

**How to avoid:**
- Assign each GT point to exactly **one** camera at ray-set build time, deterministically:
  - require in-FOV and `z > 0`;
  - choose the camera minimising the normalized radius `((u−cx)/fx)² + ((v−cy)/fy)²` (most central projection), with a documented tie-break.
- Freeze `camera_id` per ray; never re-allocate at evaluation time.
- Report per-camera metrics separately **and** the deduplicated aggregate.
- State the overlap fraction and the assignment rule in the report.

**Warning signs:**
- Total ray count ≈ 2× the GT points in the front cone.
- Per-camera metrics differ substantially while the aggregate looks like one of them.
- Changing camera order changes the aggregate.
- Adding a second camera "improves" recall without changing the mesh.

**Phase to address:**
P1 (ray-set construction) and P4. **Gate G2.**

---

### C16: Accuracy-on-hits / hole hiding

**What goes wrong:**
Range MAE is computed only on hits. A mesh with holes — or a conservative caster — looks *more accurate* because the hard rays are absent. This is the accuracy–completeness tradeoff, and it is the easiest way to accidentally produce a flattering mesh baseline. The same trap applies to "precision" computed only over predicted points.

**Why it happens:**
The README correctly defines MAE "only over hits" and treats misses separately, but it is easy to report without the coverage number next to it.

**How to avoid:**
- Treat **(accuracy, coverage)** as a single result: MAE/RMSE/MedAE **and** hit rate / recall in the same table row.
- Report an error CDF that includes misses at a declared truncation penalty (labelled), plus the standard hit-only metrics.
- Report per-bucket and per-semantic-class accuracy **and** recall; require both.
- Make any "improved" claim conditional on coverage not dropping; otherwise call it a trade, not an improvement.
- Add a coverage–accuracy scatter/Pareto to the report, not just a single point.

**Warning signs:**
- A variant with more holes has lower MAE.
- MAE and miss rate move in opposite directions across configs.
- Recall on the ground class < 0.7 while MAE looks excellent.
- Only hit-only metrics appear in the results table.

**Phase to address:**
P5 — Metrics harness. **Gate G5.**

---

### C17: CD / F-score degeneracy and definition ambiguity

**What goes wrong:**
Two problems compound.
1. **Definition ambiguity:** CD is reported as mean vs. sum, squared vs. not, one-way vs. symmetric; F-score thresholds are written as "5" vs. "0.05"; precision/recall matching radii and one-to-one vs. many-to-one policies differ. Two papers' "CD 0.14" may not be comparable.
2. **GT-ray degeneracy:** each predicted point lies on the ray to its GT point, so pred→GT distance is bounded by the radial error and typically close to it. CD becomes a monotone-ish transform of range error, and with one point per ray, precision and recall become the same conditional error CDF (modulo the hit-rate factor). Reporting CD/F as independent evidence of geometric fidelity overstates the result.

**Why it happens:**
CD and F-score are copied from the point-cloud literature, where the two clouds are independent samples of a surface. Under GT rays they are not.

**How to avoid:**
- Write the exact formulas in code and in the report:
  - `CD-L1 = ½( mean_{p∈P} min_{q∈Q}‖p−q‖ + mean_{q∈Q} min_{p∈P}‖q−p‖ )`
  - `CD-L2` with squared distances; state **mean vs. sum** and units (m vs. m²).
  - `F@τ = 2PR/(P+R)`, `P = |{p : NN(p,Q) < τ}|/|P|`, `R = |{q : NN(q,P) < τ}|/|Q|`, τ in **metres**.
- State that under GT rays CD ≈ f(range error) and F@τ ≈ error CDF; do not present them as independent confirmations.
- Compute nearest neighbours in **one** coordinate frame (LiDAR/world) with one KD-tree; per-camera-frame NN is a different metric and must not be pooled with a global one.
- If a genuine CD is claimed, add a secondary evaluation on a fixed sensor-model ray set or a mesh-sampled point cloud.
- Never replace misses with a constant without labelling the number as "clipped CD with miss penalty".

**Warning signs:**
- CD and MAE are almost identical (expected — but only if you acknowledge it).
- "F@5cm" ≈ 1.0 (threshold given in metres vs. centimetres bug).
- Swapping P and Q changes CD (asymmetry bug or deliberate one-way metric).
- CD changes when duplicate frames are added (density/pooling bug).

**Phase to address:**
P5 — Metrics harness. **Gate G5.**

---

### C18: Grazing-incidence amplification and ground domination

**What goes wrong:**
Range error is surface offset divided by the incidence cosine: `δ_r ≈ δ_n / |n·d|`. At 40 m, the ground is seen at ~2°, so a **2 cm** mesh height error becomes **~50 cm** of range error. The 0–20 m bucket is dominated by ground returns, so aggregate MAE mostly measures the road plane's height bias — and a global vertical shift of a few centimetres can "improve" the metric without improving the mesh. Mesh vs. Gaussian comparisons then depend on the ground fraction rather than on representation quality.

**Why it happens:**
Range error is intuitive but anisotropic. Nobody plots error against incidence angle, so the amplification is invisible.

**How to avoid:**
- Report error vs. **incidence angle** and vs. **semantic class** (ground / facade / vegetation / thin structure / vehicle).
- Add a **point-to-plane** distance using GT local normals: `|(p̂ − q)·n_q|`, which removes the `1/|n·d|` amplification and measures true surface offset.
- Estimate and report a global height offset separately; do not tune it away silently.
- Keep per-class counts; never let the road's point share dominate an unweighted average.
- For later mesh-vs-Gaussian comparisons, match or stratify the semantic composition.

**Warning signs:**
- RMSE is 5–10× MAE.
- Error grows steeply with range on the ground but not on facades.
- Translating the entire mesh by 3–5 cm changes MAE by 2×.
- Error histogram is dominated by near-ground points.

**Phase to address:**
P5 — Metrics harness. **Gate G5.**

---

### C19: Cherry-picking thresholds, buckets, frames, and segments

**What goes wrong:**
Trying F@5/20/50 and reporting the best; reporting only the 0–20 m bucket because it looks good; choosing the segment or frame list after seeing metrics; dropping "bad" frames; tuning thresholds or Poisson parameters on the evaluation frames; reporting the best of N runs. Each is individually plausible and collectively destroys the milestone's evidentiary value.

**Why it happens:**
The project has many free parameters (segment, frame range, voxel size, τ, bucket) and no pre-registration step. Metric-driven iteration is normal in research, but it must not select the reported domain.

**How to avoid:**
- **Pre-register** in P1: segment ID, frame range, ray-set construction, exclusions, thresholds, buckets, aggregation, and the primary metric. Write it to `decisions.md` with a timestamp and hash.
- Freeze the ray set before reconstruction (C20).
- Report **all** thresholds and **all** buckets with counts; no "best τ".
- Split frames: a small dev set for parameter tuning, a disjoint reported set. Never tune on the reported set.
- Report mean ± std over ≥ 3 seeds/runs where nondeterminism exists.
- Keep every run's config in the results directory, including discarded ones.

**Warning signs:**
- Results exist for only one bucket or one τ.
- The frame list or segment ID changes between runs.
- The report says "we use τ = 0.2 m" but F@5 was also computed and omitted.
- Tuning history is not recoverable from the repo.

**Phase to address:**
P1 (pre-registration) and P6 (audit). **Gate G5.**

---

### C20: Reproducibility and artifact drift

**What goes wrong:**
The mesh changes silently when preprocessing changes; COLMAP/OpenMVS runs differ between machines or thread counts; library versions are unpinned; FPS depends on CPU model and thread count; the metrics cannot be regenerated from the stored mesh. Six weeks later nobody can tell which mesh produced which number.

**Why it happens:**
Multi-threaded photogrammetry and BVH libraries are not bit-reproducible by default, and derived artifacts (depth maps, fused clouds, meshes, ray sets) are large and usually left unversioned.

**How to avoid:**
- `run_manifest.json` with: git SHA, dataset release, calibration hash, frame-list hash, ray-set hash, mesh hash, library versions (COLMAP/OpenMVS/Open3D/Embree), CPU model and thread count, seeds, wall time.
- Pin threads (`OMP_NUM_THREADS`, `torch.set_num_threads`, COLMAP `--SiftExtraction.num_threads`) and `CUDA_VISIBLE_DEVICES`.
- Containerise or lock the environment; record the COLMAP/OpenMVS commit, not just "latest".
- Store large derived artifacts outside git with checksums (DVC or a run directory); never rely on filenames like `mesh_final_v3.ply`.
- Make evaluation runnable from `ray_set.parquet + mesh.ply + manifest` alone, with **no reconstruction step**.
- Run timing measurements ≥ 3 times after warm-up and report median and IQR.

**Warning signs:**
- Numbers change between runs with no code change.
- "Which mesh was this?" is unanswerable.
- Only one person can run the pipeline.
- No manifest, no hashes, no pinned versions.

**Phase to address:**
P6 — Reproducibility & report; enforced from P1 onward. **Gate G5 re-run.**

---

### C21: Reusing the README's 2DGS→TSDF mesh recipe for the "mesh-only" baseline

**What goes wrong:**
README §0.5.2 and §4.1 define the mesh as **2DGS depth → TSDF → Marching Cubes**, sharing the 2DGS training. Milestone v1.0 instead requires an **open-source MVS mesh** and explicitly defers 2DGS→TSDF→MC. If the team follows the README literally: (a) the "mesh-only" baseline is not independent — it is derived from a Gaussian model; (b) the mesh cannot be produced without first training 2DGS (GPU-hours) and debugging that stack; (c) the mesh and the Gaussian baseline share depth noise and failure modes, so mesh-vs-Gaussian comparisons are correlated and cannot support the "surface-first" argument; (d) the mesh provenance in the report becomes ambiguous.

**Why it happens:**
The README is the project's source of truth and was written for the earlier V2 plan, which included 3DGS/2DGS baselines in the same phase. The milestone narrowed scope to mesh-only but did not rewrite the reconstruction section.

**How to avoid:**
- Make the MVS mesh the **v1.0 primary** and record the route + version in the manifest.
- Keep 2DGS→TSDF→MC as a clearly labelled **later ablation** (P3), not as the baseline.
- Do not share depth maps, training runs, or checkpoints between the mesh baseline and any Gaussian baseline.
- In the report, state the mesh provenance explicitly ("COLMAP MVS from images + KITTI poses, no Gaussian model in the loop").
- If time forces a choice, choose the route that can produce a mesh **without** any Gaussian training.

**Warning signs:**
- The mesh pipeline imports `gsplat`, 2DGS, or a Gaussian checkpoint.
- The mesh is described as "watertight" and derived from rendered depth rather than MVS.
- The same depth maps feed both the mesh and the Gaussian baseline.
- The plan's first task is "train 2DGS" rather than "build the COLMAP model".

**Phase to address:**
P2 — Reconstruction (route decision); claims audit at P6. **Gate G3.**

---

## Moderate Pitfalls

### M1: Ego-ROI removal collateral damage

**What goes wrong:** Removing every point with `r < 2.5 m` is the quickest way to drop ego-vehicle returns, but it also deletes real ground, curb, and wall points at 0–2.5 m — the nearest part of the 0–20 m bucket. Removing too little leaves hood/roof points that are guaranteed misses.

**Prevention:** Remove points inside a 3D box in Velodyne coordinates (e.g. `|x| < 2.5`, `|y| < 1.2`, `-2.0 < z < 0.3` relative to the sensor), then report the residual near-field GT count. Do not use an isotropic radius. Document the ROI and the excluded range in the GT waterfall.

**Warning signs:** A suspiciously empty near-field in BEV; ego-shaped holes in the ground; miss rate spikes at 1–3 m.

**Phase:** P1. **Gate G2.**

### M2: No semantic stratification for glass, vegetation, and thin structures

**What goes wrong:** LiDAR often returns from *behind* glass, from interior surfaces, and from individual leaves; RGB MVS reconstructs the glass plane and a fuzzy vegetation volume. Thin poles are reconstructed poorly or not at all. These classes produce heavy-tailed errors that dominate RMSE and make the mesh baseline look worse than it is (or better, if they are silently dropped).

**Prevention:** Report metrics per semantic class (ground, building/facade, vegetation, pole/sign, vehicle, window/glass). Decide explicitly whether glass and thin structures are in the primary metric or a secondary table. Never drop a class without reporting the drop.

**Warning signs:** RMSE ≫ MAE; isolated 1–10 m errors on windows; recall near zero on poles while overall recall looks fine.

**Phase:** P5. **Gate G5.**

### M3: Near-epsilon, `t_min`, and self-intersection

**What goes wrong:** With `t_min = 0`, rays may hit the triangle containing the origin, or numerical noise produces a hit at `t ≈ 1e-6`. With `t_min` too large (e.g. 0.5 m), real geometry at 0.2–0.5 m is missed. Poisson shells near the camera make this worse.

**Prevention:** Use `t_min = 1e-4` m for unit directions, exclude hits below `r_min` from the evaluated domain, and log the `t_pred` histogram. Test a ray whose origin lies exactly on a triangle and one whose origin is inside a closed box.

**Warning signs:** `t_pred` spike at 0; hits on the camera's own near geometry; a cluster of near-zero ranges.

**Phase:** P4. **Gate G4.**

### M4: Non-watertight ray–triangle intersection and precision leaks

**What goes wrong:** Rays can slip through shared edges/cracks in a non-watertight mesh and hit a surface behind, producing a rare but huge range error. Using float32 world coordinates far from the origin amplifies this. Embree and a hand-written Möller–Trumbore disagree exactly at edges; neither is "wrong", but the difference must be characterised.

**Prevention:** Keep geometry in double precision until the final float32 cast, or translate the mesh to a local origin near the camera. Use a watertight ray–triangle test (Embree) for production; use Möller–Trumbore as the reference and compare on 10k rays. Report the count of "leak-like" hits (e.g. `t_pred > t_neighbors + 5 m` at depth discontinuities).

**Warning signs:** Error tail with `r_pred ≫ r_gt` at silhouettes; results change when the mesh is translated; backend disagreement > 1e-6 m away from edges.

**Phase:** P4. **Gate G4.**

### M5: OpenMVS default meshing parameters leave long triangles

**What goes wrong:** `ReconstructMesh` defaults (`--remove-spurious 20`) leave very long triangles spanning gaps in sparse regions; the mesh renders as a featureless blob and ray casting hits surfaces that were never observed. The official sample workspace lowers this to 4 and adds hole closing and smoothing.

**Prevention:** Start from `--remove-spurious 4 --close-holes 30 --smooth 2` (as in the OpenMVS sample), inspect the largest-triangle edge-length distribution, and crop components whose bounding box exceeds the observed corridor. Re-tune only on the dev frames.

**Warning signs:** Triangle edge lengths of tens of metres; a "blob" silhouette; hit rate ≈ 1 across gaps.

**Phase:** P3. **Gate G3.**

### M6: Compute and memory blow-up

**What goes wrong:** A dense TSDF over a 200 m corridor at 5 cm is ~2.5×10⁹ voxels — impossible. Poisson at octree depth 12 on a site-scale cloud needs tens of GB. Full-resolution MVS on 4 cameras × hundreds of frames consumes GPU-hours and hundreds of GB of depth maps. The deadline disappears before the first metric.

**Prevention:** Bound the sub-segment (100–300 m, 50–150 frames), crop to a corridor, use `ScalableTSDFVolume` (sparse blocks) or tile Poisson with overlap, start at 10 cm voxels for the pipeline shakedown and only then go to 2–2.5 cm, and cache depth maps/fused clouds by hash. Measure peak RSS/VRAM and record it.

**Warning signs:** OOM at the TSDF/Poisson stage; a single frame taking minutes; disk filling with depth maps; no time left for evaluation.

**Phase:** P2/P3. **Gate G3.**

### M7: KD-tree pooling and density bias in CD

**What goes wrong:** Concatenating per-frame GT-ray predictions into one cloud gives nearby surfaces enormous density (seen from hundreds of frames) and far surfaces tiny density. Symmetric CD then weights near geometry heavily; predicted-duplicate points do not change recall but do change the NN statistics. Per-frame CD averaged over frames is a different number from pooled CD.

**Prevention:** Choose one aggregation and state it. Prefer per-frame CD then mean ± std, with the pooled number as a secondary. If voxel-downsampling, fix the voxel size and report the sensitivity. Never compare a pooled CD against a paper's per-frame CD.

**Warning signs:** CD changes by > 20% when frames are subsampled; near-ground points dominate nearest-neighbour distances; duplicate predicted points at the same location.

**Phase:** P5. **Gate G5.**

### M8: Sparse pose frames and silent nearest-pose fallback

**What goes wrong:** `poses.txt` omits frames with little movement, so not every image/LiDAR frame has a pose. A `nearest` lookup returns the wrong pose by up to several centimetres or more, silently. Dropping frames without logging changes the evaluated set.

**Prevention:** Build an explicit frame→pose table, assert exact frame-index matches, and either drop missing frames (logged) or interpolate with a documented error bound (and verify it against the exact poses on frames where both exist). Never use `nearest` without a distance threshold.

**Warning signs:** Pose timestamps disagree with image timestamps; frame counts differ between modalities; the evaluated frame list changes between runs.

**Phase:** P1. **Gate G2.**

### M9: Undefined FPS measurement protocol

**What goes wrong:** FPS is reported without saying whether BVH build, ray generation, mesh loading, metric computation, or I/O are included; whether it is single- or multi-threaded; whether it is cold or warm; and whether a "frame" means one camera or all cameras. The resulting number is not comparable to any published GPU baseline, yet it is presented as one.

**Prevention:** Define: rays per "frame" (camera count × rays), thread count (pin it), included stages, warm-up iterations, number of repeats, and median/IQR. Report rays/s and per-scan latency separately from BVH build and from metric evaluation. Label CPU vs. GPU comparisons as cross-hardware, not head-to-head.

**Warning signs:** FPS varies 2× between runs; a "frame" is undefined; build time omitted; CPU FPS compared directly to a GPU paper.

**Phase:** P5. **Gate G5.**

### M10: Fisheye/MVS undistortion trap (if fisheye is added)

**What goes wrong:** Fisheye images cannot be fed to a pinhole MVS pipeline without remapping; naive undistortion to a pinhole crops most of the FOV, defeating the purpose, while cubemap remapping creates seams and duplicated geometry. COLMAP MVS is built around pinhole cameras.

**Prevention:** For v1.0, exclude fisheye. If added later, remap to cubemap faces with overlap, run MVS per face, fuse in 3D, and handle seam duplication; use a fisheye camera model for ray generation rather than pinhole. Note that fisheye images have no 2D semantic labels.

**Warning signs:** FOV shrinks after undistortion; duplicated walls at cubemap seams; projection error grows toward fisheye edges.

**Phase:** P2/P3 (only if in scope). **Gate G3.**

---

## Minor Pitfalls

| Pitfall | What goes wrong | Prevention | Phase |
|---|---|---|---|
| Hardcoded image size / intrinsics | Switching segment or camera silently breaks projection | Read `S_rect`, `P_rect`, and actual PNG shape; assert equality | P1 |
| Wrong 3×4 parsing | `np.loadtxt(...).reshape(4,4)` fails or transposes R | Parse 3×4 row-major `[R\|t]`, pad with `[0,0,0,1]`, assert orthonormality | P1 |
| Float32 transforms | Millimetre-level drift and edge leaks at 1 km world coordinates | Keep transforms in float64, cast only at the ray/triangle stage | P4 |
| Thread oversubscription | COLMAP + Open3D + NumPy each spawn all cores; timing meaningless | Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `torch.set_num_threads` explicitly | P5/P6 |
| Committing data/meshes to git | Repo bloat, license issues, broken clones | `.gitignore` + DVC/checksums; keep only configs and small fixtures | P6 |
| Using test-split frames | No semantic labels, no 3D bboxes, no dynamic masks | Restrict to train/val frames listed in the dataset splits | P1 |
| Unversioned dataset release | Pose/calibration conventions differ between releases | Record the release/version and hash calibration files | P6 |
| Missing normals for Poisson | `create_from_point_cloud_poisson` returns empty or ballooned mesh | Estimate and orient normals; check the density field is non-degenerate | P3 |

---

## Reconstruction-Route Failure Profiles

The route is still TBD; each has a distinct dominant failure mode. Use this table to choose *what to validate first*, not as a route recommendation.

| Failure mode | COLMAP MVS | OpenMVS | Learning-based depth (MASt3R / Metric3D / Depth Anything V2) |
|---|---|---|---|
| Textureless road | Weak PatchMatch, geometric-consistency filter removes it → holes | Densify weak on road; mesh refinement may fill | Smooth, plausible road from monocular prior — but scale/shift ambiguity |
| Sky | Spurious far points; mask needed | Spurious far points; mask needed | Often predicts a finite "sky depth"; mask needed |
| Thin structures | Lost | Slightly better (mesh refinement) | Often smoothed away |
| Moving objects | Ghost blobs / double walls | Ghost blobs | Per-frame plausible, cross-frame inconsistent → TSDF ghosting |
| Scale | Metric **only if** the model is built from KITTI poses | Metric if input model is metric | Usually up-to-scale → alignment needed → GT-leakage risk |
| Output type | Fused cloud → Poisson (balloon risk) | Visible-surface mesh (holes but no interior) | Point maps → TSDF/Poisson |
| Determinism | Mostly deterministic with fixed threads; PatchMatch hypotheses may vary | Deterministic with fixed threads | Deterministic given weights + seed, but model-version dependent |
| Dominant compute | GPU PatchMatch + fusion | CPU densify + mesh | GPU inference + fusion |
| First gate to run | G1 + G3 (bbox/units) | G3 (long triangles) | **Scale check** (G3) before anything else |

**Confidence:** MEDIUM. The COLMAP/OpenMVS entries are grounded in their docs/source; the learning-based entries are a synthesis of documented scale ambiguity and depth-noise behaviour, not a benchmark.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|---|---|---|---|
| Hardcoding the segment/frame list in scripts | Fast start | Silent breakage on the next segment; irreproducible runs | Never — use a versioned config |
| Skipping Velodyne motion compensation | One less build step | Azimuth-correlated bias up to ~1 m; invalidates quantitative claims | Only for a smoke test, clearly labelled |
| Using GT-ray points as the predicted cloud for CD | No extra code | CD ≈ range error; metrics not independent | Acceptable **if** stated explicitly and supplemented |
| Poisson without density trim | Looks complete/watertight | Balloon surfaces, camera inside mesh, hidden holes | Never for evaluation |
| 10 cm voxels while reporting F@5 cm | Fast meshing | F@5 measures the voxel grid | Never — demote F@5 instead |
| Pure-Python ray–triangle loops | No compiled dependency | 100–1000× slower; fake FPS | Only for ≤ 100 correctness rays |
| Nearest-pose lookup | Avoids dropping frames | cm-level silent pose error | Only with a distance threshold and logging |
| Voxel-downsampling before CD | Speed/memory | Metric changes with voxel size | Only with fixed size + sensitivity table |
| One-frame evaluation | Instant feedback | High variance; cherry-pick risk | Smoke test only |
| Reusing MVS depth as GT | Fast "ground truth" | Circular evaluation, GT leakage | Never |

---

## Integration Gotchas

Mistakes that appear when wiring the mesh pipeline to COLMAP/OpenMVS/Open3D and to the evaluator.

| Integration | Common Mistake | Correct Approach |
|---|---|---|
| COLMAP `images.txt` | Writing camera→world poses | COLMAP stores **world→camera** (`qvec, tvec`); verify by reprojecting |
| COLMAP camera model | Folding `P_rect`'s `Tx` baseline into `K`, or using `SIMPLE_PINHOLE` with 4 cameras | Use `K = P_rect[:, :3]`, one camera per physical camera, `--ImageReader.single_camera 0` |
| COLMAP dense workspace | Mixing rectified images with raw intrinsics | Use `data_rect` for image_00/01, or undistort `data_rgb` first |
| OpenMVS `InterfaceCOLMAP` | Feeding only images + poses without a valid sparse model/points3D | Provide the COLMAP dense workspace, or build the model explicitly from KITTI poses |
| OpenMVS `ReconstructMesh` | Using defaults on a sparse street cloud | `--remove-spurious 4 --close-holes 30 --smooth 2`, then inspect edge lengths |
| Units | Mesh in COLMAP/normalized units, rays in metres | Assert mesh bbox vs GT extent (0.8–1.2); never scale to fit metrics |
| Open3D `RaycastingScene.cast_rays` | Treating `t_hit` as metres with unnormalized directions | Normalize directions or compute `‖hit − origin‖`; test on a unit sphere |
| Open3D mesh frame | Mesh in camera coordinates, rays in world | One world-frame mesh; transform rays, not the mesh, per camera |
| pyrender/OpenGL | Assuming +z forward | Convert with an explicit axis shim; test on a non-symmetric object |
| KD-tree frame | CD computed in camera frame per camera | Aggregate in one frame (LiDAR/world) with one KD-tree, or per-frame then average |
| Embree vs. Möller–Trumbore | Expecting bit-identical `t` | Cross-check with tolerance; characterise edge disagreements |
| Dataset release | Mixing v1/v2 poses or calibration | Pin the release; hash `calibration/` and `data_poses/` |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|---|---|---|---|
| Dense TSDF over the corridor | OOM at the fusion stage | `ScalableTSDFVolume`, crop to 100–300 m, start at 10 cm | Corridor > ~50 m at 5 cm voxels |
| Poisson octree depth ≥ 12 | Tens of GB, minutes-to-hours | Depth 10–11 for a 200 m scene; tile with overlap | Scene extent > ~100 m at 2 cm |
| Python per-ray Möller–Trumbore | Minutes per frame | Embree/Open3D `RaycastingScene`, or C++/Numba with a real BVH | > 10k rays × > 100k triangles |
| Rebuilding the BVH per frame | Latency dominated by build | Build once per mesh; reuse for all frames | > 10 frames |
| `cKDTree` on 10⁷ × 10⁷ points | OOM / swap | Chunk queries, `workers=-1`, or per-frame CD | > ~5×10⁶ points per cloud |
| Loading all images at once for MVS | VRAM OOM | Use COLMAP's workspace/batch options; cap `max_image_size` | 4 cameras × > 100 frames at 1080p |
| float64 depth maps on disk | 2× I/O, slow fusion | Store float32 + confidence; keep float64 only for transforms | > 50 frames |
| Evaluating all frames for every config | Days of wall time | Freeze a 10–20-frame dev subset; full set only for final numbers | > 3 configs × > 100 frames |
| Re-reading PLY GT per frame | Minutes of I/O per run | Cache to `.npz`/Parquet once, keyed by hash | > 50 frames |

---

## Integrity & Compliance Mistakes

This replaces the template's generic security section: the dominant risk here is scientific integrity and dataset compliance, not adversarial input.

| Mistake | Risk | Prevention |
|---|---|---|
| Using LiDAR to set mesh scale, orientation, or hole filling | GT leakage; the "RGB-only reconstruction" claim becomes false | Build the model from KITTI poses; validate (not fit) against LiDAR after freezing |
| Selecting the segment/frames after seeing metrics | Cherry-picking; non-reproducible claims | Pre-register segment + frame list in P1; hash it |
| Reporting only the best τ or bucket | Inflated F-score | Report all τ and all buckets with counts |
| Silently dropping occluded/dynamic/thin-structure points | Metrics improve without disclosure | GT waterfall with counts at every exclusion step |
| Tuning Poisson/voxel parameters on the reported frames | Overfitting the evaluation | Dev/report frame split; tune on dev only |
| Redistributing KITTI-360 images or derived meshes without checking the licence | Licence violation (non-commercial terms; faces/plates) | Keep data out of git; check the dataset licence before publishing derived assets |
| Committing dataset credentials | Account compromise | Environment variables / `.netrc` outside the repo; never in configs |

---

## "Looks Done But Isn't" Checklist

- [ ] **Calibration:** chain implemented in one module; G1 overlay and camera-height checks pass; image_01 baseline handled correctly.
- [ ] **Sensor model:** 2 perspective + 2 fisheye documented; the evaluated FOV is explicit; no phantom cameras.
- [ ] **GT source:** per-frame `.bin`, motion-compensated, transformed by exact frame poses; accumulated PLYs are not used as GT.
- [ ] **Dynamic masks:** applied to images, GT, projections, and validation; residual dynamic fraction logged; ego ROI removed in 3D.
- [ ] **Ray set:** frozen with SHA-256; behind-camera, out-of-FOV, near-field, and occluded points excluded with counts; one camera per ray.
- [ ] **Range definition:** `r_gt = ‖p_C‖` from the camera origin, not `‖p_L‖`; units are metres.
- [ ] **Mesh frame/scale:** bbox matches GT extent; camera centres match the trajectory; no similarity transform was fitted to metrics.
- [ ] **Poisson/TSDF:** density trim applied; camera-outside test passes; hit rate is not ≈ 1.0 in all directions.
- [ ] **Resolution:** voxel/decimation error is below the smallest reported τ, or F@5 is demoted with an explanation.
- [ ] **Ray casting:** analytic tests pass; `t` is metric; misses are `inf` and never enter a mean; near-epsilon tested.
- [ ] **Metrics:** CD formula/units stated; F@τ thresholds in metres; all τ and all buckets reported with counts.
- [ ] **Coverage:** MAE/RMSE always reported with hit rate/recall; per-class and per-bucket counts present.
- [ ] **Incidence:** point-to-plane error or error-vs-incidence reported; global height offset reported, not tuned away.
- [ ] **Throughput:** FPS protocol defined (threads, stages, warm-up, repeats, rays/frame); CPU-vs-GPU labelled.
- [ ] **Reproducibility:** manifest complete; metrics regenerate from frozen ray set + mesh; timing has ≥ 3 repeats.
- [ ] **Claims:** no GT leakage; no post-hoc segment/threshold selection; every reported number traceable to a run directory.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---|---|---|
| Wrong calibration chain | MEDIUM | Re-derive with `kitti360Scripts`; re-run G1 overlay; regenerate ray set and all metrics (mesh may survive if only rays were wrong) |
| Motion distortion discovered late | MEDIUM | Apply uncurling offline to all frames, rebuild the ray set, re-run evaluation; the mesh is unaffected if it was built from images only |
| Accumulated clouds used as GT | MEDIUM | Rebuild GT from per-frame `.bin`; regenerate ray set; re-run metrics; discard all earlier numbers |
| Dynamic contamination in mesh | HIGH | Re-mask, re-fuse, re-mesh; if bboxes were available, remove fused points and re-mesh instead of full re-MVS |
| Mesh in wrong frame/scale | HIGH | Rebuild the COLMAP model from KITTI poses and re-run MVS; do not "fix" with a similarity transform fitted to metrics |
| Poisson balloon / camera inside mesh | LOW–MEDIUM | Raise density quantile, crop, re-mesh; no need to re-run MVS |
| Holes on textureless road | MEDIUM | Add slanted-plane/ground prior or fill with a declared plane; re-mesh; report the prior |
| Miss/NaN bug | LOW | Fix masking and denominators; re-run metrics only |
| Occlusion contamination | LOW | Add the GT-only visibility flag; regenerate the ray set; re-run metrics |
| Wrong CD/F definition | LOW | Fix formulas; re-run metrics; update the report text and any comparisons |
| Unreproducible numbers | MEDIUM–HIGH | Rebuild the manifest, pin versions/threads, freeze artifacts; if the mesh is lost, re-run reconstruction and accept a new baseline |

---

## Pitfall-to-Phase Mapping

| # | Pitfall | Prevention Phase | Verification |
|---|---|---|---|
| C1 | Calibration chain inversion | P1 | G1 overlay < 2 px; camera height 1.4–1.9 m; round-trip < 1e-9 |
| C2 | Rectified/unrectified mixing; image_01 baseline | P1 | Both cameras' overlays pass; `K = P_rect[:, :3]` + physical pose |
| C3 | Velodyne motion distortion | P1 | Signed range error vs. azimuth is flat; residual bias reported |
| C4 | Accumulated clouds as GT | P1 | Per-frame point count ≈ 0.17M; ray builder reads `.bin` + pose |
| C5 | Wrong sensor/FOV model | P1 | 2 perspective + 2 fisheye documented; ray FOV matches image bounds |
| C6 | Partial dynamic masking | P1, re-audit P3 | Residual dynamic fraction < 1%; no bbox–mesh intersection |
| C7 | Textureless road/sky MVS failure | P2/P3 | Road depth coverage and height noise reported; no sky caps |
| C8 | Moving-object/exposure ghosting | P2/P3 | Largest component > 95%; no floating components in BEV |
| C9 | Mesh frame/scale mismatch | P2/P3 | Bbox ratio 0.8–1.2; camera centres match trajectory; no fitted transform |
| C10 | Poisson balloon / camera inside mesh | P3 | Camera-outside test; hit rate 0.3–0.9 forward, ~0 behind |
| C11 | Resolution vs. metric thresholds | P3/P5 | Voxel/decimation error < τ_min, or F@5 demoted with reason |
| C12 | Frame/ray convention errors | P1/P4 | G4 analytic tests; hit rate plausible; non-symmetric scene check |
| C13 | Miss/FOV/range-denominator errors | P1/P4 | No NaN/Inf; waterfall counts sum; `r_gt` from camera origin |
| C14 | Camera-origin occlusion | P4 | Occluded flag present; occluded fraction reported; RMSE tail explained |
| C15 | Multi-camera overlap double counting | P1/P4 | Exactly one camera per ray; per-camera + dedup aggregate reported |
| C16 | Accuracy-on-hits / hole hiding | P5 | Every accuracy row has a coverage number; CDF with misses |
| C17 | CD/F degeneracy and definitions | P5 | Formulas + units in report; degeneracy acknowledged; all τ reported |
| C18 | Grazing-incidence / ground domination | P5 | Error vs. incidence and per-class; point-to-plane reported |
| C19 | Cherry-picking | P1/P6 | Pre-registered config + hash; all buckets/thresholds present |
| C20 | Reproducibility / artifact drift | P6 | Manifest complete; metrics regenerate from frozen artifacts |
| C21 | 2DGS-derived mesh used as the "mesh-only" baseline | P2/P6 | Mesh produced without any Gaussian model; provenance recorded |
| M1 | Ego-ROI collateral damage | P1 | 3D ROI used; near-field GT count reported |
| M2 | No semantic stratification | P5 | Per-class table with counts |
| M3 | Near-epsilon / `t_min` | P4 | G4 edge cases pass; `t_pred` histogram has no zero spike |
| M4 | Non-watertight intersections | P4 | Leak count reported; double-precision transforms |
| M5 | OpenMVS long triangles | P3 | Edge-length distribution bounded; no > 10 m triangles across gaps |
| M6 | Compute/memory blow-up | P2/P3 | Peak RSS/VRAM recorded; runtime inside the phase budget |
| M7 | KD-tree pooling/density bias | P5 | Aggregation stated; subsample sensitivity < 20% |
| M8 | Sparse pose frames / nearest lookup | P1 | Explicit frame table; missing frames dropped and logged |
| M9 | Undefined FPS protocol | P5 | Protocol documented; median/IQR over ≥ 3 repeats |
| M10 | Fisheye undistortion trap | P2/P3 | Only relevant if fisheye enters scope; FOV preserved; seams checked |

---

## Sources

**KITTI-360 official documentation** — folder structure, calibration file semantics, pose semantics, coordinate systems, sensor list (2 perspective + 2 fisheye), accumulated 3D semantics, 2D label coverage.
https://cvlibs.net/datasets/kitti-360/documentation.php — **HIGH**

**kitti360Scripts devkit** — `loadCalibrationCameraToPose`, `loadCalibrationRigid`, camera-to-camera composition, `curlVelodyneData` motion compensation, 3D bbox loaders.
https://github.com/autonomousvision/kitti360Scripts — **HIGH**

**KITTI raw readme** — Velodyne continuous sweep and "untwisting"; sensor coordinate conventions.
https://www.cvlibs.net/datasets/kitti/raw_data.php — **HIGH**

**Liao et al., KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding (TPAMI 2022)** — dataset design, poses, semantics.
https://arxiv.org/abs/2109.08224 — **HIGH**

**COLMAP documentation** — `images.txt` stores world→camera (`qvec, tvec`); camera models; dense stereo and geometric consistency.
https://colmap.github.io/ — **HIGH**

**OpenMVS wiki and source** — `InterfaceCOLMAP`, `ReconstructMesh` defaults (`--remove-spurious 20`), sample workspace using `--remove-spurious 4 --close-holes 30 --smooth 2`.
https://github.com/cdcseacave/openMVS/wiki ; https://github.com/cdcseacave/openMVS_sample — **HIGH**

**Open3D documentation** — Poisson surface reconstruction requires normals and returns a density field; `RaycastingScene` (Embree) `cast_rays` semantics.
https://www.open3d.org/docs/ — **HIGH**

**Poisson density-trimming practice** — low-density "skin" across gaps, quantile trimming, ballooning from inconsistent normals, memory at depth ≥ 12.
https://3d-geospatial.com/point-cloud-mesh-processing-pipelines/surface-reconstruction-algorithms/ — **MEDIUM** (practitioner source, consistent with Open3D docs and common pipelines)

**MVS on urban scenes** — textureless roads, moving-object contamination, Poisson holes around dynamic objects.
"Revisiting PatchMatch Multi-View Stereo for Urban 3D Reconstruction" (arXiv:2207.08439); RoMe (arXiv:2306.11368) — **MEDIUM/HIGH**

**Chamfer distance / F-score definition variability** — inconsistent mean/sum, squared/unsquared, one-way/symmetric, and limited sensitivity of thresholded F-score.
Empir3D (arXiv:2306.03660); "Rethinking Metrics ... for 3D Point Cloud Generation" (arXiv:2511.05308) — **MEDIUM**

**Derived analysis (this file)** — GT-ray CD/F degeneracy, camera-origin occlusion, grazing-incidence amplification, and the GT waterfall are first-principles analyses of the project's own protocol, not claims from a single paper. Treat them as **MEDIUM** confidence and validate empirically in P4/P5: the occlusion fraction and the CD-vs-MAE correlation are directly measurable on the frozen ray set.

---

*Pitfalls research for: mesh-only KITTI-360 reconstruction + GT-defined-ray LiDAR evaluation*
*Researched: 2026-10-30*



