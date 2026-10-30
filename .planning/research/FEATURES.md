# Feature Research

**Domain:** Mesh-only LiDAR simulation baseline + GT-defined-ray evaluation on KITTI-360 (offline, CPU, single segment)
**Researched:** 2026-10-30
**Confidence:** MEDIUM (protocol/metric definitions HIGH; expected absolute error bands MEDIUM-LOW — see §Evidence quality)
**Scope:** Milestone v1.0 only — one low-dynamic KITTI-360 segment, RGB→mesh reconstruction, CPU BVH GT-ray casting, full metric suite. No 3DGS/2DGS curves, no GPU, no hybrid.

---

## 0. How this domain actually works (orientation for requirements)

A mesh-only LiDAR simulation baseline has four separable stages, and each one can silently invalidate the others:

1. **Dataset/pose preparation.** KITTI-360 gives metric poses (`data_poses/*/cam0_to_world.txt`, `poses.txt`), calibration (`calib_cam_to_velo.txt`, `perspective.txt`), 64-beam HDL-64E scans in Velodyne coordinates (x forward, y left, z up; ~120 m range, 26.8° vertical FOV, ~10⁵ points/scan), 2 rectified perspective cameras (1408×376, ≈90° HFOV) + 2 fisheye cameras (360° imagery total), and 2D/3D semantic labels. **No SfM is needed.**
2. **Reconstruction.** Because poses are known, this is a *known-pose MVS* problem, not SfM. The mesh route (COLMAP MVS / OpenMVS / learning-based) is still undecided, so every downstream interface must be mesh-route-agnostic.
3. **GT-ray casting.** For each static LiDAR point `p_L`, transform to the camera frame, normalize to a direction, cast from the camera optical center, and compare `r_pred` to `r_gt = ||p_C||`. Miss (`r_pred = ∞`) is excluded from range error but counted in recall/miss accounting.
4. **Evaluation.** Range metrics are per-ray and exact. Point-cloud metrics (CD/F-score) are computed on the resulting point set — and, critically, **are not independent evidence from range error** under this protocol (§1.5, D12).

The published LiDAR-NVS literature (LiDAR-GS, GS-LiDAR, LiDAR-RT, LiDAR4D, LiDAR-NeRF) optimizes a representation *against LiDAR supervision* and then reports CD/F/depth-RMSE on held-out scans. This milestone deliberately does **not** do that: LiDAR is eval-only, and the representation is a mesh from RGB. So literature numbers are context, not targets (§5, §7).

**Existing work to build on: none.** The repository currently contains documentation only (`README.md`, `docs/`, `.planning/`). Every feature below is greenfield; the only "dependencies on existing work" are *external* assets and decisions: the KITTI-360 download, the still-open mesh-route decision (T8), and already-sanctioned libraries (`open3d` for mesh I/O/distance queries and as an Embree reference, `kitti360Scripts` for calibration/pose semantics, `scipy`/Open3D for NN search, numpy/torch already in the environment). Nothing in this milestone requires a dependency the project has not already accepted.

---

## 1. Metric contract (normative — requirements must copy this verbatim)

This section exists because the literature is genuinely ambiguous: the widely-copied `fscore(dist1, dist2, threshold=0.05)` helper in the LiDAR-NVS codebases (LiDAR4D `utils/metrics.py`, GS-LiDAR `chamfer/fscore.py`) receives **squared** distances, so its "F-score@5 cm" threshold is really **≈22.4 cm** in Euclidean terms; and depth tables mix units (RMSE in cm, MedAE in m). Our milestone must be self-consistent and state units.

### 1.1 Ray set and validity

A ray is **valid** iff all hold:

| Condition | Default | Rationale |
|---|---|---|
| GT point is static (not dynamic-class, not ego-vehicle, not out-of-ROI) | required | Static-only scope |
| `r_gt = \|\|p_C\|\| ∈ [r_min, r_max]` | `[1.5 m, 50.0 m]` | Drop ego/self-return region; 50 m is the primary cutoff |
| GT point projects inside the chosen camera image with margin | ≥ 5 px from each border | A ray outside the reconstructed frustum is a guaranteed miss, not a finding |
| GT point in front of camera (`z_C > 0`) | required | Camera-frame convention |
| Frame has a valid pose and calibration | required | `poses.txt` is subsampled — see §3 dependency T3 |

Ray definition (camera `c`, frame `f`, LiDAR point `p_L`):

```
p_C   = T_CL(f,c) · p_L              # T_CL = inv(cam_c_to_world(f)) · inv(calib_cam_to_velo)
d     = p_C / ||p_C||                # unit direction, camera frame
r_gt  = ||p_C||                      # metres
o     = (0,0,0)                      # camera optical centre
```

The **ray set is frozen and hashed** (SHA-256 over `[frame_id, cam_id, p_L, d, r_gt]`) so every reported number is reproducible and every method is scored on identical rays.

### 1.2 Hit / miss

```
r_pred(i) = min{ t > t_min : ray (o=0, d_i) intersects mesh }   else ∞
t_min = 0.10 m          # numerical self-hit guard
t_max = 60 m            # cast bound; larger than r_max so "hit but far" is an error, not a miss
```

- **Miss**: no intersection with `t ∈ (t_min, t_max]` → `r_pred = ∞`.
- **Hit**: any intersection, including back-faces. **Do not back-face cull** — MVS meshes are open, non-manifold and inconsistently oriented.
- Deterministic tie-break: smallest `t`; equal `t` → smallest triangle id.

### 1.3 Range metrics (hits only, metres)

Let `H = {i : r_pred(i) < ∞}` and `e_i = r_pred(i) − r_gt(i)` (signed; **positive = predicted farther**).

| Metric | Definition | Units | Notes |
|---|---|---|---|
| MAE | `(1/\|H\|) Σ_{i∈H} \|e_i\|` | m | Headline; heavy-tailed |
| RMSE | `sqrt((1/\|H\|) Σ e_i²)` | m | Sensitive to outliers; always report with MedAE |
| MedAE | `median_{i∈H}(\|e_i\|)` | m | Robust; the number to trust |
| Bias | `(1/\|H\|) Σ e_i` | m | **Required diagnostic** (D2, §2.2) |
| P90 / P95 | 90th/95th percentile of `\|e_i\|` | m | Tail behaviour; optional but cheap |

If `|H| = 0`, the value is **`null`** (JSON) / empty (CSV), never `0` and never `NaN`.

### 1.4 Miss / coverage metrics

| Metric | Definition | Units |
|---|---|---|
| Miss rate | `\|M\| / \|R\|`, `M = R \ H` | % |
| Ray recall (hit rate) | `1 − miss rate` | % |
| Miss rate by bucket | miss rate restricted to rays in that bucket | % |
| Far-hit count | `\|{i ∈ H : r_pred(i) − r_gt(i) > 5 m}\|` | count |

**Ray recall ≠ point recall.** Ray recall is "did any surface get hit"; point recall@τ is "was the *right* surface hit within τ". Both are required and must be labelled distinctly in the report.

### 1.5 Point-cloud metrics (LiDAR/world frame, `P = {p_pred,i}`, `Q = {p_gt,i}`)

With `d(p,Q) = min_{q∈Q} ||p − q||₂` (Euclidean, metres):

| Metric | Definition | Units | Convention note |
|---|---|---|---|
| CD‑L1 | `½[ (1/\|P\|)Σ_{p∈P} d(p,Q) + (1/\|Q\|)Σ_{q∈Q} d(q,P) ]` | m | "L1" = mean *unsquared* Euclidean distance. 2DGS/OMeGa-style |
| CD‑L2 | `½[ (1/\|P\|)Σ_{p∈P} d(p,Q)² + (1/\|Q\|)Σ_{q∈Q} d(q,P)² ]` | m² | "L2" = mean **squared** distance. Do **not** silently sum instead of averaging halves |
| CD‑acc (pred→GT) | `(1/\|P\|)Σ_{p∈P} d(p,Q)` | m | Accuracy / spurious-geometry side |
| CD‑comp (GT→pred) | `(1/\|Q\|)Σ_{q∈Q} d(q,P)` | m | Completeness / missing-geometry side |
| Precision(τ) | `\|{p∈P : d(p,Q) ≤ τ}\| / \|P\|` | % | τ in **metres, unsquared** |
| Recall(τ) | `\|{q∈Q : d(q,P) ≤ τ}\| / \|Q\|` | % | |
| F(τ) | `2·P(τ)·R(τ) / (P(τ)+R(τ))`, `0` if `P+R = 0` | — | |

Report `τ ∈ {0.05, 0.20, 0.50} m`. **Primary τ = 0.20 m**; `0.05 m` is reported as a strict indicator only (RGB reconstruction will not reach 5 cm — README §1.4 already warns this).

> **Non-independence caveat (must appear in the report).** Under GT-defined rays, every predicted point lies on a GT ray and there is a near 1:1 correspondence with GT points. Consequently `CD‑acc`, `Precision` and `CD‑L1` are largely a re-encoding of the range error, and `CD‑comp`/`Recall` are largely a re-encoding of misses plus large range errors. They are still worth reporting (they aggregate into familiar numbers and expose occlusion/parallax effects), but they are **not** independent evidence. State this, and use D12 (correlation audit) and D11 (point-to-mesh, which is not ray-parametrized) if independent geometric evidence is needed.

### 1.6 Distance buckets

Bucketed by **`r_gt`**, not `r_pred`:

| Bucket | Range (m) | Status |
|---|---|---|
| B1 | `[0, 20)` | primary |
| B2 | `[20, 40)` | primary |
| B3 | `[40, 50]` | primary (upper bound = primary cutoff) |
| B4 (optional) | `(50, 60]` | secondary only, explicitly labelled "out of primary range" |

Resolve the current inconsistency: README/PROJECT.md say "0–20 / 20–40 / 40–60" while also saying "primary 0–50 m". The testable resolution is **B3 = [40,50]**, with a separate 50–60 m row if desired. Buckets must be half-open except the last so no ray is counted twice.

### 1.7 Efficiency metrics (fixed hardware, warm runs)

| Metric | Definition |
|---|---|
| `rays/s` | `\|R\| / t_cast`, `t_cast` = cast + resolve only, median of 5 warm runs |
| Cast FPS | `n_frames / Σ t_cast` |
| End-to-end FPS | `n_frames / (ray gen + cast + resolve + metric accumulation)`, I/O excluded |
| BVH build time | seconds, single measurement + thread count |
| Peak RSS | MB (process peak), measured per run |
| Mesh size | `#vertices`, `#triangles` (after cleanup) |

Always record: CPU model, core/thread count, RAM, OS, compiler/ISA flags, Python/Open3D/Embree versions. FPS without these is meaningless.

### 1.8 Aggregation and statistics

- **Headline = micro-average** over pooled rays (each ray weighted equally).
- Also report **macro-average over frames ± std**, because failures are frame-correlated (vegetation clusters, junctions).
- Optional: bootstrap 95% CI clustered by frame (differentiator; prevents "one bad frame" narratives).
- Per-camera results first, then merge (`cam00`, `cam01`, …), because coverage differs wildly by camera.
- **Deduplicate before merging:** a GT point visible in two cameras must contribute one ray, not two (otherwise overlap regions are double-weighted and ranges are counted twice). Assign each point to exactly one camera by a deterministic rule (e.g., largest margin to the image border, then lowest camera id) and record the assignment count per camera.

---

## 2. Feature Landscape

### 2.1 Table Stakes (missing = baseline is not credible)

| # | Feature | Why Expected | Complexity | Notes / dependencies |
|---|---|---|---|---|
| T1 | **Segment manifest + low-dynamic selection** | "Single low-dynamic segment" is a locked decision; the choice must be justified, not vibes | LOW | Score frames by dynamic-point fraction (3D semantics / 2D masks), prefer residential, avoid junctions. Emit a manifest (seq id, frame window, counts) |
| T2 | **Calibration + pose composition module** | Every number depends on `T_CL`; getting the convention wrong silently rotates/translates the whole eval | MEDIUM | Load `calib_cam_to_velo`, `calib_cam_to_pose`, `perspective.txt`, `cam0_to_world.txt`; `T_CL = inv(T_cam0_world) · inv(T_cam0_velo)`; document axes (velo x-fwd/y-left/z-up; cam x-right/y-down/z-fwd; IMU x-fwd/y-right/z-down). Unit tests on known points |
| T3 | **Frame alignment & pose availability handling** | Images, Velodyne, and poses are separate streams; `poses.txt` is **subsampled** (frames skipped at low speed) | MEDIUM | Build the intersection of frames that have image + scan + pose; never interpolate silently. Record dropped frames |
| T4 | **Dynamic/ego filtering of GT** | Static-only scope; ego returns and moving cars would poison both sides of the metric | MEDIUM | Dynamic: 3D bbox of dynamic instances and/or 2D semantic mask projection; ego: drop `r < 1.5–2 m` and self-occlusion wedge. Count and report exclusions |
| T5 | **Static GT extraction + caching (`lidar_static.npz`)** | The frozen ray set is the contract for reproducibility | LOW | Store `p_L, frame, cam, r_gt, class, valid`; hash it |
| T6 | **GT-defined ray generator** | The locked protocol; also the only way to avoid LiDAR–camera parallax bias | LOW | `p_C = T_CL p_L`, `d = p_C/‖p_C‖`; float64 then cast float32; assert `‖d‖=1` |
| T7 | **FOV / visibility filter per camera** | A ray outside the reconstruction frustum is a guaranteed miss; including it makes recall uninterpretable | LOW-MEDIUM | Project to image, require margin; optionally require the point be inside the MVS coverage mask if available |
| T8 | **Mesh reconstruction pipeline (route-agnostic)** | Core deliverable | HIGH | Known-pose MVS: write `cameras.txt`/`images.txt` (COLMAP world→cam, qw qx qy qz), use **rectified** images + `P_rect_*`; fuse depth → mesh. Must be scripted + config-driven, not a notebook |
| T9 | **Mesh cleanup + statistics** | Raw MVS meshes are huge, noisy, with floaters and holes; un-decimated meshes make CPU casting impractical | MEDIUM | Statistical outlier removal, component filtering, decimation to a stated triangle budget, optional hole filling **flagged**; emit `#tri`, `#vert`, bbox, watertightness, hole count |
| T10 | **Scale & frame sanity gate** | #1 silent failure: mesh in wrong frame/scale → plausible-looking but meaningless errors | LOW | Assert mesh bbox overlaps GT bbox; assert median GT→mesh distance on planar classes is small; **never** run Umeyama/ICP alignment (it would hide pose bugs) |
| T11 | **CPU BVH with Möller–Trumbore** | Locked backend; the correctness reference for later GPU work | MEDIUM-HIGH | Two-sided, `t_min`/`t_max`, epsilon `1e-6`, deterministic tie-break, returns `t`, `tri_id`, barycentrics. **Native code path required** (C++/pybind11, Cython, or numba) — pure Python will not cast 10⁵–10⁶ rays |
| T12 | **Reference cross-check against Embree/Open3D** | A hand-written BVH is exactly where off-by-one and winding bugs hide | LOW | Open3D `RaycastingScene.cast_rays` already returns `t_hit = inf` and `INVALID_ID` on miss, matching our contract. Assert ≥99.9% hit/miss agreement and `\|Δt\| ≤ 1e-4 m` |
| T13 | **Predicted point reconstruction** | Needed for all cloud metrics | LOW | `p_pred = T_CL⁻¹ (r_pred · d)`; assert reprojection residual ~0 |
| T14 | **Nearest-neighbour engine** | CD/F/P/R need exact NN; brute force is O(N²) | LOW | `scipy.spatial.cKDTree` or Open3D `KDTreeFlann`; tie-breaking irrelevant at these scales; measure query time |
| T15 | **Full metric suite with buckets** | Locked deliverable (README §1.4) | MEDIUM | Implement §1 exactly; per-frame and pooled; semantic strata optional (D1) |
| T16 | **Machine-readable artifacts (JSON + CSV)** | "Metrics JSON/CSV" is an explicit deliverable; schema drift kills comparability | LOW-MEDIUM | JSON with `schema_version`, config hash, ray-set hash, per-camera + per-bucket blocks, `null` for undefined. CSV: one row per frame × camera × bucket |
| T17 | **Reproducibility harness** | "Re-runnable, not one-off" is a stated constraint | MEDIUM | Single entrypoint, config file, seeds, cached stages keyed by input hashes, environment capture, `run_manifest.json` with git SHA |
| T18 | **Analytic self-tests** | Proves the caster/metrics before any real data is involved | LOW | Plane/sphere/cube: assert `\|r_pred − r_analytic\| < 1e-4 m`; known misses return `∞`; degenerate rays handled |
| T19 | **Runtime/memory instrumentation** | FPS/memory are required metrics, and CPU cost shapes the later GPU story | LOW | Timer around cast/resolve, `psutil`/`getrusage` peak RSS, BVH build timer |
| T20 | **Protocol + metric documentation in the report** | Numbers are meaningless without conventions (units, τ semantics, ray set) | LOW | State: GT rays, camera origin, 0–50 m, buckets, miss = ∞, unsquared τ, CD averaging convention |

### 2.2 Differentiators (valuable, not required to be "credible")

| # | Feature | Value Proposition | Complexity | Notes / dependencies |
|---|---|---|---|---|
| D1 | **Semantic stratification (thin vs planar)** | Directly produces the evidence the later hybrid thesis needs: mesh recall collapses on poles/fences/vegetation, holds on road/facade. Turns a scalar into an argument | MEDIUM | KITTI-360 classes: thin = `pole`, `smallpole`, `traffic sign`, `traffic light`, `lamp`, `fence`, `guard rail`, `vegetation`; planar = `road`, `sidewalk`, `building`, `wall`, `terrain`. Labels exist only on accumulated clouds / 2D masks → needs projection or NN association (D1 depends on T4) |
| D2 | **Signed bias + tail quantiles + residual map** | Distinguishes "mesh is systematically too close" (floaters, smoothing) from "too far" (missed thin structures). A single MAE cannot | LOW | Requires only `e_i`; add a front-view error heatmap and a spherical miss map (one of the required plots) |
| D3 | **Asymmetric CD decomposition (acc vs comp)** | Shows *which* failure dominates: spurious surface vs missing surface. Prevents over-reading a single CD number | LOW | Already in §1.5; plot both vs range bucket |
| D4 | **Harness sensitivity / negative-control test** | Proves the metric is not inert: deliberately perturb calibration (e.g., swap cam00/cam01 extrinsics, add 5 cm translation) and show monotone degradation | LOW-MEDIUM | Kills the "our pipeline must be right because numbers look OK" failure mode |
| D5 | **Protocol ablation: camera-origin GT rays vs LiDAR-origin beam-pattern rays** | Quantifies the parallax bias the protocol was designed to avoid; cheap once the caster exists; strengthens the methods section | MEDIUM | Secondary protocol only; report coverage differences, never mix into the headline table |
| D6 | **Route-agnostic mesh ingestion + A/B route evidence** | The mesh route is *undecided*; the harness should settle it with numbers (COLMAP vs OpenMVS vs feed-forward MVS) rather than opinion | LOW if T8 is clean | Same config key `mesh.path`; run T15 on 2 meshes; compare runtime + accuracy + thin-structure recall |
| D7 | **Mesh diagnostics correlated with recall** | Explains *why* recall is low: hole area, triangle density vs distance, watertightness, near-camera floaters | MEDIUM | Emit per-frame coverage map from casting; correlate miss rate with local triangle density |
| D8 | **Compute-normalized reporting (error vs rays/s, error vs #triangles)** | Establishes the baseline point of the later Pareto curve without doing GPU work now | LOW | Just extra columns/plots; avoids the "we'll add speed later" gap |
| D9 | **Bootstrap CIs clustered by frame** | Honest uncertainty; prevents single-frame anecdotes | LOW-MEDIUM | Needs ≥30 frames; cheap in numpy |
| D10 | **Deterministic stratified ray subsampling** | Keeps CPU runtime tractable while remaining unbiased (stratify by range × azimuth) | LOW-MEDIUM | Must prove full-set vs subsampled agreement on ≥1 frame; report both |
| D11 | **Point-to-mesh distance on GT→mesh (no ray parametrization)** | An *independent* geometric check: ray casting measures first-visible surface, point-to-mesh measures nearest surface. Catches occluded-surface errors the ray protocol cannot see | MEDIUM | Open3D `compute_distance` / closest-point queries; report separately, not mixed into CD |
| D12 | **Metric collinearity audit (range error vs NN distance)** | Documents that under GT-ray protocol CD/F are largely a re-encoding of range error, so the report doesn't double-count evidence | LOW | Scatter/ρ between `\|e_i\|` and `d(p_pred, Q)`; a short paragraph in the report |

### 2.3 Anti-Features (explicitly excluded — with reasons)

| # | Anti-Feature | Why Requested | Why Problematic | Alternative |
|---|---|---|---|---|
| A1 | 3DGS / 2DGS Gaussian baselines and curves | "Everyone compares to Gaussian" | Locked out of v1.0; doubles compute, training, and evaluation surface; muddies the mesh-only correctness question | Defer; the harness is representation-agnostic by design (T8/T13), so the later milestone plugs in |
| A2 | GPU / CUDA / OptiX casting | Obvious speedup | The P1 deliverable is a *correctness reference*; optimizing now hides bugs behind parallelism and burns the milestone budget | T11 CPU + T12 Embree cross-check; GPU is P2 |
| A3 | GS-derived mesh (2DGS→TSDF→MC) as the mesh source | README §0.5.2 lists it | Explicitly deferred in PROJECT.md; requires a 2DGS training stack that is also out of scope | Feed-forward MVS / COLMAP / OpenMVS only; keep TSDF→MC as a *post-processing* step if the route produces depth maps |
| A4 | LiDAR-supervised mesh fitting / depth supervision | Would improve numbers a lot | **GT leakage**: LiDAR is eval-only by design; the whole claim "from RGB reconstruction" collapses | Use GT only for metrics; if depth priors are wanted later, use monocular priors, never LiDAR |
| A5 | Mesh built from the accumulated GT static point cloud | "Instant perfect mesh" | Circular: evaluates a GT-derived surface with GT points; also uses LiDAR in reconstruction | Allowed at most as a clearly-labelled *oracle upper bound* diagnostic, never as a baseline row |
| A6 | Intensity / ray-drop / multi-return / beam divergence modelling | Published LiDAR sims do it | Out of scope (geometry-only); requires radiometric calibration and supervision we don't have; adds metrics nobody can compare | State "geometry-only" in the report; revisit in a later milestone |
| A7 | Dynamic objects / scene graph / non-rigid | Real driving scenes move | Scope is static-only; dynamics need tracking, decomposition, temporal eval | Mask dynamics out (T4); single static segment |
| A8 | Multi-sequence / multi-dataset / generalization | "One segment is weak evidence" | Multiplies preprocessing + reconstruction + tuning cost; contrary to the locked single-segment decision | One segment, reported with explicit limitations; scale later |
| A9 | PSNR/SSIM/LPIPS on depth or intensity | "Papers report them" | Photometric/2D metrics conflate geometry with appearance and invite unfair comparison to camera NVS | Keep geometry metrics in metres; if a depth-image view is needed, report range error maps instead |
| A10 | Tuning the mesh (decimation, hole filling, thresholds) per test frame or per bucket | Makes numbers look better | Destroys the "reconstruction" claim and is undetectable in the report | Freeze mesh parameters **before** evaluation; one mesh per segment; log the config hash |
| A11 | Umeyama / ICP alignment of mesh to GT before eval | "Removes scale/pose error" | Hides exactly the bug we are testing for, and is a form of leakage | Use provided metric poses; fail loudly on T10 instead |
| A12 | Full-360° fisheye evaluation in v1.0 | KITTI-360 has 360° imagery | Fisheye undistortion + sparse side/back coverage makes recall uninformative and adds a large preprocessing surface | Front perspective camera(s) first; add `image_01` when T7 is stable; fisheye later |
| A13 | BVH micro-optimizations (SIMD packets, occupancy grids, persistent threads, LOD) | "Free speed" | Premature: changes results subtly (traversal order, epsilon) and consumes the correctness budget; speed is P2's job | Simple, well-tested BVH; only thread-parallel `for` over rays is allowed |
| A14 | Reporting only pooled means without buckets / miss accounting | Simpler table | Exactly the cherry-picking the milestone is designed to avoid; far-range failures dominate | Full §1 suite: buckets, miss rate, per-camera, per-frame spread |
| A15 | Claiming "real-time" from the CPU baseline | FPS is a required metric | CPU reference is expected to be slow; a real-time claim invites unfair comparison and distracts from correctness | Report FPS honestly as a baseline datapoint; frame it as the P1 reference |
| A16 | Training/optimizing anything end-to-end in this milestone | "Better mesh" | Learned optimization needs the Gaussian/NeRF stack that is deferred, and risks GT leakage (A4) | Use an off-the-shelf reconstruction as-is; the milestone is about evaluation validity |

---

## 3. Feature Dependencies

```
T1 manifest ──requires──> T3 frame alignment ──requires──> T2 calibration/pose
                                                                    ▲
T6 ray generator ──requires─────────────────────────────────────────┘
      ├──requires──> T7 FOV filter
      └──requires──> T5 static GT cache ──requires──> T4 dynamic/ego filter

T8 mesh pipeline (route choice)
  └──produces──> T9 cleanup/stats
                    └──requires──> T10 scale gate
                                      └──requires──> T11 BVH cast
                                                       ├──requires──> T12 Embree cross-check
                                                       ├──produces──> T13 p_pred
                                                       └──requires──> T19 instrumentation

(T6 rays + T13 p_pred)
  └──requires──> T14 NN engine
                    └──requires──> T15 metrics suite
                                     ├──produces──> T16 artifacts
                                     ├──requires──> T17 reproducibility
                                     └──produces──> plots (README §7: bucket RMSE+recall; Pareto scaffold; timing breakdown)

T18 analytic tests ──validates──> T11, T15
D4 negative control ──validates──> T2, T11, T15
D1 semantics ──enhances──> T15 (needs T4 + label association)
D5 protocol ablation ──enhances──> T6 (secondary protocol only)
D6 route A/B ──enhances──> T8 (needs T15 stable)
D11 point-to-mesh ──independent-check──> T15
D12 collinearity audit ──qualifies──> T15 (documentation, not code)
```

### Dependency notes

- **T2 → everything.** A convention error (row/column-major, inverse direction, quaternion order in COLMAP `images.txt`) produces plausible-but-wrong numbers. Gate it with T18/D4, not with "looks fine".
- **T3 is a real trap.** `poses.txt` skips frames at low speed (documented by KITTI-360). Do not assume `frame_index == line_number`. Use `cam0_to_world.txt` keyed by frame id, and only frames present in all three streams.
- **Multi-camera extrinsics are not symmetric.** Only `cam0_to_world.txt` is provided (rectified `image_00`). For `image_01` you must compose `calib_cam_to_pose['image_01']` (unrectified cam1→IMU) with `poses.txt` (IMU→world) and apply `R_rect_01` from `perspective.txt` to work in rectified coordinates. Getting this wrong produces a sub-metre but systematic error — exactly the kind D4 is designed to catch.
- **T4 before T15.** Dynamic points inflate GT→pred distances (nothing reconstructs a moving car at that instant) and depress recall for reasons unrelated to the mesh. Report exclusion counts.
- **T10 must fail loudly.** If mesh and GT bboxes do not overlap, stop; do not "fix" by aligning.
- **T12 requires no new dependency** — Open3D is already sanctioned in PROJECT.md and uses Embree internally, and its miss convention (`t_hit = inf`) matches ours.
- **D1 has a data caveat:** KITTI-360 semantic labels are on *accumulated* clouds and 2D masks, not on raw per-frame scans. Either project raw points into the 2D semantic image of the same frame (only covers image-visible points) or NN-associate to the accumulated static PLY (approximate). Document the association error.
- **D5 conflicts with the headline table** — never mix protocols in one column.
- **A10 conflicts with T17** — freeze mesh config before the first frozen-metric run; record the hash.

---

## 4. MVP Definition

### Launch with (v1.0 — this milestone)

- [ ] T1–T7 — dataset/pose/ray-set foundation, one segment, one camera (`image_00`), frozen + hashed ray set
- [ ] T8–T10 — one mesh route, cleaned + decimated, scale gate passing
- [ ] T11–T13 — CPU BVH (native path), Embree cross-check, `p_pred`
- [ ] T14–T16 — full §1 metric suite, buckets, JSON + CSV
- [ ] T17–T19 — one-command reproduction, analytic tests, timing/memory
- [ ] T20 — protocol + metric definitions in the report
- [ ] Plots: (1) MedAE/RMSE + recall vs distance bucket; (2) miss/error map; (3) CD/F vs τ (or error-vs-runtime scaffold for P2)

### Add after validation (v1.0.x, if time)

- [ ] D1 semantic stratification (thin vs planar) — highest value per unit effort; feeds the hybrid narrative
- [ ] D2 signed bias + residual heatmap (may already fall out of the plots)
- [ ] D3 asymmetric CD
- [ ] D4 negative control + D12 collinearity audit (both are cheap and make the report defensible)
- [ ] Second camera (`image_01`) and merged per-camera results
- [ ] D6 route A/B if a second mesh route is already installed

### Future consideration (v2+)

- [ ] D11 point-to-mesh independent check
- [ ] D5 protocol ablation (camera-origin vs LiDAR-origin)
- [ ] D7 mesh diagnostics ↔ recall correlation
- [ ] D8/D9/D10 statistical refinements
- [ ] Fisheye cameras, 360° coverage, multi-segment, GPU casting, Gaussian/hybrid representations

---

## 5. Expected behaviour on KITTI-360 driving scenes

These are **expectations to design the acceptance gates around**, not measured facts. Absolute values are MEDIUM-LOW confidence; the *shape* of the behaviour is HIGH confidence (it follows from MVS geometry, not from a single paper).

### 5.1 Coverage

- Only the forward wedge is reconstructable: the two perspective cameras are a forward-pointing stereo pair (≈90° HFOV each, 1408×376). Fisheye cameras cover the sides/back but are out of scope (A12). Expect the usable evaluation region to be roughly the front 80–100°, and reduce further at image borders, sky, and untextured road.
- A large fraction of raw HDL-64E returns falls outside that wedge. **Report `N_valid / N_raw` explicitly**; a headline recall computed over 360° GT would be meaningless.
- Ranges beyond ~50–60 m have few overlapping views and degrade quickly; the 0–50 m cutoff is well chosen.

### 5.2 Range error

- MVS depth error grows roughly quadratically with range (disparity error ∝ 1/z ⇒ depth error ∝ z²). Expect the 20–40 m bucket to be ~2–5× worse than 0–20 m, and 40–50 m worse again. **A flat error-vs-range curve is suspicious** (likely a range filter or leakage).
- Planar, textured, well-observed surfaces (facades, walls, road near the vehicle) should dominate the low-error tail.
- Expect heavy tails: a small fraction of rays hit a distant background through a hole or hit a floater. Hence MAE/RMSE ≫ MedAE is normal; report all three.
- Sign of the bias is diagnostic: net **negative** (pred closer) ⇒ floaters/spurious surfaces or mesh inflation; net **positive** (pred farther) ⇒ missed foreground/thin structure or holes.
- **The GT itself may have a motion-distortion floor.** HDL-64E scans are stored as one rigid point set (x,y,z,intensity only — no per-point timestamp), and the standard KITTI-format convention does not per-point compensate ego motion, so a static world is smeared by up to `v · T_scan/2` (≈0.4–0.7 m at 30–50 km/h over a 0.1 s revolution). If KITTI-360 is uncompensated (verify empirically before relying on it), this sets a soft floor on achievable range error at long range. State it in the report and use it to justify *not* chasing sub-10 cm accuracy (T20). A cheap check: residual should correlate with azimuth in a half-revolution pattern if distortion dominates.

### 5.3 Recall and thin structures

- Ray recall within the front FOV should be high on large surfaces and drop sharply on: `pole`, `smallpole`, `traffic sign`, `traffic light`, `lamp`, `fence`, `guard rail`, `vegetation`; and near image borders, sky, glass, and untextured road at grazing incidence.
- Vegetation is the hardest class: it is volumetric, semi-transparent to MVS, and produces both holes (misses) and floaters (negative bias).
- **This class gap is the milestone's most important expected result.** It is also the empirical premise for the later hybrid representation, so it must be measured, not asserted. Gate it as an *ordinal* expectation (thin-class recall@0.2 clearly below planar-class recall@0.2) rather than an absolute number.

### 5.4 Metric magnitudes

Rough expectations for a mesh from RGB MVS evaluated with GT-defined rays (to be replaced by measured values, then frozen):

| Quantity | Expected order of magnitude | Confidence |
|---|---|---|
| Miss rate inside front FOV | 10–40% | MEDIUM |
| Range MedAE, 0–20 m | 0.05–0.30 m | MEDIUM-LOW |
| Range MedAE, 20–40 m | 0.2–0.8 m | LOW |
| Range MAE, overall | 0.3–1.0 m | LOW |
| CD‑L1 (mean NN, m) | 0.2–1.0 m | LOW |
| Recall@0.20 m | 0.35–0.75 | LOW |
| Recall@0.05 m | 0.05–0.30 | LOW |
| F@0.20 m | 0.35–0.70 | LOW |

**Sanity floors (bug detectors, not quality targets):** overall MedAE < 0.02 m ⇒ suspect GT leakage (A5/A11) or that the mesh was built from LiDAR; overall MedAE > 2 m or miss rate > 70% inside the FOV ⇒ suspect calibration/pose/scale error (T2/T10), not "a bad mesh".

### 5.5 What "done" looks like (frozen acceptance)

A run is *complete* when all of the following hold. Freeze thresholds after the first successful end-to-end run, then never tune the mesh to pass them (A10).

**Gate A — harness correctness (must pass, no exceptions)**
1. Analytic plane/sphere/cube: `max |r_pred − r_analytic| < 1e-4 m`; known misses return `∞`.
2. Embree/Open3D cross-check: ≥99.9% hit/miss agreement, `max |Δt| ≤ 1e-4 m`, on ≥10⁵ real rays.
3. Determinism: two runs on the same inputs produce bit-identical metrics JSON (or differences ≤1e-9).
4. Negative control: a deliberate 5 cm extrinsic perturbation or cam00/cam01 swap degrades MedAE and recall monotonically; if not, the metric is insensitive.

**Gate B — data integrity**
5. Every dropped ray is attributable to a named filter; `N_raw = N_valid + Σ N_dropped(reason)` exactly.
6. `N_valid ≥ 20,000` per evaluated frame and ≥30 frames evaluated.
7. Scale/frame gate (T10) passes; no alignment step exists in the code path.
8. Ray-set hash and mesh hash recorded in the metrics JSON.

**Gate C — plausible accuracy** (bands from §5.4, tightened after the first run)

| Bucket | MedAE | MAE | RMSE | Miss rate |
|---|---|---|---|---|
| 0–20 m | ≤ 0.25 m | ≤ 0.50 m | ≤ 1.0 m | ≤ 25% |
| 20–40 m | ≤ 0.60 m | ≤ 1.20 m | ≤ 2.5 m | ≤ 35% |
| 40–50 m | ≤ 1.20 m | ≤ 2.50 m | ≤ 5.0 m | ≤ 50% |

Overall: CD‑L1 ≤ 1.0 m, Recall@0.20 ≥ 0.40, F@0.20 ≥ 0.40. F@0.05 is reported but **not** gated (expect it to be low).

**Gate D — expected structure (ordinal, the real evidence)**
9. Thin-structure Recall@0.20 ≤ 0.6 × planar Recall@0.20, with `n_thin ≥ 500` rays so the gap is not noise.
10. Error and miss rate increase monotonically across B1→B2→B3 (allow a documented exception with explanation).
11. `RMSE > MAE > MedAE` (heavy right tail, as expected). `F(τ)` must be non-decreasing in τ — a violation means a bug in the matching code, not a finding.
12. Bias magnitude < MedAE (a large systematic bias means a frame/scale problem, not noise).

**Gate E — deliverables**
13. One command reproduces: rays → cast → metrics → plots + report tables.
14. `metrics.json` (schema-versioned) + per-frame CSV + 3 plots + mesh stats + runtime log.
15. Report states protocol, ray-set hash, mesh route/config, hardware, and the limitations from §5.1–5.3.

---

## 6. Prioritization Matrix

| Feature | Value | Cost | Priority |
|---|---|---|---|
| T2 calibration/pose | HIGH | MEDIUM | P1 |
| T3 frame alignment | HIGH | MEDIUM | P1 |
| T4 dynamic/ego filter | HIGH | MEDIUM | P1 |
| T6 ray generator + T5 cache | HIGH | LOW | P1 |
| T7 FOV filter | HIGH | LOW | P1 |
| T8 mesh pipeline | HIGH | HIGH | P1 (route decision is the critical path) |
| T9 cleanup/stats + T10 scale gate | HIGH | MEDIUM | P1 |
| T11 BVH + T12 cross-check | HIGH | MEDIUM-HIGH | P1 |
| T13 p_pred + T14 NN | HIGH | LOW | P1 |
| T15 metric suite | HIGH | MEDIUM | P1 |
| T16 artifacts | HIGH | LOW | P1 |
| T17 reproducibility | HIGH | MEDIUM | P1 |
| T18 analytic tests | HIGH | LOW | P1 |
| T19 instrumentation | MEDIUM | LOW | P1 |
| T20 protocol docs | HIGH | LOW | P1 |
| D1 semantic stratification | HIGH | MEDIUM | P2 (best value after core) |
| D2 bias/tails/maps | HIGH | LOW | P2 |
| D3 asymmetric CD | MEDIUM | LOW | P2 |
| D4 negative control | HIGH | LOW-MEDIUM | P2 |
| D12 collinearity audit | MEDIUM | LOW | P2 |
| D6 route A/B | HIGH (decides the open question) | LOW if T8 clean | P2 |
| D10 ray subsampling | MEDIUM | LOW-MEDIUM | P2 (only if runtime demands) |
| D11 point-to-mesh | MEDIUM | MEDIUM | P3 |
| D5 protocol ablation | MEDIUM | MEDIUM | P3 |
| D7 mesh diagnostics | MEDIUM | MEDIUM | P3 |
| D8/D9 stats | LOW-MEDIUM | LOW | P3 |

**Critical path:** T2 → T3 → T4/T5 → T6/T7 → T8/T9/T10 → T11/T12 → T13/T14 → T15/T16. T8 (mesh route) is the only genuinely high-risk item; start it first and keep it behind a one-line config switch so the harness can be built against a placeholder mesh.

---

## 7. Competitor / prior-work feature analysis

| Capability | LiDARsim (CVPR'20) | LiDAR4D / LiDAR-NeRF | LiDAR-GS / GS-LiDAR | LiDAR-RT (CVPR'25) | FGGS-LiDAR | **This milestone** |
|---|---|---|---|---|---|---|
| Representation | surfel mesh + neural raydrop | 4D neural field | 3DGS / 2DGS | GS + proxy BVH | 3DGS→TSDF→mesh | **RGB MVS mesh** |
| Reconstruction supervision | LiDAR (+imagery) | LiDAR | LiDAR (+images) | LiDAR | LiDAR | **RGB only; LiDAR eval-only** |
| Ray definition | LiDAR beams | range image / beams | panoramic range image | arbitrary sensor | arbitrary | **GT-defined rays from camera origin** |
| Miss / no-return handling | explicit raydrop model | raydrop head | raydrop SH | physical | none | **miss = ∞, counted in recall** |
| Primary range | full | ≈80 m clip (normalized) | range-image bounded | full | full | **0–50 m, 20 m buckets** |
| CD convention | — | mean **squared**, summed (verified) | mean **squared**, summed (verified) | squared (unverified) | squared `chamfer3D` | **L1 (m) and L2 (m²) defined explicitly (§1.5)** |
| "F@5 cm" convention | — | threshold on **squared** distance ⇒ ≈22 cm (verified) | same helper ⇒ ≈22 cm (verified) | reported 5 cm (unverified) | reported 5 cm | **true Euclidean 5/20/50 cm** |
| Depth metrics | — | RMSE (cm-scale table) + MedAE (m) — mixed units | same tables | RMSE/MedAE | CD/F | **range error in metres only** |
| Runtime reporting | — | training hours / FPS | FPS | 30 FPS @64×2650 (GPU) | >500 FPS (GPU) | **CPU rays/s + cast FPS, explicitly non-real-time** |
| Semantic stratification | — | — | — | — | — | **planned (D1) — a gap in prior work** |
| Calibration-sensitivity test | — | — | — | — | — | **planned (D4) — a gap in prior work** |

Two conclusions that shape requirements:

1. **Our metric semantics are stricter than the de-facto LiDAR-NVS convention.** In the LiDAR4D/GS-LiDAR lineage (verified in source), "F@5 cm" is applied to *squared* distances and is therefore effectively ≈22 cm; some downstream papers inherit that helper. Our F@0.05 m will look far worse than published numbers while being more honest. The report must say this explicitly and never put our F@0.05 beside a literature "F-score 0.9" without the caveat.
2. **Semantic stratification and harness sensitivity checks are not standard in this literature.** They are cheap here and they are what turns "a mesh with 30 cm error" into evidence for the later hybrid argument.

---

## 8. Evidence quality and open questions

| Claim | Confidence | Basis |
|---|---|---|
| KITTI-360 sensors/axes/resolutions/pose files; `poses.txt` subsampled | HIGH | Official KITTI-360 documentation (fetched 2026-10-30) |
| KITTI-360 semantic class names/ids | HIGH | `kitti360Scripts/kitti360scripts/helpers/labels.py` |
| `calib_cam_to_velo` is cam0→Velodyne; `cam0_to_world` is rectified cam0→world | HIGH | Official docs + `loadCalibration.py` |
| Open3D `RaycastingScene` returns `t_hit = inf` / `INVALID_ID` on miss | HIGH | Open3D test source + API docs |
| LiDAR-NVS F-score threshold is applied to squared distance | HIGH | Source code of LiDAR4D `utils/metrics.py` and GS-LiDAR `chamfer/fscore.py` |
| CD-L2 in that lineage is mean squared distance, summed over directions | HIGH | Same sources + `chamfer_3D` semantics |
| Literature depth RMSE/MedAE units are inconsistent (cm vs m) | MEDIUM | Numeric tables (e.g., RMSE 5.89 vs MedAE 0.0198) are only self-consistent under mixed units; not stated in the papers |
| Scan-motion distortion sets a ~0.4–0.7 m floor at highway speed (no per-point timestamps in KITTI-360 `.bin`) | MEDIUM | HDL-64E 10 Hz rotation + single rigid point set; magnitude follows from ego speed, not measured here |
| Absolute error bands in §5.4 | LOW-MEDIUM | Inferred from MVS geometry + LiDAR-NVS table magnitudes; **must be replaced by measured values** |
| Mesh route (COLMAP vs OpenMVS vs feed-forward) | OPEN | Deliberately unresolved; research STACK.md should decide; D6 provides the evidence path |
| Typical triangle budget for a 100-frame segment | OPEN | Depends entirely on MVS resolution + decimation; measure and record |
| Whether `image_01` (right perspective) adds usable coverage or is redundant with `image_00` | OPEN | Depends on baseline/disparity quality; test empirically before committing to multi-camera |

### Open questions the requirements phase should resolve
1. Which camera(s) are evaluated in v1.0 — `image_00` only, or `image_00` + `image_01` merged? (Affects ray counts, coverage, and per-camera merging code.)
2. Is B3 `[40,50]` accepted (recommended) or is a 50–60 m secondary row required by the course rubric?
3. Is `t_max = 60 m` acceptable, or should the cast bound be infinite with counts of >60 m hits reported separately?
4. Does the mesh route produce a watertight mesh? If not, hole-filling must be explicitly flagged (T9) because it changes recall in a way that must be disclosed.
5. What is the wall-clock budget for one full run? This determines whether D10 ray subsampling is P1 or P2.

---

## Sources

- KITTI-360 official documentation (folder structure, calibration, poses, sensor axes, subsampled `poses.txt`, 1408×376 perspective images, HDL-64E): https://cvlibs.net/datasets/kitti-360/documentation.php — HIGH
- `kitti360Scripts` `loadCalibration.py`, `helpers/labels.py`, `helpers/project.py`: https://github.com/autonomousvision/kitti360Scripts — HIGH
- LiDAR4D metric implementation (`utils/metrics.py`, `fscore`, `chamfer_3DDist`, `DepthMeter`, `PointsMeter`): https://github.com/ispc-lab/LiDAR4D — HIGH
- GS-LiDAR metric implementation and config (`chamfer/fscore.py`, `utils/metrics_utils.py`, `configs/kitti360_nvs_*.yaml`, seq 0000): https://github.com/fudan-zvg/GS-LiDAR — HIGH
- LiDAR4D paper (CD definition, F-score threshold, KITTI-360 table): https://arxiv.org/abs/2404.02742 — HIGH for definitions, MEDIUM for numbers
- GS-LiDAR paper (KITTI-360 static CD 0.2382, F 0.9055, depth RMSE 5.8925, MedAE 0.0198): https://arxiv.org/abs/2501.13971 — MEDIUM (units ambiguous)
- LiDAR-RT (proxy BVH, 30 FPS @64×2650, CD/F/RMSE table): https://arxiv.org/abs/2412.15199 — MEDIUM
- FGGS-LiDAR (3DGS→TSDF→watertight mesh, GPU ray casting, outdoor CD 0.025 / F 0.979): https://arxiv.org/abs/2509.17390 — MEDIUM (object-level, different protocol)
- LiDARsim (reconstruct-then-simulate, mesh + learned raydrop): https://arxiv.org/abs/2006.09348 — MEDIUM
- Open3D `RaycastingScene` (Embree-backed CPU casting, `cast_rays`, `t_hit = inf` on miss): https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html and https://github.com/isl-org/Open3D/blob/master/python/test/t/geometry/test_raycasting_scene.py — HIGH
- Möller & Trumbore, "Fast, Minimum Storage Ray/Triangle Intersection", JGT 2(1):21–28, 1997: http://graphics.cornell.edu/pubs/1997/MT97.html — HIGH
- AlignMiF (KITTI-360 LiDAR CD/F-score magnitudes): https://arxiv.org/abs/2402.17483 — MEDIUM
- 2DGS/OMeGa-style CD‑L1 and F-score@5 cm conventions (mean unsquared distance; 5 cm threshold): https://arxiv.org/abs/2509.24308 — MEDIUM
- Project-internal: `README.md` §0.5.2, §1.3, §1.4, §7; `.planning/PROJECT.md`; `docs/proposal_V2.tex` — HIGH (scope authority)

---

*Feature research for: mesh-only KITTI-360 LiDAR simulation baseline + GT-ray evaluation*
*Researched: 2026-10-30*
