# Project Research Summary

**Project:** Hybrid Mesh–Gaussian LiDAR Simulation
**Domain:** Offline computer-vision / geospatial research pipeline — known-pose MVS reconstruction + ray-cast LiDAR evaluation (GPU-programming course project; CPU-first v1.0)
**Researched:** 2026-10-30
**Confidence:** MEDIUM — HIGH on mechanics, conventions and tool availability; LOW on absolute accuracy bands (they are expectations to be replaced by measured values)

---

## Executive Summary

This milestone is **not** a LiDAR-NeRF/GS task. The published neighbours (LiDAR-GS, GS-LiDAR, LiDAR-RT, LiDAR4D, LiDAR3D/NeRF) optimize a representation *against LiDAR supervision*; v1.0 deliberately does the opposite — reconstruct from RGB + provided metric poses, then use LiDAR **only at evaluation time**. That means the contribution is **measurement validity**, not state-of-the-art fidelity. Experts build this as five separable stages with hard artifact boundaries: **prepare → reconstruct → build BVH → cast fixed GT rays → evaluate/report**. Each stage reads and writes a versioned, hashed artifact; the GT ray set is generated once, frozen, and reused by every representation (mesh now, Gaussian/router later).

The recommended route is explicit: **COLMAP 4.0.4 (CPU, already installed) as a pose container + sparse triangulator, then OpenMVS 2.4.0 for densification and meshing** — `feature_extractor → exhaustive_matcher → point_triangulator` with poses/intrinsics frozen, `InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → RefineMesh`, then Open3D cleanup. KITTI-360 already supplies metric poses, so COLMAP's `mapper`/SfM is never run; it exists only because the prebuilt OpenMVS bundle **does not ship `CreateStructure`** (verified locally). Casting uses an in-repo **Numba CPU BVH with Möller–Trumbore**, cross-checked against **Open3D `RaycastingScene` (Embree)**. Learning-based reconstruction (VGGT/MASt3R) is deferred (no native mesh, up-to-scale, non-commercial, confounds the baseline). **2DGS→TSDF→Marching Cubes is deferred as a later ablation, not the baseline** — using it now would make the "mesh-only" baseline neither independent nor mesh-only.

The risks are concentrated in two places. First, **silent frame/calibration/scale errors** produce plausible-looking but meaningless numbers (calibration inversion, raw-vs-rectified mixing, accumulated semantic clouds used as per-frame GT, mesh in COLMAP's arbitrary frame). Second, **protocol degeneracy**: under GT-defined rays, CD and F-score are largely re-encodings of range error, so they are not independent evidence. Both are mitigated by the G1–G5 gates, a SHA-256-frozen ray set, pre-registration of segment/frames/τ/buckets, sanity floors (MedAE < 2 cm ⇒ suspect leakage; > 2 m or miss > 70 % ⇒ suspect calibration), and by building the **evaluation harness on synthetic/placeholder data before any real mesh exists**. The single blocking external risk is that the **local dataset is not a complete official KITTI-360 release** (no `calibration/`, `data_poses/`, `data_2d_raw/`, `data_2d_semantics/`), so acquisition is Phase 1 and must fail loudly rather than be substituted with KITTI-raw calibration.

---

## Key Findings

### Recommended Stack (from STACK.md)

COLMAP and OpenMVS are complementary, not competing: COLMAP provides the pose-consistent sparse cloud (the shipped OpenMVS bundle has no SfM front end), and OpenMVS provides the surface stage where it is strongest (visibility-aware Delaunay produced the lowest mesh error in the UseGeo evaluation; Poisson tends to close holes and hallucinate surfaces). Everything on the primary path runs **CPU-only** on this host; the installed COLMAP 4.0.4 is built **without CUDA**, so `patch_match_stereo` is not available and is not on the critical path.

**Core technologies:**
- **Python 3.11 (uv) + NumPy 2.4.4 + SciPy 1.17.1** — env, geometry, `cKDTree` for CD/F/P/R — every wheel supports 3.11; `cKDTree(workers=-1)` handles 1e5×1e5 in seconds, no PyTorch3D needed.
- **Numba 0.68.1** — JIT CPU BVH + Möller–Trumbore — satisfies the locked "CPU BVH first" decision without a C++/pybind11 build; `numpy<2.6` compatible.
- **open3d-cpu 0.20.0** — mesh I/O, cleanup, decimation, and **Embree `RaycastingScene` as the independent cast oracle**; `open3d-cpu` avoids dragging a CUDA runtime in.
- **trimesh 5.2.0** — watertight/repair/PLY introspection — complements Open3D; do not duplicate its TSDF/raycasting.
- **pandas 3.x + Matplotlib 3.10.x + Pillow 12 + PyYAML 6** — metrics tables, plots, PNG/mask I/O, config.
- **psutil / tqdm / pytest 9 / ruff** — peak RSS, progress, phase-1 gates, hygiene.
- **COLMAP 4.0.4 CPU** (`/data-20T/conda_envs/colmap/bin/colmap`, not on `PATH`) — `feature_extractor`, `exhaustive_matcher`, `point_triangulator`, `model_converter` with `--FeatureExtraction.use_gpu 0` / `--FeatureMatching.use_gpu 0`.
- **OpenMVS 2.4.0** — flat prebuilt Ubuntu x64 bundle, verified executable on this host with no `LD_LIBRARY_PATH`; **no `CreateStructure`/`ExtractKeyframes`**; `InterfaceCOLMAP` accepts **PINHOLE only**; download via the `api.github.com` release-asset endpoint because `github.com` is unreachable here.
- **Optional only if the accuracy bar is missed:** COLMAP 4.2.1 `cuda_129` for `patch_match_stereo → stereo_fusion` (COLMAP depth, OpenMVS mesh).

**Host facts that shape the plan:** RTX 4090 48 GB / driver 580.82.09 but **no `nvcc`** (no CUDA compile in v1.0); 32 cores / 125 GB RAM (OpenMVS CPU densify feasible); Ubuntu 24.04.2 / glibc 2.39; system Python 3.12.3; disk tight (59 GB on `/`, 44 GB on `/data-20T`) — download only the minimal sequence subset.

**Route decision (normative):**

| Route | v1.0 verdict | Reason |
|---|---|---|
| **COLMAP known-pose ingest + OpenMVS densify/mesh** | **PRIMARY** | Metric poses given; CPU-only; visibility-aware mesh; no Poisson ballooning; prebuilt binary verified |
| COLMAP 4.2.1 dense + `delaunay_mesher`/`poisson_mesher` | Fallback / quality upgrade | Single toolchain, best depth, but requires a CUDA build; Poisson closes holes (inflates recall) |
| VGGT / MASt3R | **Deferred (out of scope)** | No native mesh, up-to-scale (would need Umeyama or LiDAR leakage), GPU-heavy, non-commercial licences, confounds attribution |
| 2DGS → TSDF → MC | **Deferred to a later ablation** | Requires a Gaussian stack and `nvcc`; shares failure modes with any Gaussian baseline; not independent |

### Expected Features (from FEATURES.md)

**Must have — table stakes (T1–T20, all P1):**
- **Data/ray foundation:** segment manifest + low-dynamic selection (T1); one calibration/pose composition module with `T_AB` algebra (T2); frame alignment handling for subsampled `poses.txt` (T3); dynamic + ego filtering of GT (T4); static GT cache (T5); GT-defined ray generator (T6); per-camera FOV/visibility filter (T7).
- **Reconstruction:** route-agnostic, scripted mesh pipeline (T8); cleanup + mesh statistics (T9); **scale/frame sanity gate that fails loudly and never aligns** (T10).
- **Casting:** native CPU BVH + Möller–Trumbore with `t_min`/`t_max`, two-sided hits, deterministic tie-break (T11); Embree cross-check (T12); `p_pred` reconstruction (T13); KD-tree NN engine (T14); runtime/memory instrumentation (T19).
- **Metrics/deliverables:** full metric suite with buckets (T15); schema-versioned JSON + tidy CSV (T16); one-command reproducibility harness (T17); analytic plane/sphere/cube self-tests (T18); protocol + metric definitions in the report (T20).

**Should have — differentiators (after core is green):**
- **D1 semantic stratification (thin vs planar)** — highest value per unit effort; turns a scalar into the empirical premise for the hybrid thesis (mesh recall collapses on poles/fences/vegetation, holds on road/facade).
- **D2 signed bias + tail quantiles + residual/miss maps** — distinguishes "too close" (floaters/smoothing) from "too far" (holes/missed thin structure).
- **D3 asymmetric CD (accuracy vs completeness)**, **D4 negative-control perturbation test** (5 cm extrinsic shift or cam swap must degrade metrics monotonically), **D12 metric-collinearity audit** (ρ between |range error| and NN distance).
- **D6 route A/B** (decides the open question with numbers), **D11 point-to-mesh GT→mesh distance** (the only genuinely independent geometric check), **D5 camera-origin vs LiDAR-origin protocol ablation**.

**Defer / anti-features (explicitly excluded):**
- Gaussian baselines (A1), GPU/CUDA/OptiX (A2), 2DGS→TSDF→MC as the mesh source (A3), LiDAR-supervised fitting or depth supervision (A4, GT leakage), mesh built from the accumulated GT cloud (A5, circular — oracle diagnostic at most), intensity/ray-drop/multi-return (A6), dynamic/scene graph (A7), multi-dataset/multi-segment (A8), PSNR/SSIM/LPIPS (A9), per-frame/per-bucket mesh tuning (A10), Umeyama/ICP alignment before eval (A11), fisheye in v1.0 (A12), BVH micro-optimisation (A13), pooled-only reporting (A14), "real-time" claims (A15), end-to-end training (A16).

**Metric contract to freeze (normative):** `r_gt = ‖p_C‖` **from the camera optical centre**, `d = p_C/‖p_C‖`, `t_min = 1e-4 m` (numerical guard; domain gate is `r_min = 1.5 m`), `t_max = 60 m`; miss ⇒ `t_pred = +inf`, excluded from range error, counted in miss rate/ray recall; two-sided intersection (no back-face cull). Report MAE/RMSE/MedAE/**signed bias**/P90–P95 (hits only, `null` when empty); CD-L1 = mean **unsquared Euclidean**, CD-L2 = mean **squared** distance, both directions averaged with ½; P/R/F at τ ∈ {0.05, 0.20, 0.50} **metres**, primary τ = 0.20; buckets by `r_gt`: **[0,20), [20,40), [40,50]** primary, (50,60] secondary only. Headline = micro-average over pooled rays; also report macro per-frame ± std, per-camera first, and **exactly one camera per ray** (dedupe by most-central projection, then lowest camera id).

### Architecture Approach (from ARCHITECTURE.md)

A stage-oriented offline pipeline with five hard boundaries and a canonical artifact contract. The mesh lives in a **segment-local metric frame (SEG)** — same orientation as KITTI world, origin at the median camera centre — so float32 never sees kilometric coordinates. Casting is expressed as a batch `SceneBackend` protocol (`RayBatch` → `HitBatch` with `t`, `hit`, `prim_id`, `bary`, `normal`, `served_by`), so P2 can add a GPU/router backend without touching data prep, ray generation, metrics, or reporting. Reconstruction is a `MeshReconstructor` adapter whose only downstream contract is `MeshArtifact` (`scene.ply` + manifest in SEG, metres).

**Five stages:**
1. **Prepare** — dataset preflight, calibration/pose composition, segment selection, dynamic/ego masks, per-frame static GT, frozen GT ray `.npz` + manifest, known-pose COLMAP TXT model.
2. **Reconstruct** — route adapter → raw mesh → `ingest` validates scale/bbox/orientation/triangle count → cleaned canonical mesh + manifest.
3. **Build BVH** — flat, cache-friendly arrays (nodes, triangle indices, precomputed edges), built once per mesh, hashed and cached.
4. **Cast GT rays** — fixed rays transformed to SEG, batch traversal, `PredictionBundle` per frame; misses are `+inf`.
5. **Evaluate / report** — range metrics, CD/F/P/R, misses, buckets, throughput, `metrics.json` + tidy CSV + plots + report.

**Supporting components:** `hls.common.frames` (normative `T_dst_src`, SEG, pose lookup), `hls.common.calib` (KITTI-360 loaders + assertions), `hls.common.io` (hashes/manifests), `hls.data.*` (layout, frames, segment, masks, gt_rays), `hls.recon.*` (base, export_colmap, routes, ingest, postprocess), `hls.cast.*` (rays, bvh, backend, backends/cpu, pipeline), `hls.eval.*` (metrics, pointcloud, buckets, report), `hls.pipeline` (run IDs, manifests, caching). Layout: one installable `src/hls/` package, `configs/` per stage, `scripts/00_check_dataset.py … 06_report.py`, `tests/` including a synthetic end-to-end test, `artifacts/` (large, gitignored) vs `results/` (small, committed), and `docs/frames.md`, `docs/protocol.md`, `docs/repro.md`.

**Coordinate chain (single source of truth):** `T_W_I = poses.txt` (IMU→world), `T_I_Ck = calib_cam_to_pose.txt` (camera→IMU), `T_V_C0 = calib_cam_to_velo.txt` (camera 0→Velodyne), `R_rect_k` from `perspective.txt`; compose `T_Ck_rect_V = R_rect4(k) · inv(T_V_C0 · inv(T_I_C0) · T_I_Ck[k])`. Use `poses.txt` + `calib_cam_to_pose` + `R_rect` as the source of truth and use `cam0_to_world.txt` **only as an independent assertion** (`atol=1e-3`). `K = P_rect[:3,:3]` — never fold the stereo baseline `Tx` into a single-camera model.

### Critical Pitfalls (from PITFALLS.md)

1. **Calibration-chain inversion (C1)** — `calib_cam_to_pose` is camera→IMU, `calib_cam_to_velo` is cam0→Velodyne, `poses.txt` is IMU→world. One module, named transforms, round-trip/det(R) assertions, and a **projection overlay gate (< 2 px at 10–30 m)**. Warning signs: negative/0.3 m camera height, mirrored scene, > 2 px offset.
2. **Raw-vs-rectified mixing and the image_01 baseline (C2)** — use `data_rect` + `P_rect` + `R_rect` consistently; 0.1° at 50 m is 8.7 cm, the same order as the error being measured. For image_01 use its **physical pose** and `K = P_rect_01[:3,:3]`; do not add `Tx` twice (≈0.5–0.6 m lateral error).
3. **Velodyne motion distortion (C3) — required, not optional** — a 0.1 s sweep at 40 km/h spans ≈1.1 m; at driving speed distortion is metre-scale, larger than the entire error budget. Port/reuse `curlVelodyneData` (azimuth-interpolated rigid correction from `poses.txt` + `calib_cam_to_velo`) to NumPy, unit-test against the devkit, and apply the compensated points to GT rays, masks, overlays and validation. Verify: signed range error vs azimuth must be flat, not sinusoidal.
4. **Accumulated `data_3d_semantics` clouds as per-frame GT (C4)** — they are multi-frame, world-frame, and include occluded surfaces; using them makes recall meaningless and errors look better than reality. GT must come only from the per-frame `.bin` (~0.17 M points for HDL-64E), motion-compensated, with the exact frame pose. Use accumulated clouds only for label transfer (NN within ~5 cm) or mask building.
5. **Wrong sensor model (C5)** — **KITTI-360 has 2 perspective (`image_00/01`, 1408×376, ≈90° HFOV stereo) + 2 fisheye (`image_02/03`)**, not 4 pinhole. v1.0 uses 00/01 only; document the evaluated angular domain and never invent phantom cameras.
6. **Asymmetric dynamic/ego masking (C6)** — one mask artifact applied to images, GT, projections and validation. Ego removal by **3D ROI in Velodyne coordinates** (`|x|<2.5`, `|y|<1.2`, `−2.0<z<0.3`), not an isotropic radius (M1 deletes real near-field ground). Log residual dynamic fraction.
7. **MVS failure exactly where the metric is densest (C7/C8)** — textureless road, sky and glass; moving objects and exposure changes create ghost geometry. Mask sky, use geometric consistency + ≥2–3-view support, remove fused points in dynamic bboxes, keep the largest component, and report road as its own bucket.
8. **Mesh in the wrong frame/scale (C9)** — never run SfM then align; fabricate the known-pose COLMAP model, triangulate with fixed poses, and assert bbox ratio 0.8–1.2 and camera centres matching the trajectory. Any similarity transform fitted to metrics is leakage.
9. **Poisson ballooning / camera inside the mesh (C10)** — Poisson always closes the surface; keep and trim the density field (5–10 % quantile), crop to the corridor, and run the **camera-outside test** (forward hit rate 0.3–0.9, not ≈1.0; no hits within 1 m on multiple axes). Prefer visible-surface OpenMVS meshing.
10. **Miss/FOV/denominator errors (C13)** — behind-camera rays, out-of-image rays, `r_gt = ‖p_L‖` instead of `‖p_C‖`, `inf` entering `mean`/`median`, `t_max = 50` truncating 50–60 m. Use the GT waterfall (`N_raw = N_valid + Σ N_dropped(reason)` exactly) and masked statistics.
11. **Camera-origin occlusion (C14)** — removing parallax does not remove occlusion; a GT point 0.5–0.8 m from the camera can sit behind a pole. Build a **GT-only** z-buffer visibility flag, keep it in the frozen ray set, report flagged points as a separate bucket plus a "full" variant, and never use reconstructed depth to define the evaluation domain.
12. **Multi-camera double counting (C15)** — assign exactly one camera per GT point deterministically, freeze `camera_id`, report per-camera and deduplicated aggregate.
13. **Accuracy-on-hits / hole hiding (C16)** — every accuracy row must carry its coverage number in the same row; a variant with more holes must not look "better".
14. **CD/F degeneracy (C17)** — under GT rays, each predicted point lies on the ray to its GT point, so CD ≈ f(range error) and F@τ ≈ error CDF (modulo hit-rate). State this; supplement with D11/D12; never present them as independent confirmations. Also: **published "F@5 cm" is often applied to squared distances (≈22.4 cm Euclidean)** in the LiDAR4D/GS-LiDAR lineage — our true-Euclidean F@5 will look far worse and that must be stated, never compared without the caveat.
15. **Grazing incidence / ground domination (C18)** — `δ_r ≈ δ_n/|n·d|`; a 2 cm height error at 2° incidence becomes ≈50 cm at 40 m. Report error vs incidence, point-to-plane error with GT normals, per-class tables, and the global height offset.
16. **Cherry-picking (C19)** — pre-register segment, frame range, exclusions, τ, buckets and aggregation in Phase 1; freeze the ray set before reconstruction; dev/report frame split; report all τ and all buckets with counts.
17. **Reproducibility drift (C20)** — full run manifest (git SHA, dataset release, calibration/frame-list/ray-set/mesh hashes, library versions, CPU/threads, seeds), pinned threads, evaluation re-runnable from `rays + mesh + manifest` alone.
18. **2DGS route as the baseline (C21)** — mesh provenance must be "COLMAP/OpenMVS MVS from images + KITTI poses, no Gaussian model in the loop"; no shared depth maps/checkpoints with any Gaussian baseline.

**Gate definitions (freeze once, then enforce):**

| Gate | Pass criteria |
|---|---|
| **G1 Calibration** | Transform round-trip < 1e-9, `det(R)=+1`, `RᵀR=I` (1e-6); LiDAR→image overlay < 2 px at 10–30 m on ≥3 frames; camera height 1.4–1.9 m, positive and stable; `T_W_C0_rect` vs `cam0_to_world` < 1e-3 |
| **G2 Ray set** | Every ray `z_cam>0`, projects inside image with ≥5 px margin, `1.5 ≤ r_gt ≤ 50 m`, exactly one `camera_id`; waterfall counts sum exactly; SHA-256 frozen before reconstruction |
| **G3 Mesh** | Bbox ratio 0.8–1.2 vs GT extent; camera-outside test passes (forward hit rate 0.3–0.9, ~0 behind); degenerate triangles < 0.1 %; one component > 95 %; median range error < 1 m on one frame before tuning (> 5 m ⇒ debug frames, not parameters) |
| **G4 Ray cast** | Analytic sphere/plane/box < 1e-4 m (PITFALLS: < 1e-6 m); miss/parallel/edge/inside/negative-t/`t<t_min` tests; Embree cross-check ≥ 99.9 % hit/miss agreement and `max|Δt| ≤ 1e-4 m` on ≥1e5 real rays; `t` metric (unit directions) |
| **G5 Metrics/repro** | Formulas + units in code and report; all τ and all buckets with counts; no `inf`/`NaN` in a mean; hits + coverage in every table row; byte-identical regeneration from frozen artifacts; manifest complete; median/IQR over ≥3 warm runs |

---

## Corrections & Contradictions Resolved (normative)

These are the research findings that **override** earlier project documents or conflicting research files. Roadmapper and planner should treat the right-hand column as settled.

| Ambiguity / contradiction | Resolution | Consequence |
|---|---|---|
| README §6 "4 perspective cameras" | **2 perspective (`image_00/01`) + 2 fisheye (`image_02/03`)** | Budget reconstruction for 2 cameras; fisheye is out of scope (A12); evaluable GT is a forward wedge, not 360° |
| STACK says poses from `cam0_to_world.txt`; ARCHITECTURE/PITFALLS say `poses.txt` chain | **`poses.txt` + `calib_cam_to_pose` + `R_rect` is the source of truth; `cam0_to_world.txt` is an assertion only** | One chain for all cameras; image_01 derived from the baseline, not guessed |
| FEATURES: `t_min = 0.10 m`; PITFALLS: `t_min = 1e-4 m` | **`t_min = 1e-4 m` numerical guard; `r_min = 1.5 m` is the domain filter** | Avoids deleting real 0.2–0.5 m geometry while preventing self-hits; log the `t_pred` histogram |
| FEATURES: `t_max = 60 m`; PITFALLS warns 50 m truncates a "40–60" bucket | **`t_max = 60 m`; buckets [0,20), [20,40), [40,50] primary, (50,60] secondary** | A 50.1 m point is a far hit, not a miss; bucket counts sum to the evaluated total |
| README/PROJECT: buckets "0–20 / 20–40 / 40–60" *and* "primary 0–50 m" | **B3 = [40,50]; separate 50–60 row labelled out-of-primary** | Removes the double-count / contradiction |
| ARCHITECTURE calls CD-L1 "coordinate L1" (`‖p−g‖₁`); FEATURES/PITFALLS call it unsquared Euclidean | **CD-L1 = mean unsquared Euclidean; coordinate-L1 is a separate labelled variant** | Comparable to 2DGS/OMeGa lineage; no silent convention change |
| ARCHITECTURE makes COLMAP MVS primary, OpenMVS optional; STACK verified the bundle lacks `CreateStructure` | **COLMAP pose ingest + OpenMVS mesh is primary**; COLMAP `delaunay_mesher`/4.2.1 dense is the fallback | `MeshReconstructor` adapter makes this a config switch; architecture's `colmap_mvs` adapter becomes the fallback route |
| ARCHITECTURE proposes C++17/pybind11 BVH; STACK proposes Numba | **Numba is the v1.0 implementation** (no build toolchain, fastest to correctness); C++/CUDA is an upgrade behind the same `SceneBackend` and flat-array layout | Keep the flat array schema documented so P2 can reimplement; Open3D remains the oracle either way |
| ARCHITECTURE: "treat raw velodyne as GT, record limitation"; STACK/PITFALLS: uncurling required | **Yes, uncurl is required** (metre-scale at driving speed); keep a raw-vs-curled ablation and report both | GT extraction cannot start until the uncurl port passes the azimuth-flatness check |
| STACK: 60–120 frame window; FEATURES: ≥30 evaluated frames; PITFALLS: 50–150 frames | **60–120 frame window (≈50–100 m), ≥30 evaluated frames, ≥20 000 valid rays/frame** | Tractable CPU budget and statistically meaningful per-frame spread |
| "Just use the README 2DGS→TSDF recipe for the mesh" | **No — OpenMVS MVS mesh is the baseline; 2DGS→TSDF→MC is a later ablation** | Mesh provenance is independent; comparisons stay attributable |
| Local dataset believed usable | **It is NOT a complete official release**: has `data_3d_raw` (136 GB) + `data_3d_bboxes`, missing `calibration/`, `data_poses/`, `data_2d_raw/`, `data_2d_semantics/`; raw images live under `image_00/data` (not `data_rect`); KITTI-raw-style calib files present | Phase 1 must acquire official files and validate; never substitute KITTI-raw calibrations; if only raw images exist, add an explicit rectification step |

---

## Implications for Roadmap

The ordering principle from all four files: **correctness-first, evaluation-first, reconstruction-last-but-started-early**. Data and calibration gates everything; the metric harness must be provably correct on synthetic data before a real mesh is interpreted; the mesh route is the long pole and must run as a parallel spike; the BVH must be validated against Embree before any reported number; reproducibility is enforced from day one, not retrofitted.

### Phase 1: Preflight, Acquisition & Calibration Foundation
**Rationale:** The local tree is not a complete official KITTI-360 release — this is the only blocking external dependency. Calibration is the #1 silent-failure source (C1/C2/C5/C12), and every later number inherits it. Also the right place to pre-register the protocol (C19) and scaffold the package/tests so later phases are not blocked by setup.
**Delivers:** `check_dataset` readiness report (present/missing per required file, sequence inventory); minimal download (calibration + `data_poses` + `data_2d_raw` for 00/01 + `data_2d_semantics` for one sequence, ~10–20 GB); `hls` package + configs + test harness + `docs/frames.md`; `calib.npz`, `frames.csv`, calibration overlays; `decisions.md` pre-registering segment criteria, frame window, τ, buckets, exclusions.
**Addresses:** T1/T2/T3 (foundation), T17 scaffolding, T20 protocol draft; corrects sensor model to 2+2.
**Avoids:** C1, C2, C5, C12, C19, M8; dataset substitution risk.
**Gate:** **G1** + explicit readiness sign-off. If official files cannot be obtained, **stop** — do not silently fall back to KITTI-raw calibration or OXTS poses.

### Phase 2: Data Foundation & Frozen GT Ray Set
**Rationale:** The frozen ray set is the evaluation contract (C20). It depends only on Phase 1 and can run in parallel with the mesh spike. Motion uncurling is mandatory (C3) and must be validated before GT is trusted.
**Delivers:** low-dynamic segment selection manifest; per-frame Velodyne uncurling port + validation; dynamic/ego masks applied to all consumers; per-frame static GT `.npz`; camera eligibility + deterministic primary-camera assignment; deduplicated GT rays with `ray_id`, `dir_C`, `t_gt`, `p_L`, `bucket`, `occluded`, `semantic_id`, plus manifest and SHA-256; known-pose COLMAP TXT model (PINHOLE, frozen poses/intrinsics).
**Addresses:** T1–T7, T5 cache; enables D1 labels later.
**Avoids:** C3, C4, C6, C13, C14, C15, M1, M8.
**Gate:** **G2** — waterfall counts sum exactly; ray hash frozen; signed error vs azimuth flat; ~0.17 M raw points/frame.
**Research flag:** needs research — uncurl port, mask sources given missing `data_2d_semantics`, label transfer from accumulated clouds.

### Phase 3: Evaluation Harness on Synthetic + Placeholder Geometry
**Rationale:** If the metrics are wrong, every mesh result is meaningless. Synthetic tests are cheap and catch sign/axis/rectification/denominator errors before reconstruction time is spent. This phase needs only the ray schema, not a real mesh.
**Delivers:** range/CD/F/P/R/miss/bucket/timing metrics; schema-versioned `metrics.json` + CSV; analytic plane/sphere/cube tests; identity-prediction = zero error; empty scene = 100 % miss; negative-control harness (5 cm extrinsic perturbation / camera swap degrades monotonically); `docs/protocol.md` frozen.
**Addresses:** T14–T18, T20; D4 and D12 scaffolding.
**Avoids:** C13, C16, C17, C18, G5 definition drift; prevents "numbers look OK" reasoning.
**Gate:** synthetic/analytic tests pass; metric definitions frozen and hashed; negative control is sensitive.
**Research flag:** standard patterns — formulas are fully specified in FEATURES §1 and PITFALLS G5.

### Phase 4: Mesh Reconstruction Route (Long Pole)
**Rationale:** Highest-variance, highest-cost item; start the spike as soon as Phase 1 calibration and a short image window exist, in parallel with Phases 2–3. It must not block the harness. This is where the route decision gets validated on real data, not opinion.
**Delivers:** COLMAP known-pose model triangulated with frozen poses (diff `images.txt` before/after); OpenMVS `InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → RefineMesh` on a short window, then the full window; Open3D/trimesh cleanup; `MeshArtifact` in SEG + manifest; route/config decision note; optional route A/B if the fallback is already installed.
**Addresses:** T8, T9, T10; D6.
**Avoids:** C7, C8, C9, C10, C11, C21, M5, M6; prevents Poisson ballooning (visibility-aware Delaunay), long triangles (`--remove-spurious 4 --close-holes 30 --smooth 2`), and any similarity fit to metrics.
**Gate:** **G3** — bbox 0.8–1.2, camera-outside test, component/degenerate checks, median range error < 1 m before tuning.
**Research flag:** needs research — real-model COLMAP↔OpenMVS handoff, `image_id` consistency, ROI auto-crop, mask flags (`-m` means mask-path in Densify but mesh-file in RefineMesh), parameter tuning on dev frames only.

### Phase 5: CPU BVH Cast + Embree Cross-Check
**Rationale:** The locked P1 backend and the correctness reference for P2. A hand-written BVH is exactly where winding/epsilon/edge bugs hide, so the independent oracle must run before the suite.
**Delivers:** flat-array BVH build/serialize + manifest; two-sided Möller–Trumbore with `t_min/t_max`, deterministic tie-break, `prim_id`/barycentrics; `RayBatch`/`HitBatch`/`SceneBackend`; per-frame `PredictionBundle` (miss = `+inf`); throughput/RSS instrumentation; 1e4–1e5-ray Embree comparison report.
**Addresses:** T11–T13, T19.
**Avoids:** C12, C13, M3, M4, A13 (no micro-optimisation before correctness).
**Gate:** **G4** — analytic tests, edge cases, Embree agreement ≥ 99.9 % and `max|Δt| ≤ 1e-4 m`, `t` verified metric.
**Research flag:** standard patterns — Möller–Trumbore and Open3D `RaycastingScene` are fully documented.

### Phase 6: Full Evaluation, Artifacts, Plots & Report
**Rationale:** Only now is a real number meaningful. This phase runs the full frozen ray set against the accepted mesh, applies the full suite honestly, and produces the deliverables.
**Delivers:** full 0–50 m run with buckets, per-frame/per-camera/per-class tables, miss and coverage, signed bias, CD-L1/L2 + acc/comp, F@5/20/50 with true-Euclidean τ, throughput/memory/BVH table, error CDF and miss map, range-RMSE+recall-vs-bucket plot, CD–FPS scaffold point, `results/report.md` with protocol, hashes, hardware, limitations, and the degeneracy/S penalty caveats.
**Addresses:** T15, T16, T20; D1–D3, D8, D11, D12 as time allows.
**Avoids:** C16, C17, C18, C19, A10, A14, A15; no post-hoc τ/bucket selection; F@5 reported but not gated.
**Gate:** **G5** — all metrics and buckets present with counts; every accuracy row paired with coverage; formulas/units stated; sanity floors checked (MedAE < 2 cm ⇒ leakage; > 2 m or miss > 70 % ⇒ calibration bug); expected ordinal structure (thin-class recall ≤ 0.6 × planar; monotone error B1→B3; RMSE > MAE > MedAE; F non-decreasing in τ).
**Research flag:** standard, but the degeneracy/incidence audit needs careful write-up rather than new research.

### Phase 7: Reproducibility Hardening & Claims Audit
**Rationale:** "Re-runnable, not one-off" is a stated constraint; multi-threaded MVS is not bit-reproducible, so the reproducibility boundary must be defined explicitly (bitwise cast/eval from the archived mesh; approximate recon).
**Delivers:** clean-room rerun from `rays + mesh + manifest` with no reconstruction step; complete `run_manifest.json`; pinned threads/versions; `docs/repro.md`; claims audit (no GT leakage, no post-hoc selection, provenance recorded, licence notes for AGPL OpenMVS); archived `results/`.
**Addresses:** T17; C19/C20 audit.
**Avoids:** C20, C21, A4, A5, A11.
**Gate:** **G5 re-run** — metrics byte-identical (or ≤1e-9) from frozen artifacts; manifest complete; every reported number traceable to a run directory.
**Research flag:** standard patterns.

### Phase Ordering Rationale

- **Data/calibration preflight is Phase 1** because every metric is a downstream function of the transform chain and the dataset files; a missing official `data_poses/` or a substituted KITTI-raw calibration invalidates the milestone rather than merely degrading it.
- **The evaluation harness precedes reconstruction** because synthetic/analytic tests are the only cheap way to prove the metric is not inert; building it after the mesh invites interpreting a wrong number as a mesh property.
- **The mesh route is the long pole but starts early (parallel spike after Phase 1)**; the harness and ray set do not need a real mesh, so the critical path is dataset → calibration → rays → harness → cast → report, with reconstruction as a parallel track.
- **CPU BVH + Embree cross-check is a separate phase** because a backend bug and a mesh bug produce the same symptom; the oracle separates them before the expensive full run.
- **Full metrics/artifacts and reproducibility are separate phases** so that "we have numbers" and "the numbers are defensible and regenerable" are independently gated.
- This grouping avoids the dominant pitfalls by construction: frame/calibration errors die at G1, GT contamination at G2, mesh scale/presentation at G3, caster bugs at G4, and reporting/leakage at G5.

### Research Flags

**Needs deeper research during planning:**
- **Phase 1** — KITTI-360 acquisition specifics and reconciliation of the local split/repackaged tree with the official layout (which sequence, exact file subset, licence, whether raw `image_XX/data` must be rectified); also confirm the official download endpoints and sizes against the 44 GB `/data-20T` budget.
- **Phase 2** — NumPy port of `curlVelodyneData` (sweep-start timestamp convention, optional constant time offset), dynamic-mask sources given the missing `data_2d_semantics`, and label transfer from accumulated clouds; occlusion-flag thresholds (k pixels, 3σ).
- **Phase 4** — the real-model COLMAP→OpenMVS handoff (layout, PINHOLE, database `image_id` consistency), ROI auto-crop behaviour on a street corridor, the `-m` short-option asymmetry, and OpenMVS parameter tuning on dev frames only. This is the highest-uncertainty phase.

**Standard patterns (skip `/gsd-research-phase`):**
- **Phase 3** — metric formulas are fully specified (FEATURES §1, PITFALLS G5); synthetic tests are routine.
- **Phase 5** — Möller–Trumbore, flat BVH traversal and Open3D `RaycastingScene` semantics are well documented and locally verifiable.
- **Phase 7** — manifest/hash/caching patterns are established; the work is discipline, not discovery.

---

## Confidence Assessment

| Area | Confidence | Notes |
|---|---|---|
| Stack | **HIGH** (versions/tooling/local fit) / **MEDIUM** (mesh quality ranking) | OpenMVS 2.4.0 bundle executed on this host; COLMAP 4.0.4 CPU capabilities verified; `CreateStructure` absence verified. Mesh-quality ranking is extrapolated from the UseGeo UAV evaluation, not a KITTI-360 head-to-head. |
| Features | **MEDIUM** | Protocol/metric definitions are HIGH (official docs + LiDAR4D/GS-LiDAR source). Absolute error bands are LOW–MEDIUM and must be replaced by measured values. Metric non-independence is a first-principles (MEDIUM) result. |
| Architecture | **HIGH** for frames/contracts; **MEDIUM** for the route and P2 seams | Frame algebra, artifact contracts and stage boundaries follow official KITTI-360/COLMAP docs. Route choice and local dataset variant are flagged. |
| Pitfalls | **HIGH** for KITTI-360/COLMAP/OpenMVS/Open3D conventions; **MEDIUM** for derived degeneracy/occlusion/incidence analysis | The derived analyses (CD≈range, F≈error CDF, occlusion fraction, grazing amplification) are first-principles and must be validated empirically in Phases 2/5/6. |

**Overall confidence:** **MEDIUM** (mechanically HIGH, empirically unvalidated). The plan is safe because every uncertain item is behind a measurable gate.

### Gaps to Address

- **Dataset readiness (blocking):** local tree lacks official `calibration/`, `data_poses/`, `data_2d_raw/`, `data_2d_semantics/`; images are under `image_00/data`, not `data_rect`. → Resolve in Phase 1 with a `check_dataset` report and a minimal download; add an explicit rectification step or acquire `data_rect`; never substitute KITTI-raw calibration. Escalate immediately if official poses are unobtainable.
- **COLMAP→OpenMVS handoff on a real triangulated model:** only a 2-image PINHOLE TXT model was verified. → Phase 4 smoke test: layout, PINHOLE, `image_id` consistency, then a short-window end-to-end before the full window.
- **Mesh quality / triangle budget on KITTI-360 driving scenes:** unknown; expected to degrade beyond 40 m due to weak forward parallax. → Measure in Phase 4, record `#triangles`, treat rising far-range RMSE as a baseline property, report buckets.
- **Absolute error bands:** all §5.4 magnitudes are expectations. → Replace with measured values after the first end-to-end run, then freeze acceptance thresholds (never tune the mesh to pass them).
- **Motion-distortion floor:** whether KITTI-360 `.bin` scans are already compensated is unverified. → Test with signed error vs azimuth and the straight-wall BEV "banana" check; report raw-vs-curled ablation.
- **Evaluable GT coverage and value of `image_01`:** unknown until measured. → Phase 2 coverage report (`N_valid/N_raw`); add image_01 if it materially improves coverage; keep fisheye deferred.
- **BVH implementation choice (Numba vs C++/pybind11):** performance is unmeasured. → Start with Numba; micro-benchmark 1e5 rays; only escalate if throughput blocks the phase. Keep the flat-array schema and `SceneBackend` seam.
- **Metric-convention drift:** CD-L1 Euclidean vs coordinate-L1, `t_min`, `t_max`, bucket edges. → Freeze all of it in `docs/protocol.md` in Phase 3 and hash it; any change regenerates the ray-set hash and invalidates comparisons.
- **Occlusion-flag and incidence corrections:** thresholds are first-principles. → Tune on a dev frame subset, freeze, and report both flagged and full variants; report point-to-plane error alongside range error.
- **Licensing:** OpenMVS is AGPL-3.0 (fine for coursework, matters if binaries are redistributed); KITTI-360 derived assets have non-commercial terms. → Note in the report; keep data/meshes out of git.
- **Disk budget:** 59 GB `/` + 44 GB `/data-20T`, with `data_3d_raw` already 136 GB. → Download only the minimal subset; clean intermediate `scene_dense.mvs` and depth maps; archive the final mesh and ray set.

---

## Sources

### Primary (HIGH confidence)
- KITTI-360 official documentation — folder structure, calibration semantics, poses, sensor axes, **2 perspective + 2 fisheye**, subsampled `poses.txt`, 1408×376 images, HDL-64E: https://www.cvlibs.net/datasets/kitti-360/documentation.php
- `kitti360Scripts` — `loadCalibration.py`, `project.py`, `kitti360Viewer3DRaw.py` (`TrVeloToCam`, `R_rect`, `curlVelodyneData`), `helpers/labels.py`, `helpers/project.py`: https://github.com/autonomousvision/kitti360Scripts
- COLMAP documentation + FAQ + CLI — `images.txt` world→camera, PINHOLE, known-pose reconstruction (empty `points3D` + `point_triangulator`), dense/MVS/meshing, no-GPU functionality: https://colmap.github.io/
- OpenMVS v2.4.0 release + Usage/Building wiki — bundle contents, `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, `RefineMesh`, mask/ROI flags; **documented `CreateStructure` is absent from the shipped bundle (verified locally)**: https://github.com/cdcseacave/openMVS/wiki
- Open3D 0.20.0 — `RaycastingScene.cast_rays` (`t_hit=inf`, `INVALID_ID` on miss), Python 3.10–3.14, manylinux_2_35: https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html
- PyPI / conda-forge metadata — pinned versions and `requires_python` for numpy, scipy, numba, open3d-cpu, trimesh, pycolmap, gsplat, pytest, ruff, psutil, tqdm; COLMAP 4.2.1 cpu/cuda variants
- Local verification (2026-10-30) — OpenMVS 2.4.0 bundle runs flat without `LD_LIBRARY_PATH`, no `CreateStructure`, `InterfaceCOLMAP` imports a 2-image PINHOLE TXT model; COLMAP 4.0.4 CPU (no `patch_match_stereo`); no `nvcc`; RTX 4090 48 GB; 32 cores/125 GB; `/data-20T/kitti` has `data_3d_raw` + `data_3d_bboxes` but no official calibration/poses/2D raw/2D semantics; disk 59 GB / 44 GB free
- LiDAR4D (`utils/metrics.py`) and GS-LiDAR (`chamfer/fscore.py`) source — **F-score threshold applied to squared distance (≈22.4 cm for "F@5 cm")**; CD mean-squared, summed
- Möller & Trumbore, "Fast, Minimum Storage Ray/Triangle Intersection", JGT 2(1):21–28, 1997
- Project-internal: `README.md` §0.5, §1.3, §1.4, §6, §7, §8; `.planning/PROJECT.md`; `docs/proposal_V2.tex`

### Secondary (MEDIUM confidence)
- Hermann et al., "Depth estimation and 3D reconstruction from UAV-borne imagery: Evaluation on the UseGeo dataset", ISPRS Open J. Photogramm. Remote Sens. 13:100065, 2024 — COLMAP best depth, **OpenMVS lowest mesh error** (UAV, not KITTI-360): https://doi.org/10.1016/j.ophoto.2024.100065
- LiDAR-NVS paper tables — GS-LiDAR KITTI-360 static CD 0.2382 / F 0.9055 / RMSE 5.8925 / MedAE 0.0198 (units ambiguous); LiDAR4D CD definition; LiDAR-RT 30 FPS @64×2650 (GPU); FGGS-LiDAR object-level CD 0.025 / F 0.979
- KITTI raw readme — Velodyne continuous sweep and "untwisting": https://www.cvlibs.net/datasets/kitti/raw_data.php
- Liao et al., KITTI-360 TPAMI 2022 — dataset design, poses, semantics: https://arxiv.org/abs/2109.08224
- MVS on urban scenes — textureless roads, moving-object contamination: arXiv:2207.08439, arXiv:2306.11368
- CD/F definition variability — Empir3D (arXiv:2306.03660), "Rethinking Metrics … for 3D Point Cloud Generation" (arXiv:2511.05308)
- Poisson density-trimming practice — low-density skin, quantile trim, ballooning (practitioner source, consistent with Open3D docs)
- 2DGS/OMeGa conventions — mean unsquared CD-L1, 5 cm threshold: arXiv:2509.24308
- AlignMiF — KITTI-360 LiDAR CD/F magnitudes: arXiv:2402.17483

### Tertiary (LOW — validate before use)
- Waymo static-scene surface-reconstruction aggregator — OpenMVS as a competitive explicit-MVS baseline (sota2.com)
- Expected absolute error bands in FEATURES §5.4 (miss 10–40 %, MedAE 0–20 m 0.05–0.30 m, 20–40 m 0.2–0.8 m, MAE 0.3–1.0 m, CD-L1 0.2–1.0 m, R@0.2 0.35–0.75, F@0.2 0.35–0.70) — inferred from MVS geometry, **not measured on this pipeline**
- Derived first-principles analyses — GT-ray CD/F degeneracy, camera-origin occlusion, grazing-incidence amplification, GT waterfall (PITFALLS); **validate empirically in Phases 2/5/6**

---

*Research completed: 2026-10-30*
*Ready for roadmap: yes — 7 suggested phases, 5 hard gates, 3 phases flagged for deeper research (1, 2, 4)*
