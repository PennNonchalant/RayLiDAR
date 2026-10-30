# Requirements: Hybrid Mesh–Gaussian LiDAR Simulation

**Defined:** 2026-10-30
**Milestone:** v1.0 — KITTI-360 Mesh-Only Baseline & GT-Ray Evaluation
**Core Value:** Produce a geometrically credible, quantitatively evaluated LiDAR simulation from an RGB reconstruction, enabling a measured hybrid mesh + Gaussian speed/fidelity argument.

## v1.0 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Data Foundation

- [ ] **DATA-01**: Pipeline can run a dataset preflight that reports present/missing files per required KITTI-360 component and fails loudly if official calibration, `data_poses`, or `data_2d_semantics` are missing
- [ ] **DATA-02**: Pipeline can acquire the minimal official KITTI-360 subset (calibration, poses, `image_00` rectified images, semantics) for one segment without downloading unnecessary sequences
- [ ] **DATA-03**: Pipeline can select one low-dynamic segment using a documented dynamic-fraction score and emit a segment manifest (sequence ID, frame window, counts, rationale)
- [ ] **DATA-04**: Pipeline can compose KITTI-360 calibration and poses through one `T_AB` module with named transforms, round-trip assertions, `det(R)=+1`, and orthonormality checks
- [ ] **DATA-05**: Pipeline can intersect image, Velodyne, and pose streams, dropping frames missing any stream and logging every dropped frame
- [ ] **DATA-06**: Pipeline can motion-uncurl Velodyne scans using ego-motion and verify that signed range error versus azimuth is flat, not sinusoidal
- [ ] **DATA-07**: Pipeline can apply one dynamic/ego mask artifact to images, GT, projections, and evaluation, and report exclusion counts per reason
- [ ] **DATA-08**: Pipeline can produce a hashed per-frame static GT cache containing `p_L`, frame, camera, `r_gt`, class, and validity
- [ ] **DATA-09**: Pipeline can validate calibration with LiDAR-to-image overlays under 2 px at 10–30 m on at least 3 frames (G1), plus camera-height and transform round-trip checks

### Frozen GT Rays

- [ ] **RAY-01**: Pipeline can generate GT-defined rays with `p_C = T_CL p_L`, unit direction `d`, `r_gt = ||p_C||`, `t_min = 1e-4 m`, domain floor `r_min = 1.5 m`, and `t_max = 60 m`
- [ ] **RAY-02**: Pipeline can filter rays by camera FOV/visibility with a 5 px image margin and deterministic primary-camera assignment so each GT point contributes exactly one ray
- [ ] **RAY-03**: Pipeline can emit a frozen ray set containing `ray_id`, `dir_C`, `t_gt`, `p_L`, `bucket`, `occluded`, and `semantic_id`, with a SHA-256 manifest
- [ ] **RAY-04**: Pipeline can compute a GT-only occlusion flag and report both occluded-excluded and full variants without using reconstructed depth
- [ ] **RAY-05**: Pipeline can verify the GT waterfall `N_raw = N_valid + Σ N_dropped(reason)` exactly and fail on any mismatch
- [ ] **RAY-06**: Pipeline can freeze the ray set before reconstruction and detect any later mutation by hash comparison

### Mesh Reconstruction

- [ ] **RECON-01**: Pipeline can export a known-pose COLMAP model from KITTI poses (no SfM/mapper, no Umeyama) with frozen intrinsics and rectified `image_00` images
- [ ] **RECON-02**: Pipeline can run the OpenMVS chain `InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → RefineMesh` as a scripted, config-driven route
- [ ] **RECON-03**: Pipeline can ingest any mesh route behind a `MeshReconstructor` adapter into a canonical SEG-frame `MeshArtifact` in metres
- [ ] **RECON-04**: Pipeline can clean and decimate the mesh and report vertex/triangle counts, bbox, component count, and degenerate-triangle fraction
- [ ] **RECON-05**: Pipeline can enforce the G3 mesh gate: bbox ratio 0.8–1.2, camera-outside test, one component over 95%, degenerate triangles under 0.1%, and median range error under 1 m on a smoke frame, failing loudly instead of aligning
- [ ] **RECON-06**: Pipeline can record the route and config hash, and can switch to a second route through configuration for A/B comparison

### CPU Casting

- [ ] **CAST-01**: Pipeline can build a flat-array CPU BVH once per mesh and serialize it with a content hash
- [ ] **CAST-02**: Pipeline can cast two-sided Möller–Trumbore rays with `t_min`/`t_max`, epsilon `1e-6`, deterministic tie-break, returning `t`, `hit`, `prim_id`, barycentrics, and normal
- [ ] **CAST-03**: Pipeline can reconstruct `p_pred = T_CL⁻¹(r_pred · d)` and assert near-zero reprojection residual
- [ ] **CAST-04**: Pipeline can cross-check casts against Open3D `RaycastingScene` on at least 1e5 rays with ≥99.9% hit/miss agreement and `max|Δt| ≤ 1e-4 m`
- [ ] **CAST-05**: Pipeline can pass analytic plane, sphere, and cube tests under 1e-4 m plus edge cases (miss, parallel, inside, negative `t`, `t < t_min`)
- [ ] **CAST-06**: Pipeline can measure cast FPS, rays/s, BVH build time, and peak RSS together with fixed hardware and version metadata

### Evaluation & Artifacts

- [ ] **EVAL-01**: Pipeline can compute range MAE, RMSE, MedAE, signed bias, and P90/P95 over hits only, returning `null` when no hits exist
- [ ] **EVAL-02**: Pipeline can compute CD-L1 (unsquared Euclidean) and CD-L2 (squared), both directions averaged with the ½ convention, and report accuracy and completeness separately
- [ ] **EVAL-03**: Pipeline can compute precision, recall, and F-score at τ = 0.05/0.20/0.50 m using true Euclidean distance, with 0.20 m as primary τ
- [ ] **EVAL-04**: Pipeline can compute miss rate and ray recall, and report coverage `N_valid/N_raw` per frame and per bucket
- [ ] **EVAL-05**: Pipeline can bucket by `r_gt` into `[0,20)`, `[20,40)`, `[40,50]` primary plus `(50,60]` secondary, with counts in every bucket
- [ ] **EVAL-06**: Pipeline can emit schema-versioned `metrics.json` and a tidy CSV (frame × camera × bucket) including config, ray-set, and mesh hashes
- [ ] **EVAL-07**: Pipeline can report micro-average, per-frame macro ± std, and per-camera results, pairing every accuracy row with its coverage
- [ ] **EVAL-08**: Pipeline can prove harness correctness with synthetic tests: identity predictions yield zero error, an empty scene yields 100% miss, and no `inf`/`NaN` enters a mean
- [ ] **EVAL-09**: Pipeline can generate the required plots: range RMSE/MedAE and recall versus distance bucket; an error and miss map; and CD/F versus τ or an error-versus-runtime scaffold
- [ ] **EVAL-10**: Pipeline can stratify results by semantic class into thin (`pole`, `fence`, `vegetation`, signs) and planar (`road`, `building`, `wall`, `terrain`) groups and report recall/error per group
- [ ] **EVAL-11**: Pipeline can report signed bias, tail quantiles, a front-view error heatmap, and a spherical miss map
- [ ] **EVAL-12**: Pipeline can regenerate metrics byte-identically from frozen rays + mesh + manifest and emit a complete `run_manifest.json` (git SHA, dataset release, hashes, versions, thread count)
- [ ] **EVAL-13**: Pipeline can run end to end from a single documented entrypoint with a config file, seeds, and stage caching
- [ ] **EVAL-14**: Pipeline can produce a report stating the protocol, metric definitions and units, hardware, artifact hashes, and explicit limitations (no fisheye, squared-distance F-score caveat, metric degeneracy, forward-wedge coverage)
- [ ] **EVAL-15**: Pipeline can pre-register segment, frame window, τ values, buckets, and exclusions in `decisions.md` before reconstruction starts

## Future Requirements

Deferred to later milestones. Tracked but not in the v1.0 roadmap.

### Reconstruction & Casting

- **RECON-F01**: Pipeline can run a second mesh route and report A/B accuracy/runtime/thin-recall evidence
- **RECON-F02**: Pipeline can compare COLMAP dense + Poisson/Delaunay meshing as a quality upgrade

### Evaluation

- **EVAL-F01**: Pipeline can run a calibration negative-control test (5 cm perturbation or camera swap) and show monotone metric degradation
- **EVAL-F02**: Pipeline can audit CD/F collinearity with range error and report the correlation
- **EVAL-F03**: Pipeline can report point-to-mesh GT-to-mesh distance as an independent geometric check
- **EVAL-F04**: Pipeline can ablate camera-origin GT rays versus LiDAR-origin beam-pattern rays
- **EVAL-F05**: Pipeline can report bootstrap confidence intervals clustered by frame
- **EVAL-F06**: Pipeline can deterministically subsample rays while proving full-set agreement
- **EVAL-F07**: Pipeline can correlate mesh diagnostics (holes, triangle density, floaters) with recall

### Coverage

- **DATA-F01**: Pipeline can add `image_01` and merge per-camera results
- **DATA-F02**: Pipeline can support fisheye cameras and wider angular coverage
- **DATA-F03**: Pipeline can generalize to multiple segments or datasets

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| 3DGS / 2DGS Gaussian baselines | Deferred; this milestone isolates mesh-only correctness and evaluation validity |
| GPU / CUDA / OptiX casting | P1 is a correctness reference; GPU acceleration is P2 |
| 2DGS→TSDF→Marching Cubes as the mesh source | Deferred to a later ablation; would couple the baseline to the Gaussian stack |
| LiDAR-supervised mesh fitting or depth supervision | GT leakage; LiDAR is evaluation-only by design |
| Mesh built from accumulated GT points | Circular evaluation; at most an oracle diagnostic |
| Intensity / ray-drop / multi-return / beam divergence | Geometry-only scope |
| Dynamic objects / scene graph | Static-only milestone; dynamics are masked out |
| Multi-dataset / multi-segment generalization | Single segment first |
| PSNR / SSIM / LPIPS | Conflates geometry with appearance and invites unfair comparison |
| Per-frame or per-bucket mesh tuning | Destroys the reconstruction claim; parameters frozen before evaluation |
| Umeyama / ICP alignment before evaluation | Hides pose/scale bugs and constitutes leakage |
| BVH micro-optimization (SIMD, occupancy grids, LOD) | Premature; correctness first, speed is P2 |
| Pooled-only reporting | Hides far-range and thin-structure failures |
| Real-time claims from the CPU baseline | Misleading; CPU reference is expected to be slow |
| End-to-end training in this milestone | Requires the deferred Gaussian/NeRF stack and risks leakage |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 2 | Pending |
| DATA-06 | Phase 2 | Pending |
| DATA-07 | Phase 2 | Pending |
| DATA-08 | Phase 2 | Pending |
| DATA-09 | Phase 1 | Pending |
| RAY-01 | Phase 2 | Pending |
| RAY-02 | Phase 2 | Pending |
| RAY-03 | Phase 2 | Pending |
| RAY-04 | Phase 2 | Pending |
| RAY-05 | Phase 2 | Pending |
| RAY-06 | Phase 2 | Pending |
| RECON-01 | Phase 4 | Pending |
| RECON-02 | Phase 4 | Pending |
| RECON-03 | Phase 4 | Pending |
| RECON-04 | Phase 4 | Pending |
| RECON-05 | Phase 4 | Pending |
| RECON-06 | Phase 4 | Pending |
| CAST-01 | Phase 5 | Pending |
| CAST-02 | Phase 5 | Pending |
| CAST-03 | Phase 5 | Pending |
| CAST-04 | Phase 5 | Pending |
| CAST-05 | Phase 5 | Pending |
| CAST-06 | Phase 5 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |
| EVAL-05 | Phase 3 | Pending |
| EVAL-06 | Phase 3 | Pending |
| EVAL-07 | Phase 3 | Pending |
| EVAL-08 | Phase 3 | Pending |
| EVAL-09 | Phase 6 | Pending |
| EVAL-10 | Phase 6 | Pending |
| EVAL-11 | Phase 6 | Pending |
| EVAL-12 | Phase 7 | Pending |
| EVAL-13 | Phase 6 | Pending |
| EVAL-14 | Phase 6 | Pending |
| EVAL-15 | Phase 1 | Pending |

**Coverage:**
- v1.0 requirements: 42 total
- Mapped to phases: 42
- Unmapped: 0

**Per-phase counts:**
- Phase 1: 5 (DATA-01, DATA-02, DATA-04, DATA-09, EVAL-15)
- Phase 2: 11 (DATA-03, DATA-05, DATA-06, DATA-07, DATA-08, RAY-01–06)
- Phase 3: 8 (EVAL-01–08)
- Phase 4: 6 (RECON-01–06)
- Phase 5: 6 (CAST-01–06)
- Phase 6: 5 (EVAL-09, EVAL-10, EVAL-11, EVAL-13, EVAL-14)
- Phase 7: 1 (EVAL-12)

---
*Requirements defined: 2026-10-30*
*Last updated: 2026-10-30 after roadmap creation (42/42 mapped)*
