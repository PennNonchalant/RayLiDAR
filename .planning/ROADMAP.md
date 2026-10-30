# Roadmap: Hybrid Mesh–Gaussian LiDAR Simulation

## Overview

Milestone v1.0 delivers a **mesh-only KITTI-360 baseline** judged by a **frozen, GT-defined-ray evaluation protocol** on CPU. The journey is correctness-first and evaluation-first: acquire and validate the official dataset and calibration chain (G1), build a per-frame static GT cache and freeze the ray set before any reconstruction (G2), prove the metric harness on synthetic geometry, reconstruct a mesh with the research-selected COLMAP→OpenMVS route (G3), validate a hand-written CPU BVH against Embree (G4), then run the full metric suite and produce defensible artifacts, plots, and a report (G5), and finally prove the whole result is re-runnable from frozen artifacts. Reconstruction is the long pole and starts as a parallel track immediately after calibration is trustworthy; the GT ray set and evaluation harness never wait for it.

**Milestone goal:** Reconstruct one KITTI-360 segment as a mesh-only scene and validate it with a reproducible GT-defined-ray LiDAR evaluation protocol.

**Gates:** G1 calibration → G2 ray set → G3 mesh → G4 ray cast → G5 metrics/reproducibility. Each gate must pass before its numbers are interpreted.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work.
- Decimal phases (2.1, 2.2): Urgent insertions after planning (none planned).

- [ ] **Phase 1: Preflight, Acquisition & Calibration Foundation** - Prove the official dataset subset and the KITTI-360 transform chain before any metric exists (G1).
- [ ] **Phase 2: Data Foundation & Frozen GT Ray Set** - Turn raw frames into motion-compensated static GT and a SHA-256-frozen GT-defined ray set (G2).
- [ ] **Phase 3: Evaluation Harness & Metrics Contract** - Prove every metric on synthetic/placeholder geometry and freeze the metric definitions.
- [ ] **Phase 4: Mesh Reconstruction Route** - Produce a canonical metric mesh via the COLMAP→OpenMVS route and pass the G3 mesh gate.
- [ ] **Phase 5: CPU BVH Cast & Embree Cross-Check** - Cast the frozen rays with a verified Möller–Trumbore BVH (G4).
- [ ] **Phase 6: Full Evaluation, Artifacts, Plots & Report** - Run the full protocol and deliver metrics, plots, and an honest report (G5).
- [ ] **Phase 7: Reproducibility Hardening & Claims Audit** - Regenerate results from frozen artifacts and audit every claim (G5 re-run).

## Phase Details

### Phase 1: Preflight, Acquisition & Calibration Foundation
**Goal**: Users can prove that the official KITTI-360 inputs for one segment are present and that the single calibration/pose chain is metrically correct, so no later number inherits a silent frame or dataset error.
**Depends on**: Nothing (first phase).
**Requirements**: DATA-01, DATA-02, DATA-04, DATA-09, EVAL-15
**Success Criteria** (what must be TRUE):
  1. `check_dataset` reports present/missing files per required KITTI-360 component (calibration, `data_poses`, `data_2d_raw` 00/01, `data_2d_semantics`, `data_3d_raw`) and exits non-zero when official calibration, poses, or semantics are absent — no KITTI-raw substitution.
  2. The minimal official subset for one segment is acquired and recorded in a readiness manifest (sequence ID, files, sizes, release); unnecessary sequences are not downloaded.
  3. One `T_AB` transform module passes round-trip `< 1e-9`, `det(R) = +1`, `RᵀR = I` within `1e-6`, camera height 1.4–1.9 m, and `T_W_C0_rect` vs `cam0_to_world` within `1e-3`.
  4. LiDAR→image overlays align to `< 2 px` at 10–30 m on at least 3 frames (G1), with no raw-vs-rectified mixing.
  5. `decisions.md` pre-registers segment selection criteria, frame window, τ values, buckets, exclusions, and primary metric with a timestamp and hash before any reconstruction starts.
**Plans:** 8 plans

- [ ] 01-01-PLAN.md — Execution contract (container/data root/artifacts root/acquisition policy) and installable package + pytest scaffold
- [ ] 01-02-PLAN.md — Dataset layout adapter, preflight checker (incl. external KITTI-raw rejection), and fail-loud exit contract
- [ ] 01-03-PLAN.md — `T_AB` frames/calibration module and `check-calib` artifacts
- [ ] 01-04-PLAN.md — Protocol pre-registration (`decisions.md`, `protocol.yaml`, hash)
- [ ] 01-05-PLAN.md — Per-sequence official subset acquisition, link sources, and readiness manifest
- [ ] 01-06-PLAN.md — LiDAR→image projection, rectified-only overlays, and advisory edge score
- [ ] 01-07-PLAN.md — Real-sequence G1 evidence: transforms, camera height, overlays, expected-pair coverage
- [ ] 01-08-PLAN.md — Human overlay sign-off (full frame×camera coverage) and G1 gate decision

**Wave structure:** Wave 1: 01-01 · Wave 2: 01-02 · Wave 3: 01-03, 01-04, 01-05 · Wave 4: 01-06 · Wave 5: 01-07 · Wave 6: 01-08

**Notes**: Package scaffolding (`src/hls/`, configs, pytest harness, `docs/frames.md`) is delivered here; it is infrastructure, not a mapped requirement. This phase is a hard stop if official calibration/poses cannot be obtained. The container runtime, dataset root, and image-reuse policy are decision checkpoints in 01-01 because `AGENTS.md` forbids inventing them. Generated artifacts (reports, overlays, manifests, hashes) live under `/data-20T/hybrid-lidar-sim/artifacts/01/` per `AGENTS.md` §3 — the repo keeps code and `.planning/` summaries only. Acquisition uses per-component/per-sequence archives only; all-sequences archives require explicit user escalation (larger staging volume or stream-extract). G1 requires a measured entry for every observed (frame, camera) pair and rejects any non-`data_rect` image.

### Phase 2: Data Foundation & Frozen GT Ray Set
**Goal**: Users can generate, validate, and freeze the exact GT-defined ray set that every representation will be judged on, from motion-compensated per-frame LiDAR with dynamic and ego content removed.
**Depends on**: Phase 1.
**Requirements**: DATA-03, DATA-05, DATA-06, DATA-07, DATA-08, RAY-01, RAY-02, RAY-03, RAY-04, RAY-05, RAY-06
**Success Criteria** (what must be TRUE):
  1. One low-dynamic segment is selected with a documented dynamic-fraction score and a manifest (sequence ID, frame window, counts, rationale); image, Velodyne, and pose streams are intersected with every dropped frame logged.
  2. Velodyne scans are motion-uncurled from ego-motion and the signed range error versus azimuth is flat (not sinusoidal), with a raw-vs-curled ablation reported.
  3. One dynamic/ego mask artifact is applied consistently to images, GT, projections, and evaluation, with exclusion counts per reason and no isotropic near-field ego deletion.
  4. A hashed per-frame static GT cache exists (`p_L`, frame, camera, `r_gt`, class, validity), with per-frame point counts consistent with HDL-64E expectations.
  5. The frozen ray set satisfies G2: every ray has `z_cam > 0`, a 5 px image margin, `1.5 ≤ r_gt ≤ 60 m`, exactly one deterministic `camera_id`; the waterfall `N_raw = N_valid + Σ N_dropped(reason)` sums exactly; the SHA-256 is recorded and any later mutation is detected; the primary 0–50 m evaluation mask is stored separately from the 50–60 m secondary rows.
**Plans**: TBD
**Notes**: Freeze the ray set before Phase 4 begins. The GT-only occlusion flag (`occluded`) is built here and must never use reconstructed depth.

### Phase 3: Evaluation Harness & Metrics Contract
**Goal**: Users can trust the metric implementation before a real mesh exists, because every metric returns the mathematically expected value on synthetic and placeholder geometry and the definitions are frozen and hashed.
**Depends on**: Phase 2 (needs the frozen ray schema; no real mesh required).
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08
**Success Criteria** (what must be TRUE):
  1. Range MAE/RMSE/MedAE/signed bias/P90/P95, CD-L1/L2 with accuracy/completeness, precision/recall/F at 0.05/0.20/0.50 m true Euclidean, miss rate/ray recall/coverage, and `r_gt` buckets all return correct values; undefined metrics return `null`, never `0` or `NaN`.
  2. Synthetic proofs pass: identity predictions yield zero error, an empty scene yields 100% miss, and no `inf`/`NaN` enters any mean.
  3. Schema-versioned `metrics.json` and a tidy frame×camera×bucket CSV are emitted with config, ray-set, and mesh hash fields; micro-average, per-frame macro ± std, and per-camera results are computed, with every accuracy row carrying its coverage.
  4. `docs/protocol.md` freezes and hashes the formulas, units, `t_min = 1e-4 m`, `r_min = 1.5 m`, `t_max = 60 m`, τ values, bucket edges, CD averaging convention, and aggregation rules.
  5. A placeholder-mesh end-to-end run produces valid `metrics.json` and CSV without requiring the real reconstruction, and the CD/F non-independence and squared-distance F-score caveats are recorded in `docs/protocol.md`.
**Plans**: TBD
**Notes**: The CD/F non-independence caveat under GT-defined rays is documented here, not discovered at report time.

### Phase 4: Mesh Reconstruction Route
**Goal**: Users can produce a clean, canonical, metric mesh of the selected segment from RGB + provided poses using the research-selected open-source route, and prove it is in the right frame and scale without aligning it.
**Depends on**: Phase 1 (calibration + images). May start in parallel with Phases 2–3; do not wait for the full ray set.
**Requirements**: RECON-01, RECON-02, RECON-03, RECON-04, RECON-05, RECON-06
**Success Criteria** (what must be TRUE):
  1. A known-pose COLMAP model (PINHOLE, frozen intrinsics, no SfM/mapper, no Umeyama) is exported from KITTI poses and rectified `image_00` images; after `point_triangulator`, `images.txt` poses are unchanged by diff.
  2. The config-driven OpenMVS route `InterfaceCOLMAP → DensifyPointCloud → ReconstructMesh → RefineMesh` runs on a short smoke window and then the full window, producing a mesh on this CPU-only host.
  3. Any route is ingested through the `MeshReconstructor` adapter into a canonical SEG-frame `MeshArtifact` in metres, with route/config hash recorded and a second route switchable by configuration for A/B comparison.
  4. Cleanup and decimation report vertex/triangle counts, bbox, connected-component count, and degenerate-triangle fraction.
  5. The G3 gate passes: bbox ratio 0.8–1.2 vs GT extent, camera-outside test (forward hit rate 0.3–0.9, no near hits on multiple axes), one component over 95%, degenerate triangles under 0.1%, and median range error under 1 m on a smoke frame — failing loudly with no alignment fix.
**Plans**: TBD
**Notes**: Highest-uncertainty phase; needs a real-model COLMAP↔OpenMVS handoff spike. Tune parameters on dev frames only. Mesh provenance must remain "MVS from images + KITTI poses, no Gaussian model in the loop".

### Phase 5: CPU BVH Cast & Embree Cross-Check
**Goal**: Users can cast the frozen GT rays against the accepted mesh with a correct, deterministic CPU Möller–Trumbore BVH whose results are independently verified against Embree before any number is reported.
**Depends on**: Phase 2 (frozen rays), Phase 4 (accepted mesh).
**Requirements**: CAST-01, CAST-02, CAST-03, CAST-04, CAST-05, CAST-06
**Success Criteria** (what must be TRUE):
  1. A flat-array CPU BVH is built once per mesh, serialized, and identified by a content hash.
  2. Two-sided Möller–Trumbore casting honours `t_min`/`t_max`, epsilon `1e-6`, and a deterministic tie-break, returning `t`, `hit`, `prim_id`, barycentrics, and normal; `t` is verified as a metric distance for unit directions.
  3. Analytic plane, sphere, and cube tests pass under `1e-4 m`, plus miss, parallel, edge, inside, negative-`t`, and `t < t_min` edge cases.
  4. The Embree/Open3D `RaycastingScene` cross-check on at least `1e5` real rays shows ≥99.9% hit/miss agreement and `max|Δt| ≤ 1e-4 m`.
  5. `p_pred = T_CL⁻¹(r_pred · d)` reconstruction has near-zero reprojection residual, and cast FPS, rays/s, BVH build time, and peak RSS are measured with fixed hardware and version metadata.
**Plans**: TBD
**Notes**: No micro-optimization before correctness. Misses are `+inf` and never enter range means.

### Phase 6: Full Evaluation, Artifacts, Plots & Report
**Goal**: Users can run the full frozen-ray protocol against the accepted mesh and obtain a complete, honest, bucket-resolved evaluation package that states its protocol, units, hardware, hashes, and limitations.
**Depends on**: Phase 3 (metrics contract), Phase 4 (accepted mesh), Phase 5 (verified caster).
**Requirements**: EVAL-09, EVAL-10, EVAL-11, EVAL-13, EVAL-14
**Success Criteria** (what must be TRUE):
  1. The full ray set is cast and evaluated on the primary 0–50 m range with `[0,20)`, `[20,40)`, `[40,50]` primary and `(50,60]` secondary buckets, each with counts; no post-hoc bucket or τ selection.
  2. Results are stratified by semantic class into thin (`pole`, `fence`, `vegetation`, signs) and planar (`road`, `building`, `wall`, `terrain`) groups, with recall and error reported per group.
  3. Signed bias, tail quantiles, a front-view error heatmap, and a spherical miss map are produced alongside the required bucket RMSE/MedAE and recall plot, error/miss map, and CD/F-versus-τ or error-versus-runtime scaffold.
  4. A single documented entrypoint runs the pipeline end to end from a config file with seeds and stage caching.
  5. The report states the protocol, metric definitions and units, hardware, artifact hashes, and explicit limitations (no fisheye, squared-distance F-score comparison caveat, metric degeneracy under GT rays, forward-wedge coverage).
**Plans**: TBD
**Notes**: Sanity floors are bug detectors: MedAE < 2 cm suggests leakage; MedAE > 2 m or miss > 70% suggests calibration/frame error. Never tune the mesh to pass thresholds.

### Phase 7: Reproducibility Hardening & Claims Audit
**Goal**: Users can regenerate the reported metrics from frozen artifacts alone and trace every claim to a run directory, with the reproducibility boundary stated rather than implied.
**Depends on**: Phase 6.
**Requirements**: EVAL-12
**Success Criteria** (what must be TRUE):
  1. A clean-room rerun from frozen rays + archived mesh + manifest, with no reconstruction step, regenerates metrics byte-identically (or within `1e-9`).
  2. `run_manifest.json` is complete: git SHA, dataset release, calibration/frame-list/ray-set/mesh hashes, library and tool versions, thread count, seeds, and peak RSS.
  3. The claims audit finds no GT leakage, no post-hoc segment/frame/τ/bucket selection, recorded mesh provenance ("MVS from images + KITTI poses"), and explicit licence notes (OpenMVS AGPL, KITTI-360 non-commercial).
  4. `docs/repro.md` documents exact rerun commands and the boundary between approximate reconstruction and bitwise evaluation.
  5. Every reported number is traceable to a run directory, and the archived `results/` set is committed with hashes.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7. Phase 4 may start after Phase 1 in parallel with Phases 2–3.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Preflight, Acquisition & Calibration Foundation | 0/8 | Ready to execute | - |
| 2. Data Foundation & Frozen GT Ray Set | 0/— | Not started | - |
| 3. Evaluation Harness & Metrics Contract | 0/— | Not started | - |
| 4. Mesh Reconstruction Route | 0/— | Not started | - |
| 5. CPU BVH Cast & Embree Cross-Check | 0/— | Not started | - |
| 6. Full Evaluation, Artifacts, Plots & Report | 0/— | Not started | - |
| 7. Reproducibility Hardening & Claims Audit | 0/— | Not started | - |
