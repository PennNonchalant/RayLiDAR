# Hybrid Mesh–Gaussian LiDAR Simulation

## What This Is

A GPU-accelerated LiDAR simulation system that reconstructs an outdoor scene from multi-view RGB images and camera poses, then casts LiDAR rays against the reconstructed scene to synthesize point clouds comparable to real LiDAR scans. It is a GPU Programming course project whose long-term thesis is that a hybrid mesh + Gaussian representation can deliver LiDAR fidelity with far fewer primitives and much higher throughput than Gaussian-only pipelines.

## Core Value

Produce a geometrically credible, quantitatively evaluated LiDAR simulation from an RGB reconstruction, so that a hybrid mesh + Gaussian speed/fidelity argument can be measured instead of asserted.

## Current Milestone: v1.0 KITTI-360 Mesh-Only Baseline & GT-Ray Evaluation

**Goal:** Reconstruct one KITTI-360 segment as a mesh-only scene and validate it with a reproducible GT-defined-ray LiDAR evaluation protocol.

**Target features:**
- KITTI-360 preprocessing: auto-pick a low-dynamic segment, use provided poses/calib, mask dynamic objects, extract static LiDAR GT (0–50 m, 20 m buckets)
- Open-source mesh reconstruction (route chosen by research: COLMAP MVS / OpenMVS / learning-based); 2DGS→TSDF→MC deferred as prep for later representation work
- GT-defined ray casting on CPU BVH (Möller–Trumbore), with miss = inf and recall/miss accounting
- Full eval harness: range MAE/RMSE/MedAE, CD L1/L2, F@5/20/50 cm, precision/recall, miss rate, distance buckets, FPS/memory
- Deliverables: runnable pipeline, metrics JSON/CSV, plots, short report

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — no implementation code exists; ship to validate)

### Active

<!-- Current scope. Building toward these. -->

- [ ] Pick and preprocess one low-dynamic KITTI-360 segment with provided metric poses/calib
- [ ] Produce a mesh-only reconstruction via a research-selected open-source pipeline
- [ ] Implement the GT-defined-ray evaluation protocol with CPU BVH ray casting
- [ ] Compute the full README metric suite with distance buckets and throughput
- [ ] Deliver a reproducible pipeline, metrics artifacts, plots, and a short report

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Gaussian-only baselines (3DGS / 2DGS curves) — deferred; this milestone isolates the mesh + evaluation path
- 2DGS→TSDF→Marching Cubes reconstruction — deferred until after the primary mesh-only task; kept as prep for later representation work
- P2 routed hybrid casting and GPU acceleration — next milestone; P1 is a CPU correctness/reference baseline
- P3 true hybrid representation (`M + residual 2DGS`) and reconstruction optimization — later milestone
- Dynamic scene graph — static outdoor scenes only; dynamic objects are masked out
- Intensity / ray-drop / multi-return physical modeling — geometry-only scope
- Differentiable mesh–Gaussian joint optimization, diffusion inpainting, BRDF inversion — out of scope per README
- Multi-sequence / multi-dataset generalization — single segment first

## Context

- Source of truth for the research plan: `README.md` §0.5 (V2 execution plan) and `docs/proposal_V2.pdf` / `docs/proposal_V2.tex`; §1–§12 retain V1 background.
- KITTI-360 supplies metric camera poses, calibration, multi-camera imagery, 64-line LiDAR, and semantic masks — so no COLMAP pose estimation is required.
- The evaluation protocol is GT-defined rays: for each static LiDAR GT point, transform into the camera frame, cast from the camera origin along the GT direction, and compare predicted range against GT range. Misses are `inf`, excluded from range error but counted in recall / miss-rate.
- Primary evaluation range is 0–50 m with 0–20 / 20–40 / 40–60 m buckets; predicted and GT points are compared in a unified (LiDAR) frame.
- The environment targets Python 3.11, PyTorch 2.x, CUDA 12.x; Open3D is acceptable for offline TSDF/mesh I/O.
- The project currently contains documentation only — no source tree yet.

## Constraints

- **Dataset**: KITTI-360, one low-dynamic segment — keeps preprocessing and reconstruction tractable.
- **Poses**: Use dataset-provided metric poses/calib; do not run COLMAP SfM or Umeyama alignment.
- **Representation**: Mesh-only for v1.0 — no Gaussian curves in this milestone.
- **Compute backend**: CPU BVH reference for P1 — GPU optimization is deliberately deferred to P2.
- **Evaluation**: GT-defined rays, 0–50 m primary range, fixed ray set and fixed hardware for comparisons.
- **Deliverable**: Reproducible code + metrics + plots + report — results must be re-runnable, not one-off.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mesh-only before Gaussian baselines | Isolates reconstruction + evaluation correctness; reduces risk | — Pending |
| GT-defined rays as primary protocol | Eliminates LiDAR–camera parallax bias and "GT no-return" ambiguity | — Pending |
| CPU BVH (Möller–Trumbore) first | Correctness reference before GPU optimization; cheap to validate | — Pending |
| Open-source mesh route selected by research | COLMAP MVS vs OpenMVS vs learning-based each trade maturity for quality | — Pending |
| 2DGS→TSDF→MC deferred | Prepares the later representation change without blocking P1 | — Pending |
| Full README metric suite | Avoids cherry-picking; enables honest comparison with published work | — Pending |
| Single low-dynamic segment | Keeps scope achievable for the course timeline | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-30 after starting milestone v1.0*
