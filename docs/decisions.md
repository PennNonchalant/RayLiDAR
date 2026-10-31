# Phase 2 — Pre-registered Evaluation Decisions (revised)

**Status:** PRE-REGISTERED
**Frozen at (UTC):** 2026-10-31T22:09:59Z
**Hash sidecar:** /data-20T/hybrid-lidar-sim/artifacts/01/decisions.sha256
**Machine-readable source:** configs/protocol.yaml
**Run ID:** p2-60f-step10

## Change Log

- 2026-10-31 — pre-execution revision: segment window policy changed from a 120-frame median dynamic-pixel rule to a 60-frame sliding window with step 10, selected by the lowest point-weighted dynamic union (3D bboxes + 2D semantics) after ego removal; ties select the earliest window. No selection or evaluation result had been observed. Supersedes the 2026-10-31T05:35:58Z 120-frame freeze (archived sidecars decisions.sha256.prev / decisions.sha256.json.prev).

## Scope

These decisions are frozen before any selection score, reconstruction, mesh, or evaluation number exists. Segment selection uses the criteria below, never a metric outcome.

## Segment Selection

- Candidate: `2013_05_28_drive_0000_sync`.
- Criteria: lowest point-weighted dynamic fraction over a 60-frame window, evaluated at step 10 over the full sorted usable-frame index.
- Dynamic points: union of 3D dynamic bboxes and 2D semantic dynamic classes, after ego ROI removal; per-source and union counts are recorded per frame.
- Window score: micro average `Σ dynamic / Σ (dynamic + static)` over the window, never a mean of per-frame ratios.
- Tie-break: earliest window.
- Forbidden: selecting by reconstruction error, metric value, or visual mesh quality.

## Frame Window

- Exactly 60 consecutive usable frames.
- Step 10 between candidate window starts.
- At least 60 evaluated frames.
- At least 20 000 valid rays per frame.
- Policy: every 60-frame window at step 10 over the full sorted usable-frame index; no per-frame cherry-picking; the last partial window is not a candidate.

## Metrics and Thresholds

| Item | Frozen value |
|---|---|
| τ values | 0.05 m, 0.20 m (primary), 0.50 m — true Euclidean distance |
| Primary metric | Range RMSE over hits, micro-averaged over pooled rays |
| Paired reporting | Ray recall and coverage on every accuracy row |
| Secondary metrics | CD-L1 (metres), F-score at τ = 0.20 m |
| Aggregation | Micro (pooled) headline plus macro per-frame ± std |
| Distance buckets | [0,20), [20,40), [40,50] primary; (50,60] secondary |
| `r_min` | 1.5 m |
| `t_max` | 60 m |
| Image margin | 5 px |
| Occlusion | GT-only flag; flagged points reported separately, never using reconstructed depth |

## Exclusions

- Dynamic classes: car, truck, bus, motorcycle, bicycle, person, rider.
- Ego ROI in Velodyne coordinates: `|x| < 2.5 m`, `|y| < 1.2 m`, `-2.0 < z < 0.3 m`.
- Range below 1.5 m and above 60 m.
- Behind-camera or outside-image rays using the 5 px margin.

## Change Policy

Any change to these values requires a new hash, a documented rationale, and a new run ID. Thresholds are never relaxed to make a result pass.
