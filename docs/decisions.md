# Phase 1 — Pre-registered Evaluation Decisions

**Status:** PRE-REGISTERED
**Frozen at (UTC):** 2026-10-31T05:35:58Z
**Hash sidecar:** /data-20T/hybrid-lidar-sim/artifacts/01/decisions.sha256
**Machine-readable source:** configs/protocol.yaml

## Scope

These decisions are frozen before any reconstruction, before any mesh is produced, and before any evaluation number exists. Segment selection uses the criteria below, never a metric outcome.

## Segment Selection

- Candidate: `2013_05_28_drive_0000_sync` (provisional for calibration; final selection in Phase 2).
- Criteria: lowest median dynamic-pixel fraction over a 120-frame window; at least 120 contiguous frames present in image_00, image_01, velodyne_points, and poses; residential or suburban route preferred.
- Tie-break: fewer dynamic object instances, then longer contiguous pose coverage.
- Forbidden: selecting by reconstruction error, metric value, or visual mesh quality.

## Frame Window

- 60–120 consecutive frames.
- At least 30 evaluated frames.
- At least 20 000 valid rays per frame.
- Policy: the first contiguous block satisfying the all-stream intersection; no per-frame cherry-picking.

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
