# Evaluation Protocol (Phase 3, `schema_version` 1)

This document is the single normative definition of every number the evaluation
harness reports. `metrics.json` and `metrics.csv` are generated only from the
formulas below; a reader that disagrees with this document is wrong.

If this document is edited, the protocol version and run ID MUST be bumped and every
recorded run MUST use a new run ID. A stale sidecar is a test failure
(`tests/eval_e2e/test_protocol_doc.cpp`).

## 1. Scope and version

- **`schema_version` is `1`.** Every `metrics.json` and every CSV row starts with
  this integer; a consumer MUST reject any other value.
- **Units.** All distances and coordinates are metres (`m`); squared distances are
  square metres (`m^2`); fractions and rates are dimensionless in `[0, 1]`.
- **Frames.** Metrics are computed in the KITTI-360 `W` pose frame: metres,
  right-handed, x forward, y left, z up. No metric re-expresses geometry in a
  camera frame.
- **Provenance.** Every run records SHA-256 values for the config, this protocol,
  the ray set, the mesh, and the hit set, plus the scene generator/seed and the
  predictor parameters. A number without those hashes is not reportable.

## 2. GT-defined rays, hits, and misses

A **GT-defined ray** is one per-ray record with:

- `origin_W` — the camera centre for that ray (not a single shared sensor origin),
- `dir_W` — a unit direction,
- `t_gt` — the true range from `origin_W`,
- `t_min = 1e-4 m` — the self-hit guard,
- `r_min = 1.5 m` — the minimum GT range that enters the ray set,
- `t_max = 60 m` — the cast bound.

A **hit** is a cast result with `t_min <= t_pred <= t_max`. A **miss** is
`t_pred = +inf`; a miss is never `0`, never a clipped number, and never `NaN`.
Misses never enter a range mean. Casting is two-sided: back faces are not culled,
because open MVS meshes are not consistently oriented.

## 3. Range metrics (hits only, metres)

Let `H = {i : t_pred(i) < inf}` and the signed error `e_i = t_pred(i) - t_gt(i)`
(positive means predicted farther). All six statistics use hits only:

| Metric | Definition | Units |
|---|---|---|
| MAE | `(1/|H|) * sum_{i in H} |e_i|` | m |
| RMSE | `sqrt((1/|H|) * sum_{i in H} e_i^2)` | m |
| MedAE | `median_{i in H}(|e_i|)`, averaging the two middle values for even `|H|` | m |
| Bias | `(1/|H|) * sum_{i in H} e_i` | m |
| P90 | the nearest-rank 90th percentile of `|e_i|` | m |
| P95 | the nearest-rank 95th percentile of `|e_i|` | m |

The nearest-rank rule is exact: with ascending-sorted errors `s[0..n-1]`, the
percentile value is `s[ceil(p * n) - 1]`, clamped to `[0, n-1]`. If `n_hits = 0`,
every range statistic is `null`.

## 4. Chamfer distance (metres)

For predicted points `P = {origin_W + t_pred * dir_W}` (hits only) and GT points
`Q = {origin_W + t_gt * dir_W}` (all valid rays), and `d(p, Q) = min_{q in Q} ||p - q||`
unsquared Euclidean:

```
CD-L1 = 1/2 * ( (1/|P|) * sum_{p in P} d(p, Q) + (1/|Q|) * sum_{q in Q} d(q, P) )
CD-L2 = 1/2 * ( (1/|P|) * sum_{p in P} d(p, Q)^2 + (1/|Q|) * sum_{q in Q} d(q, P)^2 )
```

- **CD-L1** uses unsquared Euclidean distance in metres; **CD-L2** uses squared
  distance in `m^2`. Coordinate L1 (`|dx| + |dy| + |dz|`) is never used.
- Both use the **1/2 symmetric average**, never a sum.
- **accuracy** is the `P -> Q` term; **completeness** is the `Q -> P` term. Both
  are reported separately in metres (L1) and `m^2` (L2).
- If either cloud is empty, every CD field is `null`; the counts stay.

## 5. Precision, recall, and F-score

With `tau` in metres and true Euclidean distance (never squared distance):

- **precision(tau) = |{p in P : d(p, Q) <= tau}| / |P|** (matched predictions),
- **recall(tau) = |{q in Q : d(q, P) <= tau}| / |Q|** (matched GT points),
- **F(tau) = 2 * P * R / (P + R)**, and `0` when `P + R = 0` while both are defined.

The reported taus are `0.05`, `0.20`, and `0.50` m; the **primary tau is 0.20 m**.
Precision is `null` when `|P| = 0`; recall is `null` when `|Q| = 0`; F is `null`
when either is.

- **Primary matching (nearest neighbour within tau).** A prediction counts when its
  nearest GT point is within tau; a GT point counts when its nearest prediction is
  within tau. This is the primary table.
- **Secondary matching (one-to-one greedy).** Candidates within tau are sorted by
  `(distance, prediction index, GT index)` and matched greedily; this is a
  sensitivity analysis against many-to-one double counting. At most `4096`
  candidates are considered per prediction, and `greedy_truncated` is set when the
  cap bites. The greedy result never replaces the primary result.

**Ray recall is not point recall.** Ray recall is "did any surface get hit";
point recall@tau is "was the right surface hit within tau". Both are labelled
separately in the output and never substituted for one another.

## 6. Miss and coverage

- **miss rate = n_misses / n_valid** and **ray recall = n_hits / n_valid** over
  valid rays (`n_valid = n_hits + n_misses`).
- **coverage = N_valid / N_raw** per frame: the denominator is every raw cast ray,
  including rays that hit nothing or hit outside `[r_min, t_max]`.
- **per-bucket coverage = N_valid_bucket / N_raw_bucket**, where `N_raw_bucket` is
  the pre-filter GT population whose `r_gt` has a defined bucket. When that
  denominator is unavailable or zero the value is `null`, never `0` and never `1`.
- Every accuracy row and every accuracy cell carries its coverage (or an explicit
  `null`). Accuracy without coverage is not reportable.

## 7. Distance buckets

Buckets are assigned by `r_gt`, never by `t_pred`:

| Bucket | Range (m) | Status |
|---|---|---|
| `0_20` | `[0,20)` | primary |
| `20_40` | `[20,40)` | primary |
| `40_50` | `[40,50]` | primary |
| `50_60` | `(50,60]` | secondary, reported separately |

Exactly `20` and exactly `40` belong to the following bucket; exactly `40` and
exactly `50` both belong to `40_50`, and `(50,60]` is secondary. Counts exist for
every bucket even when empty. **Primary micro aggregation excludes `50_60`** and
`50_60` is always reported on its own.

## 8. Aggregation

- **Micro over pooled primary rays is the headline.** It pools every primary ray
  across frames and cameras before any mean is taken, so a frame with many rays
  weighs more than a frame with few.
- **Macro** reports the per-frame mean and the sample standard deviation with
  `n - 1` in the denominator, together with `n_frames_total`,
  `n_frames_with_range`, and `n_frames_without_range`.
- **Per-camera** results are reported separately and are never silently pooled into
  the headline without saying so.
- Every accuracy cell is paired with its coverage; a mean is defined only when at
  least one frame defines it, and a standard deviation only when at least two do.

## 9. Null policy

Every undefined metric is JSON `null` and an empty CSV field. It is never `0`,
never `NaN`, never `Inf`, and never the string `nan` or `inf`. A validator
(`ValidateMetricsJson` / `ValidateMetricsCsv`) rejects a missing key, a malformed
SHA-256, a non-finite number, a missing or duplicate row, a count that does not sum
to `n_valid`, a row without coverage, and a header that differs from the frozen
column list. A zero is a real zero and must not be used as a null sentinel.

## 10. Output contract

`metrics.json` (`schema_version` 1) contains `format`, `protocol`, `inputs`,
`aggregates` (`micro_primary`, `micro_secondary`, `per_camera`, `macro`), the
complete `rows` grid, and `caveats`.

The tidy CSV has this fixed header, in this exact order:

```
schema_version,frame,camera,bucket,n_raw,n_valid,n_hits,n_misses,coverage,range_mae_m,range_rmse_m,range_medae_m,range_bias_m,range_p90_m,range_p95_m,miss_rate,ray_recall,precision_nn_tau_0.05,recall_nn_tau_0.05,f_score_nn_tau_0.05,precision_greedy_tau_0.05,recall_greedy_tau_0.05,f_score_greedy_tau_0.05,precision_nn_tau_0.20,recall_nn_tau_0.20,f_score_nn_tau_0.20,precision_greedy_tau_0.20,recall_greedy_tau_0.20,f_score_greedy_tau_0.20,precision_nn_tau_0.50,recall_nn_tau_0.50,f_score_nn_tau_0.50,precision_greedy_tau_0.50,recall_greedy_tau_0.50,f_score_greedy_tau_0.50,cd_l1_m,cd_l2_m2,cd_accuracy_l1_m,cd_completeness_l1_m,cd_accuracy_l2_m2,cd_completeness_l2_m2
```

There is exactly one row per frame x camera x bucket, ordered by frame, camera,
then bucket `0_20`, `20_40`, `40_50`, `50_60`. Empty buckets have count `0` and
empty metric fields. Counts are integers; defined metrics use round-trip decimal
formatting.

## 11. Caveats

**Caveat 1 — `cd_f_non_independent_under_gt_rays`.** Under GT-defined rays every
predicted point lies on the ray to its GT point, so CD-L1, CD accuracy, precision,
and F-score are largely a re-encoding of the range error. Completeness and recall
are largely a re-encoding of misses plus large range errors. CD and F are still
reported, but they are **not independent evidence** of geometric fidelity.

**Caveat 2 — `published_f_at_5cm_uses_squared_distance`.** The widely copied
LiDAR-NVS `fscore(..., threshold=0.05)` helper receives squared distances, so its
"F@5 cm" is about `sqrt(0.05) = 0.224` m, roughly 22.4 cm Euclidean. This project
reports a true-Euclidean F@0.05 m, which is expected to be substantially lower and
must never be compared to a published "F@5 cm" number without this caveat.

## 12. Change policy

Any change to a formula, a unit, a threshold, a bucket edge, an aggregation rule,
or the null policy requires all of: a new protocol version, a new
`schema_version` when the output shape changes, and a new run ID. Historical
numbers are never recomputed under a changed protocol without recording both.
