#pragma once

#include <raylidar/export.h>

#include <raylidar/eval/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace raylidar::eval
{
/// Deterministic cap on greedy candidates considered per prediction (T-03-09).
inline constexpr std::size_t kMaxGreedyCandidatesPerPoint = 4096;

/// Reject malformed samples before any mean is computed.
///
/// Rejects non-finite `t_gt`/`t_min`/`t_max`, a non-unit `dir_W`, a `t_gt`
/// outside `[t_min, t_max]`, a camera outside `{0, 1}`, a bucket that disagrees
/// with `BucketOf(t_gt)`, and a finite `t_pred <= 0` (or a `NaN` `t_pred`).
/// Throws `std::invalid_argument`.
RAYLIDAR_EXPORT void ValidateSamples(std::span<const RaySample> samples);

/// Range MAE/RMSE/MedAE/signed bias/P90/P95 over hits only.
///
/// `P90`/`P95` use the nearest-rank rule `sorted[ceil(p * n) - 1]`; `MedAE`
/// averages the two middle values for an even count. With no hits every field is
/// `std::nullopt`.
RAYLIDAR_EXPORT RangeMetrics ComputeRangeMetrics(std::span<const RaySample> samples);

/// Miss rate (`n_misses / n_valid`) and ray recall (`n_hits / n_valid`).
///
/// `n_raw` is accepted for interface symmetry; coverage is reported by the
/// aggregate cells, which know the denominator. With `n_valid == 0` both values
/// are `std::nullopt`.
RAYLIDAR_EXPORT MissMetrics
ComputeMissMetrics(std::span<const RaySample> samples,
                   std::optional<std::uint64_t> n_raw = std::nullopt);

/// Per-bucket cells indexed by `rays::Bucket`.
///
/// Assigns each sample to its `RaySample::bucket` (after validating it against
/// `BucketOf(t_gt)`), and includes all four buckets even when empty. Coverage is
/// `n_valid / raw_bucket[b]`, or `std::nullopt` when the bucket denominator is
/// unavailable/zero.
RAYLIDAR_EXPORT std::array<CellMetrics, 4>
ComputeBucketMetrics(std::span<const RaySample> samples, const RawCounts &raw = {});

/// GT points `origin_W + t_gt * dir_W` for every valid sample (misses included:
/// the GT point exists independently of whether the simulator hit it).
RAYLIDAR_EXPORT std::vector<Eigen::Vector3d>
GtPoints(std::span<const RaySample> samples);

/// Predicted points `origin_W + t_pred * dir_W` for hits only; misses are
/// omitted.
RAYLIDAR_EXPORT std::vector<Eigen::Vector3d>
PredictedPoints(std::span<const RaySample> samples);

/// Symmetric Chamfer distance with the frozen 1/2 convention.
///
/// `accuracy_l1 = mean_p d(p, Q)`, `completeness_l1 = mean_q d(q, P)`, and
/// `cd_l1 = 0.5 * (accuracy_l1 + completeness_l1)`; `_l2` fields use squared
/// nearest distances. Distances are true Euclidean metres, never coordinate L1.
/// If either cloud is empty every field is `std::nullopt` but the counts stay.
RAYLIDAR_EXPORT ChamferMetrics
ComputeChamferMetrics(std::span<const Eigen::Vector3d> predicted,
                      std::span<const Eigen::Vector3d> gt);

/// Precision/recall/F-score at `tau` (true Euclidean metres).
///
/// `one_to_one == false` (primary) counts each prediction whose nearest GT is
/// within `tau` and each GT whose nearest prediction is within `tau`.
/// `one_to_one == true` runs deterministic one-to-one greedy matching over
/// radius-search candidates sorted by `(distance, pred_index, gt_index)`, capped
/// at `kMaxGreedyCandidatesPerPoint` per prediction.
///
/// `precision`/`recall` are `std::nullopt` when the corresponding cloud is
/// empty; `f_score` is `std::nullopt` if either is, else `2PR/(P+R)` (0 when
/// `P + R == 0`). A non-finite or negative `tau` throws `std::invalid_argument`.
RAYLIDAR_EXPORT PrfMetrics ComputePrfMetrics(std::span<const Eigen::Vector3d> predicted,
                                             std::span<const Eigen::Vector3d> gt, double tau,
                                             bool one_to_one);
} // namespace raylidar::eval
