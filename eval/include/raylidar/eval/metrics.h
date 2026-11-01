#pragma once

#include <raylidar/export.h>

#include <raylidar/eval/types.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace raylidar::eval
{
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
} // namespace raylidar::eval
