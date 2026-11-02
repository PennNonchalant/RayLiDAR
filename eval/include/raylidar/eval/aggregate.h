#pragma once

#include <raylidar/export.h>

#include <raylidar/eval/types.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace raylidar::eval
{
/// Per-frame raw denominators (all rays before the FOV/range/validity filters).
///
/// `raw` pools both cameras; `per_camera[0]`/`per_camera[1]` carry the same
/// accounting for image_00/image_01 so per-camera and per-row coverage have a
/// defined denominator.
struct RAYLIDAR_EXPORT FrameRawCounts
{
    std::uint32_t frame = 0;
    RawCounts raw;
    std::array<RawCounts, 2> per_camera{};
};

/// One frame's primary cell plus its four buckets.
struct RAYLIDAR_EXPORT FrameMetrics
{
    std::uint32_t frame = 0;
    CellMetrics primary;
    std::array<CellMetrics, 4> buckets{};
};

/// One camera's primary result. The raw-aware overload below sets coverage from
/// the per-camera raw denominator; the one-argument overload leaves it
/// `nullopt` because no denominator is available.
struct RAYLIDAR_EXPORT CameraMetrics
{
    std::uint16_t camera = 0;
    CellMetrics primary;
};

/// Per-frame macro mean +/- sample standard deviation (`n - 1`).
///
/// A statistic's mean is defined when at least one frame defines it; its std is
/// defined only when at least two frames define it.
struct RAYLIDAR_EXPORT MacroSummary
{
    std::uint64_t n_frames_total = 0;
    std::uint64_t n_frames_with_range = 0;
    std::uint64_t n_frames_without_range = 0;

    std::optional<double> mae_mean;
    std::optional<double> mae_std;
    std::optional<double> rmse_mean;
    std::optional<double> rmse_std;
    std::optional<double> medae_mean;
    std::optional<double> medae_std;
    std::optional<double> bias_mean;
    std::optional<double> bias_std;
    std::optional<double> p90_mean;
    std::optional<double> p90_std;
    std::optional<double> p95_mean;
    std::optional<double> p95_std;
    std::optional<double> miss_rate_mean;
    std::optional<double> miss_rate_std;
    std::optional<double> ray_recall_mean;
    std::optional<double> ray_recall_std;
    std::optional<double> coverage_mean;
    std::optional<double> coverage_std;

    // Per-frame macro of the primary-tau (0.20 m) nearest-neighbour P/R/F.
    std::optional<double> precision_mean;
    std::optional<double> precision_std;
    std::optional<double> recall_mean;
    std::optional<double> recall_std;
    std::optional<double> f_score_mean;
    std::optional<double> f_score_std;
};

/// Aggregate one set of samples into a cell, pairing accuracy with coverage.
///
/// `coverage = n_valid / raw.raw_total`, or `std::nullopt` when `raw_total == 0`.
RAYLIDAR_EXPORT CellMetrics AggregateCell(std::span<const RaySample> samples, const RawCounts &raw);

/// Per-frame cells, in the order of `rawByFrame`; each primary cell pools only
/// the primary buckets (`0_20`, `20_40`, `40_50`) while `buckets` keeps all four.
RAYLIDAR_EXPORT std::vector<FrameMetrics> AggregatePerFrame(std::span<const RaySample> samples,
                                                            const std::vector<FrameRawCounts> &rawByFrame);

/// Per-camera primary cells for the cameras present, sorted by camera id.
///
/// Without raw denominators every camera's coverage is the explicit `nullopt`.
RAYLIDAR_EXPORT std::vector<CameraMetrics> AggregatePerCamera(std::span<const RaySample> samples);

/// Per-camera primary cells with coverage `n_valid / raw_total` for the camera.
///
/// The primary `n_raw` is the sum of `entry.per_camera[camera].raw_total` over
/// frames, i.e. every raw cast by that camera (including misses and
/// out-of-range hits). Coverage is `nullopt` only when that sum is zero.
RAYLIDAR_EXPORT std::vector<CameraMetrics> AggregatePerCamera(std::span<const RaySample> samples,
                                                              const std::vector<FrameRawCounts> &rawByFrame);

/// Macro mean/sample-std over the per-frame primary cells.
RAYLIDAR_EXPORT MacroSummary SummarizeMacro(std::span<const FrameMetrics> frames);

/// P/R/F at one tau for both matching policies.
struct RAYLIDAR_EXPORT TauPrf
{
    double tau = 0.0;
    PrfMetrics nearest;
    PrfMetrics greedy;
};

/// One frame x camera x bucket row.
///
/// `cell.n_raw` is the per-camera pre-filter GT population for the bucket
/// (`FrameRawCounts.per_camera[camera].raw_bucket[bucket]`), so a populated row
/// has a defined `cell.coverage` and an empty bucket has `n_raw == 0` with an
/// explicit `nullopt`. A real zero is never used as a null sentinel.
struct RAYLIDAR_EXPORT RowMetrics
{
    std::uint32_t frame = 0;
    std::uint16_t camera = 0;
    rays::Bucket bucket = rays::Bucket::B0_20;
    CellMetrics cell;
    std::array<TauPrf, 3> prf{};
    ChamferMetrics cd;
};

/// I/O-free end-to-end report.
struct RAYLIDAR_EXPORT EvalReport
{
    CellMetrics micro_primary;
    CellMetrics micro_secondary;
    // Pooled micro P/R/F and CD, so the micro blocks serialize with the same
    // cell/prf/cd shape as the per-row metrics. A default-constructed entry
    // (`tau == 0`) means the micro block was not computed.
    std::array<TauPrf, 3> micro_primary_prf{};
    ChamferMetrics micro_primary_cd;
    std::array<TauPrf, 3> micro_secondary_prf{};
    ChamferMetrics micro_secondary_cd;
    std::vector<FrameMetrics> per_frame;
    std::vector<CameraMetrics> per_camera;
    MacroSummary macro;
    std::vector<RowMetrics> rows;
};

/// Evaluate every metric over `samples` and the supplied raw denominators.
///
/// The frame set is the union of `rawByFrame` frames and frames present in
/// `samples`; rows cover every `(frame, camera in {0,1}, bucket in 0..3)`
/// combination, including empty cells. Primary aggregates exclude `B50_60`.
/// Throws `std::invalid_argument` on any sample rejected by `ValidateSamples`.
RAYLIDAR_EXPORT EvalReport EvaluateAll(std::span<const RaySample> samples,
                                       const std::vector<FrameRawCounts> &rawByFrame);
} // namespace raylidar::eval
