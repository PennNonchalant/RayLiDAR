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
struct RAYLIDAR_EXPORT FrameRawCounts
{
    std::uint32_t frame = 0;
    RawCounts raw;
};

/// One frame's primary cell plus its four buckets.
struct RAYLIDAR_EXPORT FrameMetrics
{
    std::uint32_t frame = 0;
    CellMetrics primary;
    std::array<CellMetrics, 4> buckets{};
};

/// One camera's primary result (coverage is `nullopt`: per-camera raw counts are
/// not part of the frozen inputs).
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
};

/// Aggregate one set of samples into a cell, pairing accuracy with coverage.
///
/// `coverage = n_valid / raw.raw_total`, or `std::nullopt` when `raw_total == 0`.
RAYLIDAR_EXPORT CellMetrics AggregateCell(std::span<const RaySample> samples,
                                          const RawCounts &raw);

/// Per-frame cells, in the order of `rawByFrame`; each primary cell pools only
/// the primary buckets (`0_20`, `20_40`, `40_50`) while `buckets` keeps all four.
RAYLIDAR_EXPORT std::vector<FrameMetrics>
AggregatePerFrame(std::span<const RaySample> samples,
                  const std::vector<FrameRawCounts> &rawByFrame);

/// Per-camera primary cells for the cameras present, sorted by camera id.
RAYLIDAR_EXPORT std::vector<CameraMetrics>
AggregatePerCamera(std::span<const RaySample> samples);

/// Macro mean/sample-std over the per-frame primary cells.
RAYLIDAR_EXPORT MacroSummary SummarizeMacro(std::span<const FrameMetrics> frames);
} // namespace raylidar::eval
