#pragma once

#include <raylidar/export.h>

#include <raylidar/rays/rays.h>

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

namespace raylidar::eval
{
/// Frozen evaluation thresholds in metres. True Euclidean distance is used at
/// every tau; `kPrimaryTauM` is the headline F-score threshold.
inline constexpr std::array<double, 3> kTausM = {0.05, 0.20, 0.50};
inline constexpr double kPrimaryTauM = 0.20;

/// One evaluated ray.
///
/// `t_pred` is finite for a hit and `+inf` for a miss. Every metric field in
/// this module is `std::optional<double>`; an undefined metric is `std::nullopt`
/// (serialized as JSON `null`), never `0`, `NaN`, or `inf`.
struct RAYLIDAR_EXPORT RaySample
{
    std::uint64_t ray_id = 0;
    std::uint32_t frame = 0;
    std::uint16_t camera = 0;
    rays::Bucket bucket = rays::Bucket::B0_20;
    double t_gt = 0.0;
    double t_min = rays::kRayTMinM;
    double t_max = rays::kRayMaxRangeM;
    double t_pred = std::numeric_limits<double>::infinity();
    Eigen::Vector3d origin_W = Eigen::Vector3d::Zero();
    Eigen::Vector3d dir_W = Eigen::Vector3d::Zero();
    bool occluded = false;
};

/// Range statistics over hits only (`e = t_pred - t_gt`, metres).
struct RAYLIDAR_EXPORT RangeMetrics
{
    std::optional<double> mae;
    std::optional<double> rmse;
    std::optional<double> medae;
    std::optional<double> bias;
    std::optional<double> p90;
    std::optional<double> p95;
    std::uint64_t n_hits = 0;
};

/// Miss accounting over valid rays (`n_misses / n_valid`, `n_hits / n_valid`).
struct RAYLIDAR_EXPORT MissMetrics
{
    std::optional<double> miss_rate;
    std::optional<double> ray_recall;
    std::uint64_t n_total = 0;
    std::uint64_t n_hits = 0;
    std::uint64_t n_misses = 0;
};

/// Precision/recall/F-score at one tau (true Euclidean distance).
struct RAYLIDAR_EXPORT PrfMetrics
{
    std::optional<double> precision;
    std::optional<double> recall;
    std::optional<double> f_score;
    std::uint64_t tp_pred = 0;
    std::uint64_t tp_gt = 0;
    std::uint64_t n_pred = 0;
    std::uint64_t n_gt = 0;
    bool greedy_truncated = false;
};

/// Chamfer statistics. CD-L1 uses unsquared Euclidean distance; CD-L2 uses
/// squared distance; both use the symmetric 1/2 average.
struct RAYLIDAR_EXPORT ChamferMetrics
{
    std::optional<double> cd_l1;
    std::optional<double> cd_l2;
    std::optional<double> accuracy_l1;
    std::optional<double> completeness_l1;
    std::optional<double> accuracy_l2;
    std::optional<double> completeness_l2;
    std::uint64_t n_pred = 0;
    std::uint64_t n_gt = 0;
};

/// Denominator counts for coverage. `raw_bucket` is indexed by `rays::Bucket`.
struct RAYLIDAR_EXPORT RawCounts
{
    std::uint64_t raw_total = 0;
    std::array<std::uint64_t, 4> raw_bucket = {0U, 0U, 0U, 0U};
};

/// One aggregate cell: accuracy (range/miss) paired with its coverage.
struct RAYLIDAR_EXPORT CellMetrics
{
    std::uint64_t n_raw = 0;
    std::uint64_t n_valid = 0;
    std::uint64_t n_hits = 0;
    std::uint64_t n_misses = 0;
    std::optional<double> coverage;
    RangeMetrics range;
    MissMetrics miss;
};
} // namespace raylidar::eval
