#pragma once

// Internal helpers shared by the metric and aggregation translation units.
// Not installed and not part of the public `raylidar/eval` API.

#include <raylidar/eval/types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace raylidar::eval::detail
{
/// Nearest-rank percentile on an ascending-sorted vector: `sorted[ceil(p*n)-1]`.
inline double PercentileNearestRank(const std::vector<double> &sorted, const double percentile)
{
    const auto n = sorted.size();
    auto rank = static_cast<std::size_t>(std::ceil(percentile * static_cast<double>(n)));
    if (rank == 0U)
    {
        rank = 1U;
    }
    if (rank > n)
    {
        rank = n;
    }
    return sorted[rank - 1U];
}

/// Range statistics from a vector of signed errors `e = t_pred - t_gt`.
inline RangeMetrics RangeFromErrors(std::vector<double> errors)
{
    RangeMetrics metrics;
    metrics.n_hits = errors.size();
    if (errors.empty())
    {
        return metrics; // every statistic stays std::nullopt
    }

    double sumSigned = 0.0;
    double sumSquared = 0.0;
    double sumAbsolute = 0.0;
    std::vector<double> absoluteErrors;
    absoluteErrors.reserve(errors.size());
    for (const double error : errors)
    {
        sumSigned += error;
        sumSquared += error * error;
        const double absolute = std::abs(error);
        sumAbsolute += absolute;
        absoluteErrors.push_back(absolute);
    }

    const double count = static_cast<double>(errors.size());
    metrics.mae = sumAbsolute / count;
    metrics.rmse = std::sqrt(sumSquared / count);
    metrics.bias = sumSigned / count;

    std::sort(absoluteErrors.begin(), absoluteErrors.end());
    const std::size_t middle = absoluteErrors.size() / 2U;
    if (absoluteErrors.size() % 2U == 0U)
    {
        metrics.medae = 0.5 * (absoluteErrors[middle - 1U] + absoluteErrors[middle]);
    }
    else
    {
        metrics.medae = absoluteErrors[middle];
    }
    metrics.p90 = PercentileNearestRank(absoluteErrors, 0.90);
    metrics.p95 = PercentileNearestRank(absoluteErrors, 0.95);
    return metrics;
}

inline MissMetrics MissFromCounts(const std::uint64_t nTotal, const std::uint64_t nHits)
{
    MissMetrics metrics;
    metrics.n_total = nTotal;
    metrics.n_hits = nHits;
    metrics.n_misses = nTotal - nHits;
    if (nTotal != 0U)
    {
        const double denominator = static_cast<double>(nTotal);
        metrics.miss_rate = static_cast<double>(metrics.n_misses) / denominator;
        metrics.ray_recall = static_cast<double>(nHits) / denominator;
    }
    return metrics;
}

/// Build a cell from pre-aggregated counts; coverage is null when `nRaw == 0`.
inline CellMetrics MakeCell(const std::uint64_t nRaw, const std::uint64_t nValid, const std::uint64_t nHits,
                            std::vector<double> errors)
{
    CellMetrics cell;
    cell.n_raw = nRaw;
    cell.n_valid = nValid;
    cell.n_hits = nHits;
    cell.n_misses = nValid - nHits;
    cell.range = RangeFromErrors(std::move(errors));
    cell.miss = MissFromCounts(nValid, nHits);
    if (nRaw != 0U)
    {
        cell.coverage = static_cast<double>(nValid) / static_cast<double>(nRaw);
    }
    return cell;
}
} // namespace raylidar::eval::detail
