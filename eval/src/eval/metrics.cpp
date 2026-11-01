#include <raylidar/eval/metrics.h>

#include "stats_internal.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace raylidar::eval
{
namespace
{
constexpr double kDirectionTolerance = 1.0e-9;

[[noreturn]] void Reject(const char *message)
{
    throw std::invalid_argument(message);
}
} // namespace

void ValidateSamples(const std::span<const RaySample> samples)
{
    for (const auto &sample : samples)
    {
        if (!std::isfinite(sample.t_gt) || !std::isfinite(sample.t_min) ||
            !std::isfinite(sample.t_max))
        {
            Reject("ValidateSamples: t_gt/t_min/t_max must be finite");
        }
        if (!(sample.t_min > 0.0) || !(sample.t_max >= sample.t_min))
        {
            Reject("ValidateSamples: t_min must be positive and t_max >= t_min");
        }
        if (sample.t_gt < sample.t_min || sample.t_gt > sample.t_max)
        {
            Reject("ValidateSamples: t_gt must lie inside [t_min, t_max]");
        }
        const double directionNorm = sample.dir_W.norm();
        if (!std::isfinite(directionNorm) ||
            std::abs(directionNorm - 1.0) > kDirectionTolerance)
        {
            Reject("ValidateSamples: dir_W must be a finite unit vector");
        }
        if (sample.camera > 1U)
        {
            Reject("ValidateSamples: camera must be 0 or 1");
        }
        if (sample.bucket != rays::BucketOf(sample.t_gt))
        {
            Reject("ValidateSamples: bucket disagrees with BucketOf(t_gt)");
        }
        if (!std::isfinite(sample.t_pred))
        {
            if (sample.t_pred != std::numeric_limits<double>::infinity())
            {
                Reject("ValidateSamples: a non-finite t_pred must be +inf (a miss)");
            }
        }
        else if (sample.t_pred <= 0.0)
        {
            Reject("ValidateSamples: a finite t_pred must be positive");
        }
    }
}

RangeMetrics ComputeRangeMetrics(const std::span<const RaySample> samples)
{
    std::vector<double> errors;
    errors.reserve(samples.size());
    for (const auto &sample : samples)
    {
        if (std::isfinite(sample.t_pred))
        {
            errors.push_back(sample.t_pred - sample.t_gt);
        }
    }
    return detail::RangeFromErrors(std::move(errors));
}

MissMetrics ComputeMissMetrics(const std::span<const RaySample> samples,
                               const std::optional<std::uint64_t> n_raw)
{
    (void)n_raw; // coverage lives on the aggregate cells that own the denominator
    std::uint64_t hits = 0;
    for (const auto &sample : samples)
    {
        if (std::isfinite(sample.t_pred))
        {
            ++hits;
        }
    }
    return detail::MissFromCounts(samples.size(), hits);
}

std::array<CellMetrics, 4> ComputeBucketMetrics(const std::span<const RaySample> samples,
                                                const RawCounts &raw)
{
    std::array<std::vector<double>, 4> errors;
    std::array<std::uint64_t, 4> valid = {0U, 0U, 0U, 0U};
    std::array<std::uint64_t, 4> hits = {0U, 0U, 0U, 0U};

    for (const auto &sample : samples)
    {
        const rays::Bucket expected = rays::BucketOf(sample.t_gt);
        if (sample.bucket != expected)
        {
            Reject("ComputeBucketMetrics: bucket disagrees with BucketOf(t_gt)");
        }
        const auto index = static_cast<std::size_t>(sample.bucket);
        if (index > 3U)
        {
            Reject("ComputeBucketMetrics: bucket outside 0-3");
        }
        ++valid[index];
        if (std::isfinite(sample.t_pred))
        {
            ++hits[index];
            errors[index].push_back(sample.t_pred - sample.t_gt);
        }
    }

    std::array<CellMetrics, 4> cells;
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        cells[index] = detail::MakeCell(raw.raw_bucket[index], valid[index], hits[index],
                                        std::move(errors[index]));
    }
    return cells;
}
} // namespace raylidar::eval
