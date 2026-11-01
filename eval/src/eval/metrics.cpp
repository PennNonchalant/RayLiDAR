#include <raylidar/eval/metrics.h>

#include <raylidar/eval/knn.h>

#include "stats_internal.h"

#include <algorithm>
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

std::vector<Eigen::Vector3d> GtPoints(const std::span<const RaySample> samples)
{
    std::vector<Eigen::Vector3d> points;
    points.reserve(samples.size());
    for (const auto &sample : samples)
    {
        points.push_back(sample.origin_W + sample.dir_W * sample.t_gt);
    }
    return points;
}

std::vector<Eigen::Vector3d> PredictedPoints(const std::span<const RaySample> samples)
{
    std::vector<Eigen::Vector3d> points;
    points.reserve(samples.size());
    for (const auto &sample : samples)
    {
        if (std::isfinite(sample.t_pred))
        {
            points.push_back(sample.origin_W + sample.dir_W * sample.t_pred);
        }
    }
    return points;
}

ChamferMetrics ComputeChamferMetrics(const std::span<const Eigen::Vector3d> predicted,
                                     const std::span<const Eigen::Vector3d> gt)
{
    ChamferMetrics metrics;
    metrics.n_pred = predicted.size();
    metrics.n_gt = gt.size();
    if (predicted.empty() || gt.empty())
    {
        return metrics; // all fields stay std::nullopt, counts are preserved
    }

    const PointCloudIndex predictedIndex{predicted};
    const PointCloudIndex gtIndex{gt};

    double accuracyL1 = 0.0;
    double accuracyL2 = 0.0;
    for (const auto &point : predicted)
    {
        const double distance = gtIndex.Nearest(point).second;
        accuracyL1 += distance;
        accuracyL2 += distance * distance;
    }

    double completenessL1 = 0.0;
    double completenessL2 = 0.0;
    for (const auto &point : gt)
    {
        const double distance = predictedIndex.Nearest(point).second;
        completenessL1 += distance;
        completenessL2 += distance * distance;
    }

    const double nPred = static_cast<double>(predicted.size());
    const double nGt = static_cast<double>(gt.size());
    accuracyL1 /= nPred;
    accuracyL2 /= nPred;
    completenessL1 /= nGt;
    completenessL2 /= nGt;

    metrics.accuracy_l1 = accuracyL1;
    metrics.completeness_l1 = completenessL1;
    metrics.cd_l1 = 0.5 * (accuracyL1 + completenessL1);
    metrics.accuracy_l2 = accuracyL2;
    metrics.completeness_l2 = completenessL2;
    metrics.cd_l2 = 0.5 * (accuracyL2 + completenessL2);
    return metrics;
}

PrfMetrics ComputePrfMetrics(const std::span<const Eigen::Vector3d> predicted,
                             const std::span<const Eigen::Vector3d> gt, const double tau,
                             const bool one_to_one)
{
    if (!std::isfinite(tau) || tau < 0.0)
    {
        throw std::invalid_argument("ComputePrfMetrics: tau must be finite and non-negative");
    }

    PrfMetrics metrics;
    metrics.n_pred = predicted.size();
    metrics.n_gt = gt.size();

    if (one_to_one)
    {
        std::uint64_t matched = 0;
        if (!predicted.empty() && !gt.empty())
        {
            struct Candidate
            {
                double distance;
                std::size_t predIndex;
                std::size_t gtIndex;
            };

            const PointCloudIndex gtIndex{gt};
            std::vector<Candidate> candidates;
            std::vector<std::pair<std::size_t, double>> matches;
            for (std::size_t predIndex = 0; predIndex < predicted.size(); ++predIndex)
            {
                gtIndex.RadiusSearch(predicted[predIndex], tau, matches);
                const std::size_t kept =
                    std::min(matches.size(), kMaxGreedyCandidatesPerPoint);
                if (matches.size() > kept)
                {
                    metrics.greedy_truncated = true;
                }
                for (std::size_t k = 0; k < kept; ++k)
                {
                    candidates.push_back(
                        Candidate{matches[k].second, predIndex, matches[k].first});
                }
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate &lhs, const Candidate &rhs) {
                          if (lhs.distance != rhs.distance)
                          {
                              return lhs.distance < rhs.distance;
                          }
                          if (lhs.predIndex != rhs.predIndex)
                          {
                              return lhs.predIndex < rhs.predIndex;
                          }
                          return lhs.gtIndex < rhs.gtIndex;
                      });

            std::vector<char> predUsed(predicted.size(), 0);
            std::vector<char> gtUsed(gt.size(), 0);
            for (const auto &candidate : candidates)
            {
                if (predUsed[candidate.predIndex] != 0 ||
                    gtUsed[candidate.gtIndex] != 0)
                {
                    continue;
                }
                predUsed[candidate.predIndex] = 1;
                gtUsed[candidate.gtIndex] = 1;
                ++matched;
            }
        }

        metrics.tp_pred = matched;
        metrics.tp_gt = matched;
        if (!predicted.empty())
        {
            metrics.precision = static_cast<double>(matched) /
                                static_cast<double>(predicted.size());
        }
        if (!gt.empty())
        {
            metrics.recall = static_cast<double>(matched) / static_cast<double>(gt.size());
        }
    }
    else
    {
        if (!predicted.empty())
        {
            if (gt.empty())
            {
                metrics.precision = 0.0;
            }
            else
            {
                const PointCloudIndex gtIndex{gt};
                std::uint64_t truePositives = 0;
                for (const auto &point : predicted)
                {
                    if (gtIndex.Nearest(point).second <= tau)
                    {
                        ++truePositives;
                    }
                }
                metrics.tp_pred = truePositives;
                metrics.precision = static_cast<double>(truePositives) /
                                    static_cast<double>(predicted.size());
            }
        }

        if (!gt.empty())
        {
            if (predicted.empty())
            {
                metrics.recall = 0.0;
            }
            else
            {
                const PointCloudIndex predictedIndex{predicted};
                std::uint64_t truePositives = 0;
                for (const auto &point : gt)
                {
                    if (predictedIndex.Nearest(point).second <= tau)
                    {
                        ++truePositives;
                    }
                }
                metrics.tp_gt = truePositives;
                metrics.recall =
                    static_cast<double>(truePositives) / static_cast<double>(gt.size());
            }
        }
    }

    if (metrics.precision.has_value() && metrics.recall.has_value())
    {
        const double precision = *metrics.precision;
        const double recall = *metrics.recall;
        metrics.f_score = (precision + recall > 0.0)
                              ? 2.0 * precision * recall / (precision + recall)
                              : 0.0;
    }
    return metrics;
}
} // namespace raylidar::eval
