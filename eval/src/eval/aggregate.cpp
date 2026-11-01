#include <raylidar/eval/aggregate.h>

#include <raylidar/eval/metrics.h>

#include "stats_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace raylidar::eval
{
namespace
{
struct MeanStd
{
    std::optional<double> mean;
    std::optional<double> stddev;
};

MeanStd MeanAndSampleStd(const std::vector<double> &values)
{
    MeanStd result;
    if (values.empty())
    {
        return result;
    }

    double sum = 0.0;
    for (const double value : values)
    {
        sum += value;
    }
    const double mean = sum / static_cast<double>(values.size());
    result.mean = mean;

    if (values.size() >= 2U)
    {
        double squaredDeviation = 0.0;
        for (const double value : values)
        {
            const double delta = value - mean;
            squaredDeviation += delta * delta;
        }
        result.stddev =
            std::sqrt(squaredDeviation / static_cast<double>(values.size() - 1U));
    }
    return result;
}

template <typename Getter>
std::vector<double> CollectDefined(const std::span<const FrameMetrics> frames, Getter getter)
{
    std::vector<double> values;
    values.reserve(frames.size());
    for (const auto &frame : frames)
    {
        const std::optional<double> value = getter(frame);
        if (value.has_value())
        {
            values.push_back(*value);
        }
    }
    return values;
}

template <typename Getter>
void FillMeanStd(const std::span<const FrameMetrics> frames, Getter getter,
                 std::optional<double> &meanOut, std::optional<double> &stdOut)
{
    const MeanStd summary = MeanAndSampleStd(CollectDefined(frames, getter));
    meanOut = summary.mean;
    stdOut = summary.stddev;
}
} // namespace

CellMetrics AggregateCell(const std::span<const RaySample> samples, const RawCounts &raw)
{
    std::vector<double> errors;
    errors.reserve(samples.size());
    std::uint64_t hits = 0;
    for (const auto &sample : samples)
    {
        if (std::isfinite(sample.t_pred))
        {
            ++hits;
            errors.push_back(sample.t_pred - sample.t_gt);
        }
    }
    return detail::MakeCell(raw.raw_total, samples.size(), hits, std::move(errors));
}

std::vector<FrameMetrics> AggregatePerFrame(const std::span<const RaySample> samples,
                                            const std::vector<FrameRawCounts> &rawByFrame)
{
    std::vector<FrameMetrics> result;
    result.reserve(rawByFrame.size());

    for (const auto &entry : rawByFrame)
    {
        std::array<std::vector<double>, 4> errorsByBucket;
        std::array<std::uint64_t, 4> validByBucket = {0U, 0U, 0U, 0U};
        std::array<std::uint64_t, 4> hitsByBucket = {0U, 0U, 0U, 0U};

        for (const auto &sample : samples)
        {
            if (sample.frame != entry.frame)
            {
                continue;
            }
            const rays::Bucket expected = rays::BucketOf(sample.t_gt);
            if (sample.bucket != expected)
            {
                throw std::invalid_argument(
                    "AggregatePerFrame: bucket disagrees with BucketOf(t_gt)");
            }
            const auto index = static_cast<std::size_t>(sample.bucket);
            ++validByBucket[index];
            if (std::isfinite(sample.t_pred))
            {
                ++hitsByBucket[index];
                errorsByBucket[index].push_back(sample.t_pred - sample.t_gt);
            }
        }

        FrameMetrics frame;
        frame.frame = entry.frame;

        std::vector<double> primaryErrors;
        std::uint64_t primaryValid = 0;
        std::uint64_t primaryHits = 0;
        std::uint64_t primaryRaw = 0;
        for (std::size_t index = 0; index < 3U; ++index)
        {
            primaryValid += validByBucket[index];
            primaryHits += hitsByBucket[index];
            primaryRaw += entry.raw.raw_bucket[index];
            primaryErrors.insert(primaryErrors.end(), errorsByBucket[index].begin(),
                                 errorsByBucket[index].end());
        }
        frame.primary =
            detail::MakeCell(primaryRaw, primaryValid, primaryHits, std::move(primaryErrors));

        for (std::size_t index = 0; index < 4U; ++index)
        {
            frame.buckets[index] =
                detail::MakeCell(entry.raw.raw_bucket[index], validByBucket[index],
                                 hitsByBucket[index], std::move(errorsByBucket[index]));
        }

        result.push_back(std::move(frame));
    }
    return result;
}

std::vector<CameraMetrics> AggregatePerCamera(const std::span<const RaySample> samples)
{
    std::map<std::uint16_t, std::vector<double>> errorsByCamera;
    std::map<std::uint16_t, std::uint64_t> validByCamera;
    std::map<std::uint16_t, std::uint64_t> hitsByCamera;

    for (const auto &sample : samples)
    {
        if (sample.bucket == rays::Bucket::B50_60)
        {
            continue; // primary aggregation only
        }
        ++validByCamera[sample.camera];
        if (std::isfinite(sample.t_pred))
        {
            ++hitsByCamera[sample.camera];
            errorsByCamera[sample.camera].push_back(sample.t_pred - sample.t_gt);
        }
    }

    std::vector<CameraMetrics> result;
    result.reserve(validByCamera.size());
    for (auto &[camera, valid] : validByCamera)
    {
        CameraMetrics metrics;
        metrics.camera = camera;
        // Per-camera raw denominators are not part of the frozen inputs, so the
        // cell carries an explicit null coverage.
        metrics.primary =
            detail::MakeCell(0U, valid, hitsByCamera[camera], std::move(errorsByCamera[camera]));
        result.push_back(std::move(metrics));
    }
    return result;
}

MacroSummary SummarizeMacro(const std::span<const FrameMetrics> frames)
{
    MacroSummary summary;
    summary.n_frames_total = frames.size();
    for (const auto &frame : frames)
    {
        if (frame.primary.range.n_hits != 0U)
        {
            ++summary.n_frames_with_range;
        }
    }
    summary.n_frames_without_range =
        summary.n_frames_total - summary.n_frames_with_range;

    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.mae; },
                summary.mae_mean, summary.mae_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.rmse; },
                summary.rmse_mean, summary.rmse_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.medae; },
                summary.medae_mean, summary.medae_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.bias; },
                summary.bias_mean, summary.bias_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.p90; },
                summary.p90_mean, summary.p90_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.p95; },
                summary.p95_mean, summary.p95_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.miss.miss_rate; },
                summary.miss_rate_mean, summary.miss_rate_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.miss.ray_recall; },
                summary.ray_recall_mean, summary.ray_recall_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.coverage; },
                summary.coverage_mean, summary.coverage_std);
    return summary;
}
} // namespace raylidar::eval
