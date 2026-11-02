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
        result.stddev = std::sqrt(squaredDeviation / static_cast<double>(values.size() - 1U));
    }
    return result;
}

template <typename Getter> std::vector<double> CollectDefined(const std::span<const FrameMetrics> frames, Getter getter)
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
void FillMeanStd(const std::span<const FrameMetrics> frames, Getter getter, std::optional<double> &meanOut,
                 std::optional<double> &stdOut)
{
    const MeanStd summary = MeanAndSampleStd(CollectDefined(frames, getter));
    meanOut = summary.mean;
    stdOut = summary.stddev;
}

void AssignMeanStd(const std::vector<double> &values, std::optional<double> &meanOut, std::optional<double> &stdOut)
{
    const MeanStd summary = MeanAndSampleStd(values);
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
                throw std::invalid_argument("AggregatePerFrame: bucket disagrees with BucketOf(t_gt)");
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
        for (std::size_t index = 0; index < 3U; ++index)
        {
            primaryValid += validByBucket[index];
            primaryHits += hitsByBucket[index];
            primaryErrors.insert(primaryErrors.end(), errorsByBucket[index].begin(), errorsByBucket[index].end());
        }
        // The primary denominator is every raw cast in the frame (misses and
        // out-of-range hits included), never the surviving bucket-hit sum.
        frame.primary = detail::MakeCell(entry.raw.raw_total, primaryValid, primaryHits, std::move(primaryErrors));

        for (std::size_t index = 0; index < 4U; ++index)
        {
            frame.buckets[index] = detail::MakeCell(entry.raw.raw_bucket[index], validByBucket[index],
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
        // No denominator was supplied, so the cell carries an explicit null.
        metrics.primary = detail::MakeCell(0U, valid, hitsByCamera[camera], std::move(errorsByCamera[camera]));
        result.push_back(std::move(metrics));
    }
    return result;
}

std::vector<CameraMetrics> AggregatePerCamera(const std::span<const RaySample> samples,
                                              const std::vector<FrameRawCounts> &rawByFrame)
{
    std::map<std::uint16_t, std::uint64_t> rawByCamera;
    for (const auto &entry : rawByFrame)
    {
        for (std::size_t camera = 0; camera < entry.per_camera.size(); ++camera)
        {
            rawByCamera[static_cast<std::uint16_t>(camera)] += entry.per_camera[camera].raw_total;
        }
    }

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
        const std::uint64_t rawTotal = rawByCamera[camera];
        metrics.primary = detail::MakeCell(rawTotal, valid, hitsByCamera[camera], std::move(errorsByCamera[camera]));
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
    summary.n_frames_without_range = summary.n_frames_total - summary.n_frames_with_range;

    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.mae; }, summary.mae_mean, summary.mae_std);
    FillMeanStd(
        frames, [](const FrameMetrics &f) { return f.primary.range.rmse; }, summary.rmse_mean, summary.rmse_std);
    FillMeanStd(
        frames, [](const FrameMetrics &f) { return f.primary.range.medae; }, summary.medae_mean, summary.medae_std);
    FillMeanStd(
        frames, [](const FrameMetrics &f) { return f.primary.range.bias; }, summary.bias_mean, summary.bias_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.p90; }, summary.p90_mean, summary.p90_std);
    FillMeanStd(frames, [](const FrameMetrics &f) { return f.primary.range.p95; }, summary.p95_mean, summary.p95_std);
    FillMeanStd(
        frames, [](const FrameMetrics &f) { return f.primary.miss.miss_rate; }, summary.miss_rate_mean,
        summary.miss_rate_std);
    FillMeanStd(
        frames, [](const FrameMetrics &f) { return f.primary.miss.ray_recall; }, summary.ray_recall_mean,
        summary.ray_recall_std);
    FillMeanStd(
        frames, [](const FrameMetrics &f) { return f.primary.coverage; }, summary.coverage_mean, summary.coverage_std);
    return summary;
}

EvalReport EvaluateAll(const std::span<const RaySample> samples, const std::vector<FrameRawCounts> &rawByFrame)
{
    ValidateSamples(samples);

    // Frame set = supplied raw frames plus any frame observed in the samples.
    // Keep the whole FrameRawCounts so per-camera denominators survive.
    std::map<std::uint32_t, FrameRawCounts> frameRaw;
    for (const auto &entry : rawByFrame)
    {
        frameRaw[entry.frame] = entry;
    }
    for (const auto &sample : samples)
    {
        frameRaw.try_emplace(sample.frame, FrameRawCounts{sample.frame, RawCounts{}, {}});
    }

    std::vector<FrameRawCounts> completeRaw;
    completeRaw.reserve(frameRaw.size());
    for (const auto &[frame, entry] : frameRaw)
    {
        FrameRawCounts complete = entry;
        complete.frame = frame;
        completeRaw.push_back(std::move(complete));
    }

    std::vector<RaySample> primarySamples;
    std::vector<RaySample> secondarySamples;
    primarySamples.reserve(samples.size());
    for (const auto &sample : samples)
    {
        if (sample.bucket == rays::Bucket::B50_60)
        {
            secondarySamples.push_back(sample);
        }
        else
        {
            primarySamples.push_back(sample);
        }
    }

    RawCounts primaryRaw;
    RawCounts secondaryRaw;
    for (const auto &entry : completeRaw)
    {
        for (std::size_t index = 0; index < 3U; ++index)
        {
            primaryRaw.raw_bucket[index] += entry.raw.raw_bucket[index];
        }
        // Every raw cast is the primary denominator, including casts that hit
        // nothing, hit out of range, or landed in the secondary bucket.
        primaryRaw.raw_total += entry.raw.raw_total;
        secondaryRaw.raw_bucket[3] += entry.raw.raw_bucket[3];
        secondaryRaw.raw_total += entry.raw.raw_bucket[3];
    }

    EvalReport report;
    report.micro_primary = AggregateCell(primarySamples, primaryRaw);
    report.micro_secondary = AggregateCell(secondarySamples, secondaryRaw);

    // Pooled micro P/R/F and CD so the micro blocks serialize with the same
    // cell/prf/cd shape as the per-row metrics.
    {
        const auto predicted = PredictedPoints(primarySamples);
        const auto gt = GtPoints(primarySamples);
        for (std::size_t tauIndex = 0; tauIndex < kTausM.size(); ++tauIndex)
        {
            report.micro_primary_prf[tauIndex].tau = kTausM[tauIndex];
            report.micro_primary_prf[tauIndex].nearest = ComputePrfMetrics(predicted, gt, kTausM[tauIndex], false);
            report.micro_primary_prf[tauIndex].greedy = ComputePrfMetrics(predicted, gt, kTausM[tauIndex], true);
        }
        report.micro_primary_cd = ComputeChamferMetrics(predicted, gt);
    }
    {
        const auto predicted = PredictedPoints(secondarySamples);
        const auto gt = GtPoints(secondarySamples);
        for (std::size_t tauIndex = 0; tauIndex < kTausM.size(); ++tauIndex)
        {
            report.micro_secondary_prf[tauIndex].tau = kTausM[tauIndex];
            report.micro_secondary_prf[tauIndex].nearest = ComputePrfMetrics(predicted, gt, kTausM[tauIndex], false);
            report.micro_secondary_prf[tauIndex].greedy = ComputePrfMetrics(predicted, gt, kTausM[tauIndex], true);
        }
        report.micro_secondary_cd = ComputeChamferMetrics(predicted, gt);
    }

    report.per_frame = AggregatePerFrame(samples, completeRaw);
    report.per_camera = AggregatePerCamera(samples, completeRaw);
    report.macro = SummarizeMacro(report.per_frame);

    // Extend the macro block with the per-frame primary-tau P/R/F.
    std::vector<double> precisions;
    std::vector<double> recalls;
    std::vector<double> fScores;
    precisions.reserve(completeRaw.size());
    recalls.reserve(completeRaw.size());
    fScores.reserve(completeRaw.size());
    for (const auto &entry : completeRaw)
    {
        std::vector<RaySample> framePrimary;
        for (const auto &sample : samples)
        {
            if (sample.frame == entry.frame && sample.bucket != rays::Bucket::B50_60)
            {
                framePrimary.push_back(sample);
            }
        }
        const auto predicted = PredictedPoints(framePrimary);
        const auto gt = GtPoints(framePrimary);
        const PrfMetrics prf = ComputePrfMetrics(predicted, gt, kPrimaryTauM, false);
        if (prf.precision.has_value())
        {
            precisions.push_back(*prf.precision);
        }
        if (prf.recall.has_value())
        {
            recalls.push_back(*prf.recall);
        }
        if (prf.f_score.has_value())
        {
            fScores.push_back(*prf.f_score);
        }
    }
    AssignMeanStd(precisions, report.macro.precision_mean, report.macro.precision_std);
    AssignMeanStd(recalls, report.macro.recall_mean, report.macro.recall_std);
    AssignMeanStd(fScores, report.macro.f_score_mean, report.macro.f_score_std);

    // One row per frame x camera x bucket, including empty cells.
    report.rows.reserve(completeRaw.size() * 8U);
    for (const auto &entry : completeRaw)
    {
        for (std::uint16_t camera = 0; camera < 2U; ++camera)
        {
            for (int bucketIndex = 0; bucketIndex < 4; ++bucketIndex)
            {
                RowMetrics row;
                row.frame = entry.frame;
                row.camera = camera;
                row.bucket = static_cast<rays::Bucket>(bucketIndex);

                std::vector<RaySample> rowSamples;
                for (const auto &sample : samples)
                {
                    if (sample.frame == entry.frame && sample.camera == camera &&
                        static_cast<int>(sample.bucket) == bucketIndex)
                    {
                        rowSamples.push_back(sample);
                    }
                }

                // Row coverage is per-camera, per-bucket: N_valid_bucket /
                // N_raw_bucket from the camera's pre-filter bucket population.
                RawCounts rowRaw;
                rowRaw.raw_total = entry.per_camera[camera].raw_bucket[bucketIndex];
                row.cell = AggregateCell(rowSamples, rowRaw);

                const auto predicted = PredictedPoints(rowSamples);
                const auto gt = GtPoints(rowSamples);
                for (std::size_t tauIndex = 0; tauIndex < kTausM.size(); ++tauIndex)
                {
                    row.prf[tauIndex].tau = kTausM[tauIndex];
                    row.prf[tauIndex].nearest = ComputePrfMetrics(predicted, gt, kTausM[tauIndex], false);
                    row.prf[tauIndex].greedy = ComputePrfMetrics(predicted, gt, kTausM[tauIndex], true);
                }
                row.cd = ComputeChamferMetrics(predicted, gt);

                report.rows.push_back(std::move(row));
            }
        }
    }
    return report;
}
} // namespace raylidar::eval
