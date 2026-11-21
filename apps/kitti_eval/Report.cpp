#include "Report.h"

#include "simulation/PlyIO.h"
#include "simulation/SceneCoordinates.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <utility>

namespace raylidar::kitti
{
namespace
{
namespace sim = raylidar::simulation;

constexpr float kEvaluationRangeMin = 1.5f;
constexpr float kFarRange = 30.0f;

using Clock = std::chrono::steady_clock;

void PrintTimings(const sim::ScanTimings &timings, double generate, std::size_t rayCount)
{
    const double total = generate + timings.Total;

    std::cout << "  Ray generation (CPU):   " << generate << " ms\n"
              << "  Upload rays/indices (CPU): " << timings.Upload << " ms\n"
              << "  Submit + wait (CPU):    " << timings.SubmitWait << " ms\n"
              << "    Ray generation (GPU): " << timings.RayGen << " ms\n"
              << "    Trace (GPU):          " << timings.Trace << " ms\n"
              << "    Copy hits (GPU):      " << timings.CopyHits << " ms\n"
              << "  Readback + points (CPU):" << timings.Readback << " ms\n"
              << "  End-to-end:             " << total << " ms (" << 1000.0 / total << " scans/s, "
              << static_cast<double>(rayCount) / (total * 1.0e3) << " Mrays/s)\n";
}

std::vector<std::array<float, 3>> ToWorld(std::span<const std::array<float, 3>> local,
                                          const std::array<double, 3> &origin)
{
    std::vector<std::array<float, 3>> world;
    world.reserve(local.size());

    for (const auto &point : local)
    {
        world.push_back({
            static_cast<float>(point[0] + origin[0]),
            static_cast<float>(point[1] + origin[1]),
            static_cast<float>(point[2] + origin[2]),
        });
    }

    return world;
}

float Percentile(std::vector<float> values, double fraction)
{
    if (values.empty())
    {
        return 0.0f;
    }

    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

struct GtMetrics
{
    std::size_t Considered = 0;
    double SignedError = 0.0;
    double SquaredError = 0.0;
    std::vector<float> AbsoluteErrors;

    double HitRate() const
    {
        return Considered > 0 ? 100.0 * static_cast<double>(AbsoluteErrors.size()) / static_cast<double>(Considered)
                              : 0.0;
    }

    double Mean() const
    {
        return SignedError / HitCount();
    }

    double Rmse() const
    {
        return std::sqrt(SquaredError / HitCount());
    }

    double Within(float distance) const
    {
        const auto count = std::ranges::count_if(AbsoluteErrors, [distance](float error) { return error <= distance; });
        return 100.0 * static_cast<double>(count) / HitCount();
    }

  private:
    double HitCount() const
    {
        return static_cast<double>(std::max<std::size_t>(1, AbsoluteErrors.size()));
    }
};

template <typename Filter>
GtMetrics Summarize(const sim::GtRays &gt, std::span<const rt::Hit> hits, const Filter &filter)
{
    GtMetrics metrics;
    for (std::size_t index = 0; index < gt.Rays.size(); ++index)
    {
        if (gt.RangeGt[index] < kEvaluationRangeMin || !filter(index))
        {
            continue;
        }

        ++metrics.Considered;
        if (hits[index].Valid == 0)
        {
            continue;
        }

        const double error = static_cast<double>(hits[index].T) - gt.RangeGt[index];
        metrics.SignedError += error;
        metrics.SquaredError += error * error;
        metrics.AbsoluteErrors.push_back(static_cast<float>(std::abs(error)));
    }
    return metrics;
}

void PrintSummaryRow(std::string_view name, const sim::GtRays &gt, const sim::ScanResult &result)
{
    const auto all = Summarize(gt, result.Hits, [](std::size_t) { return true; });
    const auto far = Summarize(gt, result.Hits, [&gt](std::size_t index) { return gt.RangeGt[index] >= kFarRange; });

    std::cout << std::left << std::setw(10) << name << std::right << std::setw(8) << all.HitRate() << std::setw(9)
              << all.Mean() << std::setw(9) << Percentile(all.AbsoluteErrors, 0.5) << std::setw(9) << all.Rmse()
              << std::setw(9) << far.HitRate() << std::setw(9) << far.Mean() << std::setw(9)
              << Percentile(far.AbsoluteErrors, 0.5) << std::setw(9) << far.Rmse() << std::setw(12)
              << result.Timings.Trace << '\n';
}

} // namespace

void EvaluateGtRays(const sim::GtRays &gt, const sim::ScanResult &result, std::string_view sceneName,
                    const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory)
{
    const auto metrics = Summarize(gt, result.Hits, [](std::size_t) { return true; });
    const double raysPerSecond = static_cast<double>(gt.Rays.size()) / (result.Timings.Trace * 1.0e-3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== GT camera rays vs " << sceneName << " ===\n"
              << "Rays traced:        " << gt.Rays.size() << '\n'
              << "Trace time:         " << result.Timings.Trace << " ms (" << raysPerSecond / 1.0e6 << " Mrays/s)\n"
              << "Evaluated (t>=1.5): " << metrics.Considered << '\n'
              << "Hit rate:           " << metrics.HitRate() << " %\n"
              << "Mean error:         " << metrics.Mean() << " m (signed)\n"
              << "Median |error|:     " << Percentile(metrics.AbsoluteErrors, 0.5) << " m\n"
              << "P90 |error|:        " << Percentile(metrics.AbsoluteErrors, 0.9) << " m\n"
              << "RMSE:               " << metrics.Rmse() << " m\n"
              << "|error| <= 0.1 m:   " << metrics.Within(0.1f) << " %\n"
              << "|error| <= 0.5 m:   " << metrics.Within(0.5f) << " %\n"
              << "Per-stage cost (rays loaded from disk, no generation):\n";

    PrintTimings(result.Timings, 0.0, gt.Rays.size());

    if (outputDirectory.empty())
    {
        return;
    }

    const std::uint32_t frame = gt.Frames.front();
    std::vector<std::array<float, 3>> simulated;
    std::vector<std::array<float, 3>> measured;

    for (std::size_t index = 0; index < gt.Rays.size() && gt.Frames[index] == frame; ++index)
    {
        const auto &ray = gt.Rays[index];
        const auto &hit = result.Hits[index];

        measured.push_back({
            ray.Origin[0] + gt.RangeGt[index] * ray.Direction[0],
            ray.Origin[1] + gt.RangeGt[index] * ray.Direction[1],
            ray.Origin[2] + gt.RangeGt[index] * ray.Direction[2],
        });

        if (hit.Valid != 0)
        {
            simulated.push_back({
                ray.Origin[0] + hit.T * ray.Direction[0],
                ray.Origin[1] + hit.T * ray.Direction[1],
                ray.Origin[2] + hit.T * ray.Direction[2],
            });
        }
    }

    const auto frameName = std::to_string(frame);
    sim::PlyIO::WritePointCloud(outputDirectory / ("gt_" + frameName + "_measured.ply"), ToWorld(measured, origin));
    sim::PlyIO::WritePointCloud(outputDirectory / ("gt_" + frameName + "_simulated.ply"), ToWorld(simulated, origin));
}

void PrintHybridRouting(const sim::GtRays &gt, const sim::HybridScanResult &hybridResult)
{
    const auto total = static_cast<double>(gt.Rays.size());
    const auto isSource = [&hybridResult](sim::HitSource source) {
        return [&hybridResult, source](std::size_t index) { return hybridResult.Sources[index] == source; };
    };
    const auto fromMesh = Summarize(gt, hybridResult.Result.Hits, isSource(sim::HitSource::Mesh));
    const auto fromGaussian = Summarize(gt, hybridResult.Result.Hits, isSource(sim::HitSource::Gaussian));

    std::cout << "\nHybrid routing (mesh misses go to the 2D Gaussians):\n"
              << "  Rays sent to Gaussian:  " << 100.0 * static_cast<double>(hybridResult.RaysSentToGaussian) / total
              << " %\n"
              << "  Final hits from mesh:   " << 100.0 * static_cast<double>(hybridResult.MeshHits) / total
              << " % (median " << Percentile(fromMesh.AbsoluteErrors, 0.5) << " m)\n"
              << "  Final hits from Gauss:  " << 100.0 * static_cast<double>(hybridResult.GaussianHits) / total
              << " % (median " << Percentile(fromGaussian.AbsoluteErrors, 0.5) << " m)\n"
              << "  Wall time: mesh " << hybridResult.MeshMilliseconds << " ms, routing "
              << hybridResult.RoutingMilliseconds << " ms, Gaussian " << hybridResult.GaussianMilliseconds << " ms\n";
}

void PrintComparison(const sim::GtRays &gt, const sim::ScanResult &meshResult, const sim::ScanResult &gaussianResult,
                     const sim::HybridScanResult &hybridResult, std::span<const std::uint8_t> untrustedTriangles)
{
    std::cout << "\n=== GT camera rays: mesh vs 2D Gaussian vs hybrid (far = GT range >= 30 m) ===\n"
              << std::left << std::setw(10) << "Mode" << std::right << std::setw(8) << "hit%" << std::setw(9) << "mean"
              << std::setw(9) << "median" << std::setw(9) << "RMSE" << std::setw(9) << "far hit%" << std::setw(9)
              << "far mean" << std::setw(9) << "far med" << std::setw(9) << "far RMSE" << std::setw(12) << "trace ms"
              << '\n';
    PrintSummaryRow("Mesh", gt, meshResult);
    PrintSummaryRow("Gaussian", gt, gaussianResult);
    PrintSummaryRow("Hybrid", gt, hybridResult.Result);

    PrintHybridRouting(gt, hybridResult);

    const auto onTriangles = [&meshResult, &untrustedTriangles](bool untrusted) {
        return [&meshResult, &untrustedTriangles, untrusted](std::size_t index) {
            const auto &hit = meshResult.Hits[index];
            return hit.Valid != 0 && hit.PrimitiveId < untrustedTriangles.size() &&
                   (untrustedTriangles[hit.PrimitiveId] != 0) == untrusted;
        };
    };
    const auto earlyRate = [&gt, &meshResult](const auto &filter) {
        std::size_t count = 0;
        std::size_t early = 0;
        for (std::size_t index = 0; index < gt.Rays.size(); ++index)
        {
            if (gt.RangeGt[index] < kEvaluationRangeMin || !filter(index))
            {
                continue;
            }
            ++count;
            early += meshResult.Hits[index].T < gt.RangeGt[index] - 1.0f ? 1 : 0;
        }
        return std::pair{count, count > 0 ? 100.0 * static_cast<double>(early) / static_cast<double>(count) : 0.0};
    };
    const auto [trustedCount, trustedEarly] = earlyRate(onTriangles(false));
    const auto [untrustedCount, untrustedEarly] = earlyRate(onTriangles(true));
    const auto trusted = Summarize(gt, meshResult.Hits, onTriangles(false));
    const auto untrusted = Summarize(gt, meshResult.Hits, onTriangles(true));
    const auto meshHitCount = static_cast<double>(trustedCount + untrustedCount);

    std::cout << "\nTrust label on mesh hits (checked against GT):\n"
              << "  Trusted:   " << 100.0 * static_cast<double>(trustedCount) / meshHitCount << " % of hits, mean "
              << trusted.Mean() << " m, median " << Percentile(trusted.AbsoluteErrors, 0.5) << " m, early > 1 m "
              << trustedEarly << " %\n"
              << "  Untrusted: " << 100.0 * static_cast<double>(untrustedCount) / meshHitCount << " % of hits, mean "
              << untrusted.Mean() << " m, median " << Percentile(untrusted.AbsoluteErrors, 0.5) << " m, early > 1 m "
              << untrustedEarly << " %\n";
}

void RunVelodyneScans(const sim::LidarSensor &sensor, std::span<const sim::VelodynePose> poses,
                      const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory,
                      std::string_view outputPrefix, std::span<const std::uint8_t> untrustedTriangles, bool gpuRayGen,
                      const TraceFrame &trace)
{
    sim::ScanTimings sum;
    double generateSum = 0.0;
    double bestTrace = 1.0e30;
    std::vector<float> traceTimes;
    traceTimes.reserve(poses.size());
    std::size_t totalHits = 0;
    std::size_t measuredFrames = 0;

    for (std::size_t index = 0; index < poses.size(); ++index)
    {
        const auto pose = sim::ToLocalTransform(poses[index].WorldFromVelodyne, origin);
        double generate = 0.0;

        const auto frameScan = [&] {
            if (gpuRayGen)
            {
                return trace(pose, {});
            }

            const auto generateStart = Clock::now();
            const auto rays = sensor.GenerateRays(pose);
            generate = std::chrono::duration<double, std::milli>(Clock::now() - generateStart).count();
            return trace(pose, rays);
        }();
        const auto &result = frameScan.Result;
        totalHits += result.Points.size();

        if (!outputDirectory.empty())
        {
            const auto name = std::string(outputPrefix) + std::to_string(poses[index].Frame);
            sim::WriteLidarFrame(
                outputDirectory / (name + ".rlf"),
                sim::MakeLidarFrame(sensor, poses[index], result.Hits, frameScan.Sources, untrustedTriangles));
            if (index == 0)
            {
                sim::PlyIO::WritePointCloud(outputDirectory / (name + ".ply"), ToWorld(result.Points, origin));
            }
        }

        if (index == 0 && poses.size() > 1)
        {
            continue;
        }

        ++measuredFrames;
        generateSum += generate;
        sum += result.Timings;
        bestTrace = std::min(bestTrace, result.Timings.Trace);
        traceTimes.push_back(static_cast<float>(result.Timings.Trace));
    }

    const double frames = static_cast<double>(measuredFrames);

    sim::ScanTimings average;
    average.Upload = sum.Upload / frames;
    average.RayGen = sum.RayGen / frames;
    average.Trace = sum.Trace / frames;
    average.CopyHits = sum.CopyHits / frames;
    average.SubmitWait = sum.SubmitWait / frames;
    average.Readback = sum.Readback / frames;
    average.Total = sum.Total / frames;

    std::cout << "\n=== HDL-64E scans from T_W_V ===\n"
              << "Frames:             " << poses.size() << " x " << sensor.RayCount() << " rays\n"
              << "Average hits:       " << static_cast<double>(totalHits) / static_cast<double>(poses.size())
              << " points/frame\n"
              << "Best trace time:    " << bestTrace << " ms/frame\n"
              << "Median trace time:  " << Percentile(traceTimes, 0.5) << " ms/frame\n"
              << "P95 trace time:     " << Percentile(traceTimes, 0.95) << " ms/frame\n"
              << "Average per frame (" << measuredFrames << " frames"
              << (poses.size() > 1 ? ", first frame excluded" : "") << "):\n";

    PrintTimings(average, generateSum / frames, sensor.RayCount());
}

} // namespace raylidar::kitti
