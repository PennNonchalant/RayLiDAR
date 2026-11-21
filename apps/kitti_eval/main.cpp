#include "rt/ASBuilder.h"
#include "rt/GaussianScene.h"
#include "rt/MeshScene.h"
#include "simulation/HybridScan.h"
#include "simulation/KittiBundle.h"
#include "simulation/LidarFrame.h"
#include "simulation/LidarSensor.h"
#include "simulation/LidarSimulator.h"
#include "simulation/PlyIO.h"
#include "simulation/SceneCoordinates.h"
#include "simulation/TriangleTrust.h"
#include "vulkan/CommandContext.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

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
              << "  Upload rays (CPU):      " << timings.Upload << " ms\n"
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

// Casts every GT camera ray against the mesh and compares the hit range with the measured LiDAR range.
void EvaluateGtRays(const sim::GtRays &gt, const sim::ScanResult &result, std::string_view sceneName,
                    const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory)
{
    std::size_t considered = 0;
    std::size_t hits = 0;
    double signedSum = 0.0;
    double squaredSum = 0.0;
    std::size_t within10cm = 0;
    std::size_t within50cm = 0;
    std::vector<float> absoluteErrors;
    absoluteErrors.reserve(gt.Rays.size());

    for (std::size_t index = 0; index < gt.Rays.size(); ++index)
    {
        if (gt.RangeGt[index] < kEvaluationRangeMin)
        {
            continue;
        }

        ++considered;
        const auto &hit = result.Hits[index];

        if (hit.Valid == 0)
        {
            continue;
        }

        ++hits;
        const double error = static_cast<double>(hit.T) - gt.RangeGt[index];
        const auto absoluteError = static_cast<float>(std::abs(error));

        signedSum += error;
        squaredSum += error * error;
        within10cm += absoluteError <= 0.1f ? 1 : 0;
        within50cm += absoluteError <= 0.5f ? 1 : 0;
        absoluteErrors.push_back(absoluteError);
    }

    const double hitCount = std::max<double>(1.0, static_cast<double>(hits));
    const double raysPerSecond = static_cast<double>(gt.Rays.size()) / (result.Timings.Trace * 1.0e-3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=== GT camera rays vs " << sceneName << " ===\n"
              << "Rays traced:        " << gt.Rays.size() << '\n'
              << "Trace time:         " << result.Timings.Trace << " ms (" << raysPerSecond / 1.0e6 << " Mrays/s)\n"
              << "Evaluated (t>=1.5): " << considered << '\n'
              << "Hit rate:           " << 100.0 * static_cast<double>(hits) / static_cast<double>(considered) << " %\n"
              << "Mean error:         " << signedSum / hitCount << " m (signed)\n"
              << "Median |error|:     " << Percentile(absoluteErrors, 0.5) << " m\n"
              << "P90 |error|:        " << Percentile(absoluteErrors, 0.9) << " m\n"
              << "RMSE:               " << std::sqrt(squaredSum / hitCount) << " m\n"
              << "|error| <= 0.1 m:   " << 100.0 * static_cast<double>(within10cm) / hitCount << " %\n"
              << "|error| <= 0.5 m:   " << 100.0 * static_cast<double>(within50cm) / hitCount << " %\n"
              << "Per-stage cost (rays loaded from disk, no generation):\n";

    PrintTimings(result.Timings, 0.0, gt.Rays.size());

    if (outputDirectory.empty())
    {
        return;
    }

    // Simulated vs measured points for the first frame, in the world frame so they overlay mesh.ply.
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

// Simulates a full HDL-64E sweep from the real Velodyne pose of every frame and reports the per-stage cost.
// The first frame is excluded from the averages because it pays one-time warm-up costs.
// With an output directory, every frame is written as <prefix><frame>.rlf (see LidarFrame.h) and the first frame also
// as a world-space .ply for viewing.
// scan takes either the CPU-generated rays (span of rt::Ray) or, for GPU ray generation, the local sensor pose.
struct FrameScan
{
    sim::ScanResult Result;
    std::vector<sim::HitSource> Sources; // one per ray
};

template <typename Scan>
void RunVelodyneScans(const sim::LidarSensor &sensor, std::span<const sim::VelodynePose> poses,
                      const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory,
                      std::string_view outputPrefix, std::span<const std::uint8_t> untrustedTriangles, const Scan &scan)
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
            if constexpr (std::is_invocable_v<const Scan &, const sim::Transform &>)
            {
                return scan(pose);
            }
            else
            {
                const auto generateStart = Clock::now();
                const auto rays = sensor.GenerateRays(pose);
                generate = std::chrono::duration<double, std::milli>(Clock::now() - generateStart).count();
                return scan(std::span<const raylidar::rt::Ray>(rays));
            }
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
        sum.Upload += result.Timings.Upload;
        sum.RayGen += result.Timings.RayGen;
        sum.Trace += result.Timings.Trace;
        sum.CopyHits += result.Timings.CopyHits;
        sum.SubmitWait += result.Timings.SubmitWait;
        sum.Readback += result.Timings.Readback;
        sum.Total += result.Timings.Total;
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

void RunMesh(const std::filesystem::path &shaderPath, const std::filesystem::path &bundle,
             const std::filesystem::path &outputDirectory)
{
    if (!outputDirectory.empty())
    {
        std::filesystem::create_directories(outputDirectory);
    }

    const auto loadStart = Clock::now();
    const auto mesh = sim::PlyIO::LoadMesh(bundle / "mesh" / "mesh.ply");
    const auto origin = sim::BoundingBoxCenter(mesh.Positions);
    const auto positions = sim::ToLocalFloat(mesh.Positions, origin);
    const auto gt = sim::LoadGtRays(bundle / "gt_rays" / "gt_rays.bin", origin);
    const auto poses = sim::LoadVelodynePoses(bundle / "camera" / "camera_params.json");
    const auto loadSeconds = std::chrono::duration<double>(Clock::now() - loadStart).count();

    std::cout << std::fixed << std::setprecision(3) << "Mesh: " << positions.size() << " vertices, "
              << mesh.Indices.size() / 3 << " triangles\n"
              << "Local origin (W): " << origin[0] << ", " << origin[1] << ", " << origin[2] << '\n'
              << "Loaded bundle in " << loadSeconds << " s\n";

    raylidar::vulkan::VulkanContext context;

    const raylidar::rt::MeshDescription meshDescription{positions, mesh.Indices};
    const raylidar::rt::MeshInstanceDescription instance{};

    raylidar::rt::MeshSceneDescription sceneDescription;
    sceneDescription.Meshes = std::span(&meshDescription, 1);
    sceneDescription.Instances = std::span(&instance, 1);

    raylidar::rt::MeshScene scene(context, sceneDescription);
    {
        raylidar::rt::ASBuilder builder(context);
        raylidar::vulkan::CommandContext buildCommands(context);
        scene.RecordBuild(buildCommands.Begin(), builder);
        buildCommands.End();
        buildCommands.SubmitAndWait();
        scene.ReleaseUploadResources();
    }

    const sim::LidarSensor sensor(sim::LidarSensorDescription{});
    const auto capacity = static_cast<std::uint32_t>(std::max<std::size_t>(gt.Rays.size(), sensor.RayCount()));
    sim::LidarSimulator simulator(context, shaderPath, capacity);

    EvaluateGtRays(gt, simulator.Scan(scene, gt.Rays), "Mesh", origin, outputDirectory);
    RunVelodyneScans(sensor, poses, origin, outputDirectory, "velodyne_", {},
                     [&simulator, &scene](std::span<const raylidar::rt::Ray> rays) {
                         auto result = simulator.Scan(scene, rays);
                         auto sources = sim::SourcesFor(result.Hits, sim::HitSource::Mesh);
                         return FrameScan{std::move(result), std::move(sources)};
                     });

    // Same sweep with the rays generated on the GPU: no CPU ray generation and no ray upload.
    const auto rayGenShaderPath = shaderPath.parent_path() / "LidarRayGen.spv";
    if (std::filesystem::exists(rayGenShaderPath))
    {
        simulator.EnableGpuRayGeneration(rayGenShaderPath);
        std::cout << "\n--- GPU ray generation ---";
        RunVelodyneScans(sensor, poses, origin, outputDirectory, "velodyne_gpu_", {},
                         [&simulator, &scene, &sensor](const sim::Transform &pose) {
                             auto result = simulator.Scan(scene, sensor, pose);
                             auto sources = sim::SourcesFor(result.Hits, sim::HitSource::Mesh);
                             return FrameScan{std::move(result), std::move(sources)};
                         });
    }

    if (!outputDirectory.empty())
    {
        std::cout << "\nWrote point clouds to " << outputDirectory.string() << '\n';
    }
}

void RunGaussian(const std::filesystem::path &shaderPath, const std::filesystem::path &bundle, float sigmaCutoff,
                 float alphaThreshold, const std::filesystem::path &outputDirectory, std::uint32_t hitBatchSize,
                 std::uint32_t blasGroupSizeLog2)
{
    const auto loadStart = Clock::now();
    const auto poses = sim::LoadVelodynePoses(bundle / "camera" / "camera_params.json");
    const auto &pose = poses.front();
    const auto &worldFromVelodyne = pose.WorldFromVelodyne;
    const std::array<double, 3> origin{worldFromVelodyne[0][3], worldFromVelodyne[1][3], worldFromVelodyne[2][3]};

    const auto gaussians = sim::PlyIO::LoadGaussians(bundle / "gs" / "2dgs" / "point_cloud.ply", origin);
    const auto gt = sim::LoadGtRays(bundle / "gt_rays" / "gt_rays.bin", origin);
    const auto loadSeconds = std::chrono::duration<double>(Clock::now() - loadStart).count();

    std::cout << std::fixed << std::setprecision(3) << "2D Gaussian: " << gaussians.size() << " splats\n"
              << "Local origin (W): " << origin[0] << ", " << origin[1] << ", " << origin[2] << '\n'
              << "Loaded bundle in " << loadSeconds << " s\n";

    raylidar::vulkan::VulkanContext context;
    std::cout << "Device:             " << context.PhysicalDeviceProperties().deviceName.data() << '\n';
    const auto buildStart = Clock::now();
    const raylidar::rt::GaussianSceneDescription sceneDescription{
        .Gaussians = gaussians,
        .SigmaCutoff = sigmaCutoff,
        .BlasGroupSizeLog2 = blasGroupSizeLog2,
    };
    raylidar::rt::GaussianScene scene(context, sceneDescription);
    {
        raylidar::rt::ASBuilder builder(context);
        raylidar::vulkan::CommandContext buildCommands(context);
        scene.RecordBuild(buildCommands.Begin(), builder);
        buildCommands.End();
        buildCommands.SubmitAndWait();
        scene.ReleaseUploadResources();
    }
    const double buildMs = std::chrono::duration<double, std::milli>(Clock::now() - buildStart).count();

    const sim::LidarSensor sensor(sim::LidarSensorDescription{});
    const auto capacity = static_cast<std::uint32_t>(std::max<std::size_t>(gt.Rays.size(), sensor.RayCount()));
    sim::LidarSimulator simulator(context, shaderPath, capacity, raylidar::rt::RayQueryMode::Gaussian, hitBatchSize);

    if (!outputDirectory.empty())
    {
        std::filesystem::create_directories(outputDirectory);
    }

    std::cout << "Gaussian BLASes:    " << scene.BlasCount() << " (up to " << scene.GaussiansPerBatch()
              << " splats each)\n"
              << "Scene build:        " << buildMs << " ms\n"
              << "Tracing 2D Gaussians (in chunks)..." << std::endl;

    EvaluateGtRays(gt, sim::ScanGaussiansInChunks(simulator, scene, gt.Rays, alphaThreshold), "2D Gaussian", origin,
                   outputDirectory);

    RunVelodyneScans(sensor, poses, origin, outputDirectory, "velodyne_gaussian_", {},
                     [&simulator, &scene, alphaThreshold](std::span<const raylidar::rt::Ray> rays) {
                         auto result = sim::ScanGaussiansInChunks(simulator, scene, rays, alphaThreshold);
                         auto sources = sim::SourcesFor(result.Hits, sim::HitSource::Gaussian);
                         return FrameScan{std::move(result), std::move(sources)};
                     });

    if (!outputDirectory.empty())
    {
        std::cout << "\nWrote point clouds to " << outputDirectory.string() << '\n';
    }
}

// Accuracy summary over the GT rays, optionally restricted to far rays or to one hit source.
struct AccuracySummary
{
    double HitRate = 0.0;
    double Mean = 0.0;
    double Median = 0.0;
    double Rmse = 0.0;
};

template <typename Filter>
AccuracySummary Summarize(const sim::GtRays &gt, std::span<const raylidar::rt::Hit> hits, const Filter &filter)
{
    std::size_t considered = 0;
    double signedSum = 0.0;
    double squaredSum = 0.0;
    std::vector<float> absoluteErrors;

    for (std::size_t index = 0; index < gt.Rays.size(); ++index)
    {
        if (gt.RangeGt[index] < kEvaluationRangeMin || !filter(index))
        {
            continue;
        }

        ++considered;

        if (hits[index].Valid == 0)
        {
            continue;
        }

        const double error = static_cast<double>(hits[index].T) - gt.RangeGt[index];
        signedSum += error;
        squaredSum += error * error;
        absoluteErrors.push_back(static_cast<float>(std::abs(error)));
    }

    AccuracySummary summary;
    const auto hitCount = static_cast<double>(absoluteErrors.size());

    if (considered > 0)
    {
        summary.HitRate = 100.0 * hitCount / static_cast<double>(considered);
    }

    if (hitCount > 0.0)
    {
        summary.Mean = signedSum / hitCount;
        summary.Median = Percentile(absoluteErrors, 0.5);
        summary.Rmse = std::sqrt(squaredSum / hitCount);
    }

    return summary;
}

void PrintSummaryRow(std::string_view name, const sim::GtRays &gt, std::span<const raylidar::rt::Hit> hits,
                     double traceMilliseconds)
{
    const auto all = Summarize(gt, hits, [](std::size_t) { return true; });
    const auto far = Summarize(gt, hits, [&gt](std::size_t index) { return gt.RangeGt[index] >= kFarRange; });

    std::cout << std::left << std::setw(10) << name << std::right << std::setw(8) << all.HitRate << std::setw(9)
              << all.Mean << std::setw(9) << all.Median << std::setw(9) << all.Rmse << std::setw(9) << far.HitRate
              << std::setw(9) << far.Mean << std::setw(9) << far.Median << std::setw(9) << far.Rmse << std::setw(12)
              << traceMilliseconds << '\n';
}

// Mesh vs 2D Gaussian vs hybrid on the same GT rays and the same local frame.
void RunCompare(const std::filesystem::path &meshShaderPath, const std::filesystem::path &gaussianShaderPath,
                const std::filesystem::path &meshBundle, const std::filesystem::path &gaussianBundle, float sigmaCutoff,
                float alphaThreshold, const std::filesystem::path &outputDirectory, std::uint32_t hitBatchSize,
                std::uint32_t blasGroupSizeLog2)
{
    const auto loadStart = Clock::now();
    const auto mesh = sim::PlyIO::LoadMesh(meshBundle / "mesh" / "mesh.ply");
    const auto origin = sim::BoundingBoxCenter(mesh.Positions);
    const auto positions = sim::ToLocalFloat(mesh.Positions, origin);
    const auto gaussians = sim::PlyIO::LoadGaussians(gaussianBundle / "gs" / "2dgs" / "point_cloud.ply", origin);
    const auto gt = sim::LoadGtRays(meshBundle / "gt_rays" / "gt_rays.bin", origin);
    const auto poses = sim::LoadVelodynePoses(meshBundle / "camera" / "camera_params.json");
    const auto loadSeconds = std::chrono::duration<double>(Clock::now() - loadStart).count();

    std::cout << std::fixed << std::setprecision(3) << "Mesh: " << mesh.Indices.size() / 3
              << " triangles, 2D Gaussian: " << gaussians.size() << " splats, GT rays: " << gt.Rays.size() << '\n'
              << "Loaded in " << loadSeconds << " s\n";

    // Ground-truth-free: compare the mesh with the 2D Gaussians and flag triangles where they disagree.
    // The flags only label output points; routing them to the Gaussians made accuracy worse.
    const sim::TriangleTrustSettings trustSettings;
    const auto trustStart = Clock::now();
    const auto trust = sim::FindUntrustedTriangles(positions, mesh.Indices, gaussians, trustSettings);
    const auto trustSeconds = std::chrono::duration<double>(Clock::now() - trustStart).count();
    std::cout << "Untrusted triangles (> " << trustSettings.MaxPlaneDistance << " m from the 2D Gaussian surface): "
              << 100.0 * static_cast<double>(trust.UntrustedCount) / static_cast<double>(trust.Untrusted.size())
              << " % (" << trust.NoNearbySplatCount << " with no splat nearby), computed in " << trustSeconds << " s\n";

    raylidar::vulkan::VulkanContext context;

    const raylidar::rt::MeshDescription meshDescription{positions, mesh.Indices};
    const raylidar::rt::MeshInstanceDescription instance{};
    raylidar::rt::MeshSceneDescription meshSceneDescription;
    meshSceneDescription.Meshes = std::span(&meshDescription, 1);
    meshSceneDescription.Instances = std::span(&instance, 1);
    raylidar::rt::MeshScene meshScene(context, meshSceneDescription);

    const raylidar::rt::GaussianSceneDescription gaussianSceneDescription{
        .Gaussians = gaussians,
        .SigmaCutoff = sigmaCutoff,
        .BlasGroupSizeLog2 = blasGroupSizeLog2,
    };
    raylidar::rt::GaussianScene gaussianScene(context, gaussianSceneDescription);
    {
        raylidar::rt::ASBuilder builder(context);
        raylidar::vulkan::CommandContext buildCommands(context);
        std::cout << "Building acceleration structures..." << std::endl;
        const auto &commandBuffer = buildCommands.Begin();
        meshScene.RecordBuild(commandBuffer, builder);
        gaussianScene.RecordBuild(commandBuffer, builder);
        buildCommands.End();
        buildCommands.SubmitAndWait();
        meshScene.ReleaseUploadResources();
        gaussianScene.ReleaseUploadResources();
    }
    std::cout << "Gaussian BLASes:    " << gaussianScene.BlasCount() << " (up to " << gaussianScene.GaussiansPerBatch()
              << " splats each)\n";

    const sim::LidarSensor sensor(sim::LidarSensorDescription{});
    const auto capacity = static_cast<std::uint32_t>(std::max<std::size_t>(gt.Rays.size(), sensor.RayCount()));
    sim::LidarSimulator meshSimulator(context, meshShaderPath, capacity);
    sim::LidarSimulator gaussianSimulator(context, gaussianShaderPath, capacity, raylidar::rt::RayQueryMode::Gaussian,
                                          hitBatchSize);
    sim::HybridScanner hybrid(meshSimulator, gaussianSimulator);

    std::cout << "Tracing mesh..." << std::endl;
    const auto meshResult = meshSimulator.Scan(meshScene, gt.Rays);
    std::cout << "Tracing 2D Gaussians (in chunks)..." << std::endl;
    const auto gaussianResult = sim::ScanGaussiansInChunks(gaussianSimulator, gaussianScene, gt.Rays, alphaThreshold);
    std::cout << "Tracing hybrid..." << std::endl;
    const auto hybridResult = hybrid.Scan(meshScene, gaussianScene, gt.Rays, alphaThreshold);

    const auto total = static_cast<double>(gt.Rays.size());

    std::cout << "\n=== GT camera rays: mesh vs 2D Gaussian vs hybrid (far = GT range >= 30 m) ===\n"
              << std::left << std::setw(10) << "Mode" << std::right << std::setw(8) << "hit%" << std::setw(9) << "mean"
              << std::setw(9) << "median" << std::setw(9) << "RMSE" << std::setw(9) << "far hit%" << std::setw(9)
              << "far mean" << std::setw(9) << "far med" << std::setw(9) << "far RMSE" << std::setw(12) << "trace ms"
              << '\n';
    PrintSummaryRow("Mesh", gt, meshResult.Hits, meshResult.Timings.Trace);
    PrintSummaryRow("Gaussian", gt, gaussianResult.Hits, gaussianResult.Timings.Trace);
    PrintSummaryRow("Hybrid", gt, hybridResult.Result.Hits, hybridResult.Result.Timings.Trace);

    const auto isSource = [&hybridResult](sim::HitSource source) {
        return [&hybridResult, source](std::size_t index) { return hybridResult.Sources[index] == source; };
    };
    const auto fromMesh = Summarize(gt, hybridResult.Result.Hits, isSource(sim::HitSource::Mesh));
    const auto fromGaussian = Summarize(gt, hybridResult.Result.Hits, isSource(sim::HitSource::Gaussian));

    std::cout << "\nHybrid routing (mesh misses go to the 2D Gaussians):\n"
              << "  Rays sent to Gaussian:  " << 100.0 * static_cast<double>(hybridResult.RaysSentToGaussian) / total
              << " %\n"
              << "  Final hits from mesh:   " << 100.0 * static_cast<double>(hybridResult.MeshHits) / total
              << " % (median " << fromMesh.Median << " m)\n"
              << "  Final hits from Gauss:  " << 100.0 * static_cast<double>(hybridResult.GaussianHits) / total
              << " % (median " << fromGaussian.Median << " m)\n"
              << "  Wall time: mesh " << hybridResult.MeshMilliseconds << " ms, routing "
              << hybridResult.RoutingMilliseconds << " ms, Gaussian " << hybridResult.GaussianMilliseconds << " ms\n";

    // GT is used here only to check the trust label, never to compute it.
    const auto onTriangles = [&meshResult, &trust](bool untrusted) {
        return [&meshResult, &trust, untrusted](std::size_t index) {
            const auto &hit = meshResult.Hits[index];
            return hit.Valid != 0 && hit.PrimitiveId < trust.Untrusted.size() &&
                   (trust.Untrusted[hit.PrimitiveId] != 0) == untrusted;
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
              << trusted.Mean << " m, median " << trusted.Median << " m, early > 1 m " << trustedEarly << " %\n"
              << "  Untrusted: " << 100.0 * static_cast<double>(untrustedCount) / meshHitCount << " % of hits, mean "
              << untrusted.Mean << " m, median " << untrusted.Median << " m, early > 1 m " << untrustedEarly << " %\n";

    if (!outputDirectory.empty())
    {
        std::filesystem::create_directories(outputDirectory);
    }

    std::cout << "\n--- Mesh only ---";
    RunVelodyneScans(sensor, poses, origin, outputDirectory, "mesh_", trust.Untrusted,
                     [&meshSimulator, &meshScene](std::span<const raylidar::rt::Ray> rays) {
                         auto result = meshSimulator.Scan(meshScene, rays);
                         auto sources = sim::SourcesFor(result.Hits, sim::HitSource::Mesh);
                         return FrameScan{std::move(result), std::move(sources)};
                     });

    std::cout << "\n--- Hybrid ---";
    std::size_t routedRays = 0;
    std::size_t scannedRays = 0;
    RunVelodyneScans(sensor, poses, origin, outputDirectory, "hybrid_", trust.Untrusted,
                     [&](std::span<const raylidar::rt::Ray> rays) {
                         auto scan = hybrid.Scan(meshScene, gaussianScene, rays, alphaThreshold);
                         routedRays += scan.RaysSentToGaussian;
                         scannedRays += rays.size();
                         return FrameScan{std::move(scan.Result), std::move(scan.Sources)};
                     });

    std::cout << "Hybrid HDL-64E: " << 100.0 * static_cast<double>(routedRays) / static_cast<double>(scannedRays)
              << " % of rays sent to Gaussian\n";

    if (!outputDirectory.empty())
    {
        std::cout << "\nWrote .rlf frames to " << outputDirectory.string() << '\n';
    }
}
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const std::string_view mode = argc >= 2 ? argv[1] : "";

        if (mode == "--mesh" && (argc == 4 || argc == 5))
        {
            RunMesh(argv[2], argv[3], argc == 5 ? argv[4] : "");
            return EXIT_SUCCESS;
        }

        std::uint32_t hitBatchSize = raylidar::rt::kDefaultGaussianHitBatchSize;
        std::uint32_t blasGroupSizeLog2 = raylidar::rt::kDefaultGaussianBlasGroupSizeLog2;
        while ((mode == "--gaussian" || mode == "--compare") && argc >= 4)
        {
            const std::string_view flag = argv[argc - 2];
            const std::string_view value = argv[argc - 1];
            if (flag == "--hit-batch-size")
            {
                hitBatchSize = static_cast<std::uint32_t>(std::stoul(argv[argc - 1]));
            }
            else if (flag == "--gaussian-blas")
            {
                blasGroupSizeLog2 = value == "single" ? 0u : static_cast<std::uint32_t>(std::stoul(argv[argc - 1]));
            }
            else
            {
                break;
            }
            argc -= 2;
        }

        if (mode == "--gaussian" && (argc == 6 || argc == 7))
        {
            const float sigmaCutoff = std::stof(argv[4]);
            const float alphaThreshold = std::stof(argv[5]);
            RunGaussian(argv[2], argv[3], sigmaCutoff, alphaThreshold, argc == 7 ? argv[6] : "", hitBatchSize,
                        blasGroupSizeLog2);
            return EXIT_SUCCESS;
        }

        if (mode == "--compare" && (argc == 8 || argc == 9))
        {
            RunCompare(argv[2], argv[3], argv[4], argv[5], std::stof(argv[6]), std::stof(argv[7]),
                       argc == 9 ? argv[8] : "", hitBatchSize, blasGroupSizeLog2);
            return EXIT_SUCCESS;
        }

        std::cerr
            << "Usage: raylidar_kitti_eval --mesh <LidarTrace.spv> <bundle_dir> [output_dir]\n"
            << "       raylidar_kitti_eval --gaussian <GaussianTrace.spv> <bundle_dir> "
               "<sigma_cutoff> <alpha_threshold> [output_dir] [--hit-batch-size <K>] [--gaussian-blas <log2|single>]\n"
            << "       raylidar_kitti_eval --compare <LidarTrace.spv> <GaussianTrace.spv> <mesh_bundle> "
               "<gaussian_bundle> <sigma_cutoff> <alpha_threshold> [output_dir] [--hit-batch-size <K>] "
               "[--gaussian-blas <log2|single>]\n"
            << "Gaussian hit batch size K is positive and defaults to " << raylidar::rt::kDefaultGaussianHitBatchSize
            << ".\n"
            << "Gaussian BLAS defaults to single (also selected by 0). A positive log2 uses Morton groups of "
               "2^log2 Gaussians.\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "KITTI evaluation failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
