#include "rt/ASBuilder.h"
#include "rt/MeshScene.h"
#include "simulation/KittiBundle.h"
#include "simulation/LidarSensor.h"
#include "simulation/LidarSimulator.h"
#include "simulation/PlyIO.h"
#include "simulation/SceneCoordinates.h"
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
#include <vector>

namespace
{
namespace sim = raylidar::simulation;

constexpr float kEvaluationRangeMin = 1.5f;

using Clock = std::chrono::steady_clock;

void PrintTimings(const sim::ScanTimings &timings, double generate, std::size_t rayCount)
{
    const double total = generate + timings.Total;

    std::cout << "  Ray generation (CPU):   " << generate << " ms\n"
              << "  Upload rays (CPU):      " << timings.Upload << " ms\n"
              << "  Submit + wait (CPU):    " << timings.SubmitWait << " ms\n"
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
void EvaluateGtRays(sim::LidarSimulator &simulator, const raylidar::rt::MeshScene &scene, const sim::GtRays &gt,
                    const std::array<double, 3> &origin, const std::filesystem::path &outputDirectory)
{
    const auto result = simulator.Scan(scene, gt.Rays);

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
    std::cout << "\n=== GT camera rays vs mesh ===\n"
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
void RunVelodyneScans(sim::LidarSimulator &simulator, const raylidar::rt::MeshScene &scene,
                      const std::vector<sim::VelodynePose> &poses, const std::array<double, 3> &origin,
                      const std::filesystem::path &outputDirectory)
{
    const sim::LidarSensor sensor(sim::LidarSensorDescription{});

    sim::ScanTimings sum;
    double generateSum = 0.0;
    double bestTrace = 1.0e30;
    std::size_t totalHits = 0;
    std::size_t measuredFrames = 0;

    for (std::size_t index = 0; index < poses.size(); ++index)
    {
        const auto generateStart = Clock::now();
        const auto rays = sensor.GenerateRays(sim::ToLocalTransform(poses[index].WorldFromVelodyne, origin));
        const double generate = std::chrono::duration<double, std::milli>(Clock::now() - generateStart).count();

        const auto result = simulator.Scan(scene, rays);
        totalHits += result.Points.size();

        if (index == 0 && !outputDirectory.empty())
        {
            sim::PlyIO::WritePointCloud(outputDirectory / ("velodyne_" + std::to_string(poses[index].Frame) + ".ply"),
                                        ToWorld(result.Points, origin));
        }

        if (index == 0 && poses.size() > 1)
        {
            continue;
        }

        ++measuredFrames;
        generateSum += generate;
        sum.Upload += result.Timings.Upload;
        sum.Trace += result.Timings.Trace;
        sum.CopyHits += result.Timings.CopyHits;
        sum.SubmitWait += result.Timings.SubmitWait;
        sum.Readback += result.Timings.Readback;
        sum.Total += result.Timings.Total;
        bestTrace = std::min(bestTrace, result.Timings.Trace);
    }

    const double frames = static_cast<double>(measuredFrames);

    sim::ScanTimings average;
    average.Upload = sum.Upload / frames;
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
              << "Average per frame (" << measuredFrames << " frames, first frame excluded):\n";

    PrintTimings(average, generateSum / frames, sensor.RayCount());
}

void Run(const std::filesystem::path &shaderPath, const std::filesystem::path &bundle,
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

    EvaluateGtRays(simulator, scene, gt, origin, outputDirectory);
    RunVelodyneScans(simulator, scene, poses, origin, outputDirectory);

    if (!outputDirectory.empty())
    {
        std::cout << "\nWrote point clouds to " << outputDirectory.string() << '\n';
    }
}
} // namespace

int main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: raylidar_kitti_eval <LidarTrace.spv> <bundle_dir> [output_dir]\n";
        return EXIT_FAILURE;
    }

    try
    {
        Run(argv[1], argv[2], argc == 4 ? argv[3] : "");
        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "KITTI evaluation failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
