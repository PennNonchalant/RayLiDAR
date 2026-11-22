#include "Report.h"

#include "rt/ASBuilder.h"
#include "rt/GaussianScene.h"
#include "rt/MeshScene.h"
#include "simulation/PlyIO.h"
#include "simulation/SceneCoordinates.h"
#include "simulation/TriangleTrust.h"
#include "vulkan/CommandContext.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
namespace kitti = raylidar::kitti;
namespace rt = raylidar::rt;
namespace sim = raylidar::simulation;
namespace vulkan = raylidar::vulkan;

using Clock = std::chrono::steady_clock;

enum class Mode
{
    Mesh,
    Gaussian,
    Hybrid,
    Compare,
};

struct Options
{
    Mode RunMode = Mode::Mesh;
    fs::path MeshShader;
    fs::path GaussianShader;
    fs::path MeshBundle;
    fs::path GaussianBundle;
    fs::path OutputDirectory;
    float SigmaCutoff = 0.0f;
    float AlphaThreshold = 0.0f;
    rt::GaussianTraceOptions GaussianOptions;
    std::uint32_t BlasGroupSizeLog2 = rt::kDefaultGaussianBlasGroupSizeLog2;
    std::size_t GaussianChunkSize = 0;
    bool GpuRayGen = true;
    bool PruneGaussians = true;
    bool RouteUntrusted = false;
};

void PrintUsage(std::ostream &out)
{
    out << "Usage: raylidar_kitti_eval --mesh <LidarTrace.spv> <bundle_dir> [output_dir] [options]\n"
        << "       raylidar_kitti_eval --gaussian <GaussianTrace.spv> <bundle_dir> <sigma_cutoff> <alpha_threshold>"
           " [output_dir] [options]\n"
        << "       raylidar_kitti_eval <--hybrid|--compare> <LidarTrace.spv> <GaussianTrace.spv> <mesh_bundle>"
           " <gaussian_bundle> <sigma_cutoff> <alpha_threshold> [output_dir] [options]\n"
        << "  --ray-gen <cpu|gpu>             sweep ray generation (default gpu)\n"
        << "                                 gpu uses LidarRayGen.spv beside the first trace shader\n"
        << "Gaussian options (--gaussian, --hybrid, --compare):\n"
        << "  --hit-batch-size <K>            batch size (default " << rt::kDefaultGaussianHitBatchSize << ")\n"
        << "  --gaussian-blas <log2|single>   single (default) or 0 for one BLAS, positive log2 for Morton groups\n"
        << "  --gaussian-chunk-size <log2>    0 for all rays (default), positive log2 for 2^log2 rays per chunk\n"
        << "  --gaussian-prune <on|off>       drop weak/oversized splats at load time (default on)\n"
        << "  --candidate-alpha-bound <on|off> tighten using a single candidate (default on)\n"
        << "  --prefix-alpha-bound <on|off>    tighten using a candidate prefix (default on)\n"
        << "  --route-untrusted <on|off>      hybrid: also send rays that hit untrusted triangles to the Gaussians"
           " (default off)\n";
}

std::optional<Options> ParseOptions(std::span<char *> args)
{
    if (args.empty())
    {
        return std::nullopt;
    }

    const std::string_view mode = args.front();
    Options options;
    std::size_t next = 0;
    if (mode == "--mesh" && args.size() >= 3)
    {
        options.MeshShader = args[1];
        options.MeshBundle = args[2];
        next = 3;
    }
    else if (mode == "--gaussian" && args.size() >= 5)
    {
        options.RunMode = Mode::Gaussian;
        options.GaussianShader = args[1];
        options.GaussianBundle = args[2];
        options.SigmaCutoff = std::stof(args[3]);
        options.AlphaThreshold = std::stof(args[4]);
        next = 5;
    }
    else if ((mode == "--hybrid" || mode == "--compare") && args.size() >= 7)
    {
        options.RunMode = mode == "--hybrid" ? Mode::Hybrid : Mode::Compare;
        options.MeshShader = args[1];
        options.GaussianShader = args[2];
        options.MeshBundle = args[3];
        options.GaussianBundle = args[4];
        options.SigmaCutoff = std::stof(args[5]);
        options.AlphaThreshold = std::stof(args[6]);
        next = 7;
    }
    else
    {
        return std::nullopt;
    }

    if (next < args.size() && !std::string_view(args[next]).starts_with("--"))
    {
        options.OutputDirectory = args[next++];
    }
    for (; next + 1 < args.size(); next += 2)
    {
        const std::string_view flag = args[next];
        const std::string_view value = args[next + 1];
        if (flag == "--ray-gen" && (value == "cpu" || value == "gpu"))
        {
            options.GpuRayGen = value == "gpu";
        }
        else if (options.RunMode == Mode::Mesh)
        {
            return std::nullopt;
        }
        else if (flag == "--hit-batch-size")
        {
            options.GaussianOptions.HitBatchSize = static_cast<std::uint32_t>(std::stoul(args[next + 1]));
        }
        else if (flag == "--gaussian-blas")
        {
            options.BlasGroupSizeLog2 = value == "single" ? 0u : static_cast<std::uint32_t>(std::stoul(args[next + 1]));
        }
        else if (flag == "--gaussian-chunk-size")
        {
            std::size_t log2 = 0;
            const auto end = value.data() + value.size();
            const auto [parsedEnd, error] = std::from_chars(value.data(), end, log2);
            if (error != std::errc{} || parsedEnd != end || log2 >= std::numeric_limits<std::size_t>::digits)
            {
                return std::nullopt;
            }
            options.GaussianChunkSize = log2 == 0 ? 0 : std::size_t{1} << log2;
        }
        else if (flag == "--gaussian-prune" && (value == "on" || value == "off"))
        {
            options.PruneGaussians = value == "on";
        }
        else if (flag == "--route-untrusted" && (value == "on" || value == "off"))
        {
            options.RouteUntrusted = value == "on";
        }
        else if (flag == "--candidate-alpha-bound" && (value == "on" || value == "off"))
        {
            options.GaussianOptions.CandidateAlphaBound = value == "on";
        }
        else if (flag == "--prefix-alpha-bound" && (value == "on" || value == "off"))
        {
            options.GaussianOptions.PrefixAlphaBound = value == "on";
        }
        else
        {
            return std::nullopt;
        }
    }

    if (next != args.size())
    {
        return std::nullopt;
    }
    return options;
}

void PrintGaussianConfiguration(const rt::GaussianScene &scene, const Options &options)
{
    std::cout << "Gaussian BLASes:    " << scene.BlasCount() << " (up to " << scene.GaussiansPerBatch()
              << " splats each)\n"
              << "Gaussian trace:     K=" << options.GaussianOptions.HitBatchSize
              << ", candidate alpha bound=" << (options.GaussianOptions.CandidateAlphaBound ? "on" : "off")
              << ", prefix alpha bound=" << (options.GaussianOptions.PrefixAlphaBound ? "on" : "off") << '\n'
              << "Gaussian chunk size: "
              << (options.GaussianChunkSize == 0 ? "all" : std::to_string(options.GaussianChunkSize)) << '\n';
}

void RunEvaluation(const Options &options)
{
    const bool useMesh = options.RunMode != Mode::Gaussian;
    const bool useGaussian = options.RunMode != Mode::Mesh;
    const bool compare = options.RunMode == Mode::Compare;
    const auto &bundle = useMesh ? options.MeshBundle : options.GaussianBundle;
    const auto loadStart = Clock::now();
    const auto poses = sim::LoadVelodynePoses(bundle / "camera" / "camera_params.json");
    const auto mesh = useMesh ? sim::PlyIO::LoadMesh(options.MeshBundle / "mesh" / "mesh.ply") : sim::MeshData{};
    const auto origin = [&] {
        if (useMesh)
        {
            return sim::BoundingBoxCenter(mesh.Positions);
        }
        const auto &pose = poses.front().WorldFromVelodyne;
        return std::array{pose[0][3], pose[1][3], pose[2][3]};
    }();
    const auto positions = sim::ToLocalFloat(mesh.Positions, origin);
    sim::GaussianLoadStats gaussianStats;
    const auto gaussians = useGaussian
                               ? sim::PlyIO::LoadGaussians(options.GaussianBundle / "gs" / "2dgs" / "point_cloud.ply",
                                                           origin, options.PruneGaussians, &gaussianStats)
                               : std::vector<rt::Gaussian>{};
    const auto gt = sim::LoadGtRays(bundle / "gt_rays" / "gt_rays.bin", origin);
    const auto loadSeconds = std::chrono::duration<double>(Clock::now() - loadStart).count();

    std::cout << std::fixed << std::setprecision(3);
    if (useMesh)
    {
        std::cout << "Mesh: " << positions.size() << " vertices, " << mesh.Indices.size() / 3 << " triangles\n";
    }
    if (useGaussian)
    {
        std::cout << "2D Gaussian: " << gaussians.size() << " splats\n"
                  << "Gaussian prune: " << (options.PruneGaussians ? "on" : "off")
                  << " (opacity >= " << sim::kGaussianPruneOpacityMin
                  << ", max_scale <= " << sim::kGaussianPruneMaxScale << ")\n";
        if (options.PruneGaussians)
        {
            std::cout << "Gaussian load: input " << gaussianStats.Input << ", kept " << gaussianStats.Kept
                      << ", removed opacity " << gaussianStats.RemovedOpacity << ", removed scale "
                      << gaussianStats.RemovedScale << '\n';
        }
    }
    std::cout << "Local origin (W):   " << origin[0] << ", " << origin[1] << ", " << origin[2] << '\n'
              << "Loaded in " << loadSeconds << " s\n"
              << "Sweep ray generation: " << (options.GpuRayGen ? "GPU" : "CPU") << '\n';

    sim::TriangleTrustResult trust;
    if (useMesh && useGaussian)
    {
        const sim::TriangleTrustSettings settings;
        const auto trustStart = Clock::now();
        trust = sim::FindUntrustedTriangles(positions, mesh.Indices, gaussians, settings);
        const auto seconds = std::chrono::duration<double>(Clock::now() - trustStart).count();
        std::cout << "Untrusted triangles (> " << settings.MaxPlaneDistance << " m from the 2D Gaussian surface): "
                  << 100.0 * static_cast<double>(trust.UntrustedCount) / static_cast<double>(trust.Untrusted.size())
                  << " % (" << trust.NoNearbySplatCount << " with no splat nearby), computed in " << seconds << " s\n"
                  << "Hybrid routes:      mesh misses" << (options.RouteUntrusted ? " + untrusted triangles" : "")
                  << '\n';
    }
    // Empty: the hybrid sends only mesh misses to the Gaussians.
    const auto routedTriangles =
        options.RouteUntrusted ? std::span<const std::uint8_t>(trust.Untrusted) : std::span<const std::uint8_t>();

    vulkan::VulkanContext context;
    std::cout << "Device:             " << context.PhysicalDeviceProperties().deviceName.data() << '\n';
    std::optional<rt::MeshScene> meshScene;
    std::optional<rt::GaussianScene> gaussianScene;
    const auto buildStart = Clock::now();
    {
        rt::ASBuilder builder(context);
        vulkan::CommandContext commands(context);
        const auto &commandBuffer = commands.Begin();
        if (useMesh)
        {
            const rt::MeshDescription meshDescription{positions, mesh.Indices};
            const rt::MeshInstanceDescription instance{};
            meshScene.emplace(context,
                              rt::MeshSceneDescription{std::span(&meshDescription, 1), std::span(&instance, 1)});
            meshScene->RecordBuild(commandBuffer, builder);
        }
        if (useGaussian)
        {
            gaussianScene.emplace(context, rt::GaussianSceneDescription{
                                               .Gaussians = gaussians,
                                               .SigmaCutoff = options.SigmaCutoff,
                                               .BlasGroupSizeLog2 = options.BlasGroupSizeLog2,
                                           });
            gaussianScene->RecordBuild(commandBuffer, builder);
        }
        commands.End();
        commands.SubmitAndWait();
    }

    const sim::LidarSensor sensor(sim::LidarSensorDescription{});
    const auto capacity = static_cast<std::uint32_t>(std::max<std::size_t>(gt.Rays.size(), sensor.RayCount()));
    const auto &firstShader = useMesh ? options.MeshShader : options.GaussianShader;
    const auto rayGenShader = firstShader.parent_path() / "LidarRayGen.spv";
    std::optional<sim::LidarSimulator> meshSimulator;
    std::optional<sim::LidarSimulator> gaussianSimulator;
    if (meshScene)
    {
        meshScene->ReleaseUploadResources();
        meshSimulator.emplace(context, options.MeshShader, capacity);
        if (options.GpuRayGen)
        {
            meshSimulator->EnableGpuRayGeneration(rayGenShader, sensor);
        }
    }
    if (gaussianScene)
    {
        gaussianScene->ReleaseUploadResources();
        gaussianSimulator.emplace(context, options.GaussianShader, capacity, rt::RayQueryMode::Gaussian,
                                  options.GaussianOptions);
        if (options.GpuRayGen)
        {
            gaussianSimulator->EnableGpuRayGeneration(rayGenShader, sensor);
        }
        PrintGaussianConfiguration(*gaussianScene, options);
    }
    const auto buildMs = std::chrono::duration<double, std::milli>(Clock::now() - buildStart).count();
    std::cout << "Scene and pipeline setup: " << buildMs << " ms\n";

    if (!options.OutputDirectory.empty())
    {
        fs::create_directories(options.OutputDirectory);
    }

    std::optional<sim::HybridScanner> hybrid;
    if (meshSimulator && gaussianSimulator)
    {
        hybrid.emplace(*meshSimulator, *gaussianSimulator, options.GaussianChunkSize);
    }

    const auto traceMesh = [&](std::span<const rt::Ray> rays) { return meshSimulator->Scan(*meshScene, rays); };
    const auto traceGaussian = [&](std::span<const rt::Ray> rays) {
        return sim::ScanGaussians(*gaussianSimulator, *gaussianScene, rays, options.AlphaThreshold,
                                  options.GaussianChunkSize);
    };
    const auto traceHybrid = [&](std::span<const rt::Ray> rays) {
        return hybrid->Scan(*meshScene, *gaussianScene, rays, options.AlphaThreshold, routedTriangles);
    };

    if (compare)
    {
        const auto meshResult = traceMesh(gt.Rays);
        const auto gaussianResult = traceGaussian(gt.Rays);
        const auto hybridResult = traceHybrid(gt.Rays);
        kitti::PrintComparison(gt, meshResult, gaussianResult, hybridResult, trust.Untrusted);
    }
    else if (options.RunMode == Mode::Hybrid)
    {
        const auto result = traceHybrid(gt.Rays);
        kitti::EvaluateGtRays(gt, result.Result, "Hybrid", origin, options.OutputDirectory);
        kitti::PrintHybridRouting(gt, result);
    }
    else
    {
        const auto result = useMesh ? traceMesh(gt.Rays) : traceGaussian(gt.Rays);
        kitti::EvaluateGtRays(gt, result, useMesh ? "Mesh" : "2D Gaussian", origin, options.OutputDirectory);
    }

    const auto runSweeps = [&](std::string_view label, std::string_view prefix, const kitti::TraceFrame &trace) {
        std::cout << "\n--- " << label << " ---\n";
        kitti::RunVelodyneScans(sensor, poses, origin, options.OutputDirectory, prefix, trust.Untrusted,
                                options.GpuRayGen, trace);
    };

    if (compare || options.RunMode == Mode::Mesh)
    {
        runSweeps("Mesh only", compare ? "mesh_" : (options.GpuRayGen ? "velodyne_gpu_" : "velodyne_"),
                  [&](const sim::Transform &sensorToWorld, std::span<const rt::Ray> rays) {
                      auto result =
                          options.GpuRayGen ? meshSimulator->Scan(*meshScene, sensor, sensorToWorld) : traceMesh(rays);
                      auto sources = sim::SourcesFor(result.Hits, sim::HitSource::Mesh);
                      return kitti::FrameScan{std::move(result), std::move(sources)};
                  });
    }
    if (compare || options.RunMode == Mode::Gaussian)
    {
        runSweeps("Gaussian only", compare ? "gaussian_" : "velodyne_gaussian_",
                  [&](const sim::Transform &sensorToWorld, std::span<const rt::Ray> rays) {
                      auto result =
                          options.GpuRayGen
                              ? sim::ScanGaussians(*gaussianSimulator, *gaussianScene, {sensor, sensorToWorld},
                                                   options.AlphaThreshold, options.GaussianChunkSize)
                              : traceGaussian(rays);
                      auto sources = sim::SourcesFor(result.Hits, sim::HitSource::Gaussian);
                      return kitti::FrameScan{std::move(result), std::move(sources)};
                  });
    }
    if (hybrid)
    {
        std::size_t routedRays = 0;
        runSweeps("Hybrid", "hybrid_", [&](const sim::Transform &sensorToWorld, std::span<const rt::Ray> rays) {
            auto result = options.GpuRayGen ? hybrid->Scan(*meshScene, *gaussianScene, sensor, sensorToWorld,
                                                           options.AlphaThreshold, routedTriangles)
                                            : traceHybrid(rays);
            routedRays += result.RaysSentToGaussian;
            return kitti::FrameScan{std::move(result.Result), std::move(result.Sources)};
        });

        const auto rayCount = static_cast<double>(poses.size()) * sensor.RayCount();
        std::cout << "Hybrid HDL-64E: " << 100.0 * static_cast<double>(routedRays) / rayCount
                  << " % of rays sent to Gaussian\n";
    }

    if (!options.OutputDirectory.empty())
    {
        std::cout << "\nWrote .rlf frames to " << options.OutputDirectory.string() << '\n';
    }
}
} // namespace

int main(int argc, char *argv[])
{
    try
    {
        if (argc == 2 && std::string_view(argv[1]) == "--help")
        {
            PrintUsage(std::cout);
            return EXIT_SUCCESS;
        }

        const auto options = ParseOptions(std::span(argv + 1, static_cast<std::size_t>(argc - 1)));
        if (!options)
        {
            PrintUsage(std::cerr);
            return EXIT_FAILURE;
        }

        RunEvaluation(*options);
        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "KITTI evaluation failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
