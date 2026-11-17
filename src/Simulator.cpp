#include <raylidar/Simulator.h>

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

#include <chrono>
#include <optional>
#include <span>
#include <stdexcept>

namespace raylidar
{
namespace
{
using Clock = std::chrono::steady_clock;

// The public enums mirror the internal ones value for value, so they convert with a cast.
static_assert(static_cast<int>(HitSource::Gaussian) == static_cast<int>(simulation::HitSource::Gaussian));
static_assert(static_cast<int>(SurfaceTrust::Untrusted) == static_cast<int>(simulation::SurfaceTrust::Untrusted));

simulation::LidarSensorDescription ToSensorDescription(const SensorConfig &config)
{
    simulation::LidarSensorDescription description;
    description.BeamCount = config.BeamCount;
    description.AzimuthCount = config.AzimuthCount;
    description.ElevationMinDegrees = config.ElevationMinDegrees;
    description.ElevationMaxDegrees = config.ElevationMaxDegrees;
    description.RangeMin = config.RangeMin;
    description.RangeMax = config.RangeMax;
    return description;
}

bool NeedsMesh(SceneMode mode)
{
    return mode == SceneMode::Mesh || mode == SceneMode::Hybrid;
}

bool NeedsGaussians(SceneMode mode)
{
    return mode == SceneMode::Gaussian || mode == SceneMode::Hybrid;
}

// Center of the splat centers, for a Gaussian-only scene. Precision only needs to be good to a few metres here.
std::array<double, 3> GaussianCenter(const std::filesystem::path &path)
{
    const auto gaussians = simulation::PlyIO::LoadGaussians(path, {0.0, 0.0, 0.0});
    std::vector<std::array<double, 3>> centers;
    centers.reserve(gaussians.size());
    for (const auto &gaussian : gaussians)
    {
        centers.push_back({gaussian.Center[0], gaussian.Center[1], gaussian.Center[2]});
    }
    return simulation::BoundingBoxCenter(centers);
}
} // namespace

struct Simulator::Impl
{
    Impl(const SimulatorConfig &config, const std::filesystem::path &meshTraceShaderPath)
        : Mode(config.Mode), Sensor(ToSensorDescription(config.Sensor)), AlphaThreshold(config.AlphaThreshold)
    {
        if (NeedsMesh(Mode) && config.MeshPath.empty())
        {
            throw std::invalid_argument("Mesh and Hybrid modes need SimulatorConfig::MeshPath.");
        }
        if (NeedsGaussians(Mode) && config.GaussianPath.empty())
        {
            throw std::invalid_argument("Gaussian and Hybrid modes need SimulatorConfig::GaussianPath.");
        }

        const bool loadMesh = !config.MeshPath.empty() && Mode != SceneMode::Gaussian;
        const bool loadGaussians = !config.GaussianPath.empty();

        // Shift into a local frame so float positions keep sub-millimetre precision far from the world origin.
        simulation::MeshData mesh;
        std::vector<std::array<float, 3>> positions;
        if (loadMesh)
        {
            mesh = simulation::PlyIO::LoadMesh(config.MeshPath);
            Origin = simulation::BoundingBoxCenter(mesh.Positions);
            positions = simulation::ToLocalFloat(mesh.Positions, Origin);
        }
        else
        {
            Origin = GaussianCenter(config.GaussianPath);
        }

        std::vector<rt::Gaussian> gaussians;
        if (loadGaussians)
        {
            gaussians = simulation::PlyIO::LoadGaussians(config.GaussianPath, Origin);
        }

        // Ground-truth-free trust labels: where the mesh and the Gaussians disagree.
        if (loadMesh && loadGaussians)
        {
            UntrustedTriangles = simulation::FindUntrustedTriangles(positions, mesh.Indices, gaussians).Untrusted;
        }

        const rt::MeshDescription meshDescription{positions, mesh.Indices};
        const rt::MeshInstanceDescription instance{};
        if (loadMesh)
        {
            rt::MeshSceneDescription sceneDescription;
            sceneDescription.Meshes = std::span(&meshDescription, 1);
            sceneDescription.Instances = std::span(&instance, 1);
            MeshWorld.emplace(Context, sceneDescription);
        }

        if (NeedsGaussians(Mode))
        {
            const rt::GaussianSceneDescription description{
                .Gaussians = gaussians,
                .SigmaCutoff = config.SigmaCutoff,
            };
            GaussianWorld.emplace(Context, description);
        }

        {
            rt::ASBuilder builder(Context);
            vulkan::CommandContext commands(Context);
            const auto &commandBuffer = commands.Begin();
            if (MeshWorld)
            {
                MeshWorld->RecordBuild(commandBuffer, builder);
            }
            if (GaussianWorld)
            {
                GaussianWorld->RecordBuild(commandBuffer, builder);
            }
            commands.End();
            commands.SubmitAndWait();
            if (MeshWorld)
            {
                MeshWorld->ReleaseUploadResources();
            }
            if (GaussianWorld)
            {
                GaussianWorld->ReleaseUploadResources();
            }
        }

        const auto capacity = Sensor.RayCount();
        if (NeedsMesh(Mode))
        {
            MeshLidar.emplace(Context, meshTraceShaderPath, capacity);

            const auto rayGenShaderPath = meshTraceShaderPath.parent_path() / "LidarRayGen.spv";
            if (Mode == SceneMode::Mesh && std::filesystem::exists(rayGenShaderPath))
            {
                MeshLidar->EnableGpuRayGeneration(rayGenShaderPath);
                GpuRayGeneration = true;
            }
        }
        if (NeedsGaussians(Mode))
        {
            GaussianLidar.emplace(Context, config.ShaderDirectory / "GaussianTraceK16.spv", capacity,
                                  rt::RayQueryMode::Gaussian);
        }
        if (Mode == SceneMode::Hybrid)
        {
            Hybrid.emplace(*MeshLidar, *GaussianLidar);
        }
    }

    SceneMode Mode;
    vulkan::VulkanContext Context;
    simulation::LidarSensor Sensor;
    float AlphaThreshold;
    std::array<double, 3> Origin{};
    std::vector<std::uint8_t> UntrustedTriangles;
    bool GpuRayGeneration = false;

    std::optional<rt::MeshScene> MeshWorld;
    std::optional<rt::GaussianScene> GaussianWorld;
    std::optional<simulation::LidarSimulator> MeshLidar;
    std::optional<simulation::LidarSimulator> GaussianLidar;
    std::optional<simulation::HybridScanner> Hybrid; // refers to both simulators, so it is destroyed first
};

Simulator::Simulator(const SimulatorConfig &config)
    : m_impl(std::make_unique<Impl>(config, config.ShaderDirectory / "LidarTrace.spv"))
{
}

Simulator::Simulator(const std::filesystem::path &meshPath, const std::filesystem::path &traceShaderPath,
                     const SensorConfig &sensor)
{
    SimulatorConfig config;
    config.Mode = SceneMode::Mesh;
    config.MeshPath = meshPath;
    config.ShaderDirectory = traceShaderPath.parent_path();
    config.Sensor = sensor;
    m_impl = std::make_unique<Impl>(config, traceShaderPath);
}

Simulator::~Simulator() = default;

Simulator::Simulator(Simulator &&) noexcept = default;

Simulator &Simulator::operator=(Simulator &&) noexcept = default;

ScanOutput Simulator::Scan(const Pose &worldFromSensor)
{
    auto &impl = *m_impl;
    const auto transform = simulation::ToLocalTransform(worldFromSensor, impl.Origin);

    double generate = 0.0;
    const auto generateRays = [&impl, &transform, &generate] {
        const auto start = Clock::now();
        auto rays = impl.Sensor.GenerateRays(transform);
        generate = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return rays;
    };

    simulation::ScanResult result;
    std::vector<simulation::HitSource> sources;

    switch (impl.Mode)
    {
    case SceneMode::Mesh:
        if (impl.GpuRayGeneration)
        {
            result = impl.MeshLidar->Scan(*impl.MeshWorld, impl.Sensor, transform);
        }
        else
        {
            result = impl.MeshLidar->Scan(*impl.MeshWorld, generateRays());
        }
        sources = simulation::SourcesFor(result.Hits, simulation::HitSource::Mesh);
        break;
    case SceneMode::Gaussian:
        result = simulation::ScanGaussiansInChunks(*impl.GaussianLidar, *impl.GaussianWorld, generateRays(),
                                                   impl.AlphaThreshold);
        sources = simulation::SourcesFor(result.Hits, simulation::HitSource::Gaussian);
        break;
    case SceneMode::Hybrid: {
        auto hybrid = impl.Hybrid->Scan(*impl.MeshWorld, *impl.GaussianWorld, generateRays(), impl.AlphaThreshold);
        result = std::move(hybrid.Result);
        sources = std::move(hybrid.Sources);
        break;
    }
    }

    auto frame = simulation::MakeLidarFrame(impl.Sensor, simulation::VelodynePose{0, worldFromSensor}, result.Hits,
                                            sources, impl.UntrustedTriangles);

    ScanOutput output;
    output.WorldFromSensor = worldFromSensor;
    output.BeamCount = frame.BeamCount;
    output.AzimuthCount = frame.AzimuthCount;

    const auto &origin = impl.Origin;
    output.Points.reserve(result.Points.size());
    for (const auto &point : result.Points)
    {
        output.Points.push_back({point[0] + origin[0], point[1] + origin[1], point[2] + origin[2]});
    }

    output.Ranges = std::move(frame.Ranges);
    output.SensorPoints = std::move(frame.Points);
    output.Sources.reserve(frame.Sources.size());
    for (const auto source : frame.Sources)
    {
        output.Sources.push_back(static_cast<HitSource>(source));
    }
    output.Trust.reserve(frame.Trust.size());
    for (const auto trust : frame.Trust)
    {
        output.Trust.push_back(static_cast<SurfaceTrust>(trust));
    }

    output.Timings.Generate = generate;
    output.Timings.Upload = result.Timings.Upload;
    output.Timings.RayGen = result.Timings.RayGen;
    output.Timings.Trace = result.Timings.Trace;
    output.Timings.CopyHits = result.Timings.CopyHits;
    output.Timings.SubmitWait = result.Timings.SubmitWait;
    output.Timings.Readback = result.Timings.Readback;
    output.Timings.Total = generate + result.Timings.Total;

    return output;
}

std::uint32_t Simulator::RayCount() const noexcept
{
    return m_impl->Sensor.RayCount();
}

SceneMode Simulator::Mode() const noexcept
{
    return m_impl->Mode;
}

void Simulator::WriteFrame(const std::filesystem::path &path, const ScanOutput &output, std::uint32_t frameIndex)
{
    simulation::LidarFrame frame;
    frame.Frame = frameIndex;
    frame.BeamCount = output.BeamCount;
    frame.AzimuthCount = output.AzimuthCount;
    frame.WorldFromSensor = output.WorldFromSensor;
    frame.Ranges = output.Ranges;
    frame.Points = output.SensorPoints;
    frame.Sources.reserve(output.Sources.size());
    for (const auto source : output.Sources)
    {
        frame.Sources.push_back(static_cast<simulation::HitSource>(source));
    }
    frame.Trust.reserve(output.Trust.size());
    for (const auto trust : output.Trust)
    {
        frame.Trust.push_back(static_cast<simulation::SurfaceTrust>(trust));
    }
    simulation::WriteLidarFrame(path, frame);
}
} // namespace raylidar
