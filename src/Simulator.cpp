#include <raylidar/Simulator.h>

#include "rt/ASBuilder.h"
#include "rt/MeshScene.h"
#include "simulation/KittiBundle.h"
#include "simulation/LidarSensor.h"
#include "simulation/LidarSimulator.h"
#include "simulation/MeshIO.h"
#include "vulkan/CommandContext.h"
#include "vulkan/VulkanContext.h"

#include <chrono>
#include <optional>
#include <span>

namespace raylidar
{
namespace
{
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
} // namespace

struct Simulator::Impl
{
    Impl(const std::filesystem::path &meshPath, const std::filesystem::path &traceShaderPath,
         const SensorConfig &config)
        : Sensor(ToSensorDescription(config))
    {
        const auto mesh = simulation::LoadPlyMesh(meshPath);

        // Shift into a local frame so float positions keep sub-millimetre precision far from the world origin.
        Origin = simulation::BoundingBoxCenter(mesh.Positions);
        const auto positions = simulation::ToLocalFloat(mesh.Positions, Origin);

        const rt::MeshDescription meshDescription{positions, mesh.Indices};
        const rt::MeshInstanceDescription instance{};

        rt::MeshSceneDescription sceneDescription;
        sceneDescription.Meshes = std::span(&meshDescription, 1);
        sceneDescription.Instances = std::span(&instance, 1);

        Scene.emplace(Context, sceneDescription);
        {
            rt::ASBuilder builder(Context);
            vulkan::CommandContext commands(Context);
            Scene->RecordBuild(commands.Begin(), builder);
            commands.End();
            commands.SubmitAndWait();
            Scene->ReleaseUploadResources();
        }

        Lidar.emplace(Context, traceShaderPath, Sensor.RayCount());
    }

    vulkan::VulkanContext Context;
    simulation::LidarSensor Sensor;
    std::array<double, 3> Origin{};

    std::optional<rt::MeshScene> Scene;
    std::optional<simulation::LidarSimulator> Lidar;
};

Simulator::Simulator(const std::filesystem::path &meshPath, const std::filesystem::path &traceShaderPath,
                     const SensorConfig &sensor)
    : m_impl(std::make_unique<Impl>(meshPath, traceShaderPath, sensor))
{
}

Simulator::~Simulator() = default;

Simulator::Simulator(Simulator &&) noexcept = default;

Simulator &Simulator::operator=(Simulator &&) noexcept = default;

ScanOutput Simulator::Scan(const Pose &worldFromSensor)
{
    using Clock = std::chrono::steady_clock;

    const auto generateStart = Clock::now();
    const auto rays = m_impl->Sensor.GenerateRays(simulation::ToLocalTransform(worldFromSensor, m_impl->Origin));
    const double generate = std::chrono::duration<double, std::milli>(Clock::now() - generateStart).count();

    const auto result = m_impl->Lidar->Scan(m_impl->Scene->Tlas(), rays);
    const auto &origin = m_impl->Origin;

    ScanOutput output;
    output.Points.reserve(result.Points.size());

    for (const auto &point : result.Points)
    {
        output.Points.push_back({point[0] + origin[0], point[1] + origin[1], point[2] + origin[2]});
    }

    output.Ranges.reserve(result.Hits.size());

    for (const auto &hit : result.Hits)
    {
        output.Ranges.push_back(hit.Valid != 0 ? hit.T : -1.0f);
    }

    output.Timings.Generate = generate;
    output.Timings.Upload = result.Timings.Upload;
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
} // namespace raylidar
