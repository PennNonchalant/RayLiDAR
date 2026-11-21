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
#include <cstddef>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace raylidar
{
namespace
{
using Clock = std::chrono::steady_clock;

constexpr const char *kMeshTraceShader = "LidarTrace.spv";
constexpr const char *kRayGenShader = "LidarRayGen.spv";
constexpr const char *kGaussianTraceShader = "GaussianTrace.spv";
constexpr std::size_t kGaussianRaysPerChunk = 32768;

double MillisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// ---------------------------------------------------------------------------------------------------------------------
// Public <-> internal types. The public enums mirror the internal ones value for value, so they convert with a cast.

template <typename A, typename B> constexpr bool SameValue(A a, B b)
{
    return static_cast<int>(a) == static_cast<int>(b);
}

static_assert(std::is_same_v<std::underlying_type_t<HitSource>, std::underlying_type_t<simulation::HitSource>>);
static_assert(SameValue(HitSource::None, simulation::HitSource::None) &&
              SameValue(HitSource::Mesh, simulation::HitSource::Mesh) &&
              SameValue(HitSource::Gaussian, simulation::HitSource::Gaussian));

static_assert(std::is_same_v<std::underlying_type_t<SurfaceTrust>, std::underlying_type_t<simulation::SurfaceTrust>>);
static_assert(SameValue(SurfaceTrust::Unknown, simulation::SurfaceTrust::Unknown) &&
              SameValue(SurfaceTrust::Trusted, simulation::SurfaceTrust::Trusted) &&
              SameValue(SurfaceTrust::Untrusted, simulation::SurfaceTrust::Untrusted));

template <typename To, typename From> std::vector<To> CastEach(const std::vector<From> &values)
{
    std::vector<To> result;
    result.reserve(values.size());
    for (const auto value : values)
    {
        result.push_back(static_cast<To>(value));
    }
    return result;
}

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

ScanTimings ToPublicTimings(const simulation::ScanTimings &timings, double generateMilliseconds)
{
    ScanTimings result;
    result.Generate = generateMilliseconds;
    result.Upload = timings.Upload;
    result.RayGen = timings.RayGen;
    result.Trace = timings.Trace;
    result.CopyHits = timings.CopyHits;
    result.SubmitWait = timings.SubmitWait;
    result.Readback = timings.Readback;
    result.Routing = timings.Routing;
    return result; // Total is measured by the caller around the whole scan.
}

bool NeedsMesh(SceneMode mode)
{
    return mode == SceneMode::Mesh || mode == SceneMode::Hybrid;
}

bool NeedsGaussians(SceneMode mode)
{
    return mode == SceneMode::Gaussian || mode == SceneMode::Hybrid;
}

struct ShaderPaths
{
    std::filesystem::path MeshTrace;     // empty when no mesh is traced
    std::filesystem::path GaussianTrace; // empty when no Gaussians are traced
    std::filesystem::path RayGen;        // empty when rays are generated on the CPU
};

ShaderPaths FindShaders(const SimulatorConfig &config, const std::filesystem::path &meshTraceShader)
{
    ShaderPaths shaders;
    if (NeedsMesh(config.Mode))
    {
        shaders.MeshTrace = meshTraceShader;
    }
    if (NeedsGaussians(config.Mode))
    {
        shaders.GaussianTrace = config.ShaderDirectory / kGaussianTraceShader;
    }

    // Optional: without this shader, Mesh mode generates rays on the CPU.
    const auto rayGenShader = meshTraceShader.parent_path() / kRayGenShader;
    if (config.Mode == SceneMode::Mesh && config.PreferGpuRayGeneration &&
        std::filesystem::is_regular_file(rayGenShader))
    {
        shaders.RayGen = rayGenShader;
    }
    return shaders;
}

// ---------------------------------------------------------------------------------------------------------------------
// Scene loading.

// CPU copy of the scene, shifted so that the scene centre is at the origin: float positions then keep sub-millimetre
// precision even when the world coordinates are kilometres from zero. Only needed until the GPU scenes are built.
struct LocalScene
{
    std::array<double, 3> Origin{};
    std::vector<std::array<float, 3>> MeshPositions;
    std::vector<std::uint32_t> MeshIndices;
    std::vector<rt::Gaussian> Gaussians;
};

// Splat centres are float32 in the PLY, so reading them unshifted and shifting afterwards gives exactly the same
// floats as reading them relative to the origin.
std::array<double, 3> GaussianBoundsCenter(std::span<const rt::Gaussian> gaussians)
{
    std::vector<std::array<double, 3>> centers;
    centers.reserve(gaussians.size());
    for (const auto &gaussian : gaussians)
    {
        centers.push_back({gaussian.Center[0], gaussian.Center[1], gaussian.Center[2]});
    }
    return simulation::BoundingBoxCenter(centers);
}

void ShiftGaussians(std::vector<rt::Gaussian> &gaussians, const std::array<double, 3> &origin)
{
    for (auto &gaussian : gaussians)
    {
        for (std::size_t axis = 0; axis < gaussian.Center.size(); ++axis)
        {
            gaussian.Center[axis] = static_cast<float>(gaussian.Center[axis] - origin[axis]);
        }
    }
}

LocalScene LoadLocalScene(const SimulatorConfig &config)
{
    LocalScene scene;

    if (!config.GaussianPath.empty())
    {
        scene.Gaussians = simulation::PlyIO::LoadGaussians(config.GaussianPath, {0.0, 0.0, 0.0});
    }

    if (NeedsMesh(config.Mode))
    {
        auto mesh = simulation::PlyIO::LoadMesh(config.MeshPath);
        scene.Origin = simulation::BoundingBoxCenter(mesh.Positions);
        scene.MeshPositions = simulation::ToLocalFloat(mesh.Positions, scene.Origin);
        scene.MeshIndices = std::move(mesh.Indices);
    }
    else
    {
        scene.Origin = GaussianBoundsCenter(scene.Gaussians);
    }

    ShiftGaussians(scene.Gaussians, scene.Origin);
    return scene;
}

// Builds every acceleration structure in one submission, then frees the upload buffers.
void BuildAccelerationStructures(const vulkan::VulkanContext &context, std::optional<rt::MeshScene> &mesh,
                                 std::optional<rt::GaussianScene> &gaussians)
{
    rt::ASBuilder builder(context);
    vulkan::CommandContext commands(context);
    const auto &commandBuffer = commands.Begin();
    if (mesh)
    {
        mesh->RecordBuild(commandBuffer, builder);
    }
    if (gaussians)
    {
        gaussians->RecordBuild(commandBuffer, builder);
    }
    commands.End();
    commands.SubmitAndWait();

    if (mesh)
    {
        mesh->ReleaseUploadResources();
    }
    if (gaussians)
    {
        gaussians->ReleaseUploadResources();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Scan backends: one per SceneMode. Adding a mode means adding a backend; Simulator itself does not change.

struct BackendScan
{
    simulation::ScanResult Result;
    std::vector<simulation::HitSource> Sources; // one per ray
    double GenerateMilliseconds = 0.0;          // CPU ray generation, 0 when the GPU generates the rays
};

class ScanBackend
{
  public:
    virtual ~ScanBackend() = default;

    ScanBackend(const ScanBackend &) = delete;
    ScanBackend &operator=(const ScanBackend &) = delete;
    ScanBackend(ScanBackend &&) = delete;
    ScanBackend &operator=(ScanBackend &&) = delete;

    [[nodiscard]]
    virtual BackendScan Scan(const simulation::Transform &sensorToWorld) = 0;

    [[nodiscard]]
    virtual bool UsesGpuRayGeneration() const noexcept
    {
        return false;
    }

  protected:
    ScanBackend() = default;
};

struct GeneratedRays
{
    std::vector<rt::Ray> Rays;
    double Milliseconds = 0.0;
};

GeneratedRays GenerateRays(const simulation::LidarSensor &sensor, const simulation::Transform &sensorToWorld)
{
    const auto start = Clock::now();
    GeneratedRays generated;
    generated.Rays = sensor.GenerateRays(sensorToWorld);
    generated.Milliseconds = MillisecondsSince(start);
    return generated;
}

class MeshBackend final : public ScanBackend
{
  public:
    MeshBackend(const vulkan::VulkanContext &context, const rt::MeshScene &scene, const simulation::LidarSensor &sensor,
                const ShaderPaths &shaders)
        : m_scene(scene), m_sensor(sensor), m_simulator(context, shaders.MeshTrace, sensor.RayCount())
    {
        if (!shaders.RayGen.empty())
        {
            m_simulator.EnableGpuRayGeneration(shaders.RayGen, sensor);
            m_gpuRayGeneration = true;
        }
    }

    BackendScan Scan(const simulation::Transform &sensorToWorld) override
    {
        BackendScan scan;
        if (m_gpuRayGeneration)
        {
            scan.Result = m_simulator.Scan(m_scene, m_sensor, sensorToWorld);
        }
        else
        {
            const auto generated = GenerateRays(m_sensor, sensorToWorld);
            scan.GenerateMilliseconds = generated.Milliseconds;
            scan.Result = m_simulator.Scan(m_scene, generated.Rays);
        }
        scan.Sources = simulation::SourcesFor(scan.Result.Hits, simulation::HitSource::Mesh);
        return scan;
    }

    bool UsesGpuRayGeneration() const noexcept override
    {
        return m_gpuRayGeneration;
    }

  private:
    const rt::MeshScene &m_scene;
    const simulation::LidarSensor &m_sensor;
    simulation::LidarSimulator m_simulator;
    bool m_gpuRayGeneration = false;
};

class GaussianBackend final : public ScanBackend
{
  public:
    GaussianBackend(const vulkan::VulkanContext &context, const rt::GaussianScene &scene,
                    const simulation::LidarSensor &sensor, const ShaderPaths &shaders, float alphaThreshold)
        : m_scene(scene), m_sensor(sensor),
          m_simulator(context, shaders.GaussianTrace, sensor.RayCount(), rt::RayQueryMode::Gaussian),
          m_alphaThreshold(alphaThreshold)
    {
    }

    BackendScan Scan(const simulation::Transform &sensorToWorld) override
    {
        const auto generated = GenerateRays(m_sensor, sensorToWorld);

        BackendScan scan;
        scan.GenerateMilliseconds = generated.Milliseconds;
        scan.Result =
            simulation::ScanGaussians(m_simulator, m_scene, generated.Rays, m_alphaThreshold, kGaussianRaysPerChunk);
        scan.Sources = simulation::SourcesFor(scan.Result.Hits, simulation::HitSource::Gaussian);
        return scan;
    }

  private:
    const rt::GaussianScene &m_scene;
    const simulation::LidarSensor &m_sensor;
    simulation::LidarSimulator m_simulator;
    float m_alphaThreshold;
};

// Mesh first; only the rays that miss the mesh go to the Gaussians (see HybridScanner for why untrusted triangles
// are not re-routed).
class HybridBackend final : public ScanBackend
{
  public:
    HybridBackend(const vulkan::VulkanContext &context, const rt::MeshScene &meshScene,
                  const rt::GaussianScene &gaussianScene, const simulation::LidarSensor &sensor,
                  const ShaderPaths &shaders, float alphaThreshold)
        : m_meshScene(meshScene), m_gaussianScene(gaussianScene), m_sensor(sensor),
          m_meshSimulator(context, shaders.MeshTrace, sensor.RayCount()),
          m_gaussianSimulator(context, shaders.GaussianTrace, sensor.RayCount(), rt::RayQueryMode::Gaussian),
          m_scanner(m_meshSimulator, m_gaussianSimulator, kGaussianRaysPerChunk), m_alphaThreshold(alphaThreshold)
    {
    }

    BackendScan Scan(const simulation::Transform &sensorToWorld) override
    {
        const auto generated = GenerateRays(m_sensor, sensorToWorld);
        auto hybrid = m_scanner.Scan(m_meshScene, m_gaussianScene, generated.Rays, m_alphaThreshold);

        BackendScan scan;
        scan.GenerateMilliseconds = generated.Milliseconds;
        scan.Result = std::move(hybrid.Result);
        scan.Sources = std::move(hybrid.Sources);
        return scan;
    }

  private:
    const rt::MeshScene &m_meshScene;
    const rt::GaussianScene &m_gaussianScene;
    const simulation::LidarSensor &m_sensor;
    simulation::LidarSimulator m_meshSimulator;
    simulation::LidarSimulator m_gaussianSimulator;
    simulation::HybridScanner m_scanner; // refers to both simulators above, so it is declared after them
    float m_alphaThreshold;
};

std::unique_ptr<ScanBackend> CreateBackend(SceneMode mode, const vulkan::VulkanContext &context,
                                           const std::optional<rt::MeshScene> &meshScene,
                                           const std::optional<rt::GaussianScene> &gaussianScene,
                                           const simulation::LidarSensor &sensor, const ShaderPaths &shaders,
                                           const GaussianTraceConfig &gaussian)
{
    if (mode == SceneMode::Mesh)
    {
        return std::make_unique<MeshBackend>(context, *meshScene, sensor, shaders);
    }
    if (mode == SceneMode::Gaussian)
    {
        return std::make_unique<GaussianBackend>(context, *gaussianScene, sensor, shaders, gaussian.AlphaThreshold);
    }
    return std::make_unique<HybridBackend>(context, *meshScene, *gaussianScene, sensor, shaders,
                                           gaussian.AlphaThreshold);
}

// ---------------------------------------------------------------------------------------------------------------------
// Output.

ScanOutput MakeScanOutput(const simulation::LidarSensor &sensor, const std::array<double, 3> &origin,
                          std::span<const std::uint8_t> untrustedTriangles, const Pose &worldFromSensor,
                          BackendScan scan)
{
    auto frame = simulation::MakeLidarFrame(sensor, simulation::VelodynePose{0, worldFromSensor}, scan.Result.Hits,
                                            scan.Sources, untrustedTriangles);

    ScanOutput output;
    output.Points.reserve(scan.Result.Points.size());
    for (const auto &point : scan.Result.Points)
    {
        output.Points.push_back({point[0] + origin[0], point[1] + origin[1], point[2] + origin[2]});
    }

    output.Ranges = std::move(frame.Ranges);
    output.SensorPoints = std::move(frame.Points);
    output.Sources = CastEach<HitSource>(frame.Sources);
    output.Trust = CastEach<SurfaceTrust>(frame.Trust);
    output.WorldFromSensor = worldFromSensor;
    output.BeamCount = frame.BeamCount;
    output.AzimuthCount = frame.AzimuthCount;
    output.Timings = ToPublicTimings(scan.Result.Timings, scan.GenerateMilliseconds);
    return output;
}
} // namespace

// -------------------------------------------------------------------------------------------------------------------------

struct Simulator::Impl
{
    Impl(const SimulatorConfig &config, const ShaderPaths &shaders)
        : Mode(config.Mode), Sensor(ToSensorDescription(config.Sensor))
    {
        const auto scene = LoadLocalScene(config);
        Origin = scene.Origin;

        const bool hasMesh = !scene.MeshIndices.empty();

        // Ground-truth-free trust labels: where the mesh and the Gaussians disagree.
        if (hasMesh && !scene.Gaussians.empty())
        {
            UntrustedTriangles =
                simulation::FindUntrustedTriangles(scene.MeshPositions, scene.MeshIndices, scene.Gaussians).Untrusted;
        }

        if (hasMesh)
        {
            const rt::MeshDescription mesh{scene.MeshPositions, scene.MeshIndices};
            const rt::MeshInstanceDescription instance{};
            rt::MeshSceneDescription description;
            description.Meshes = std::span(&mesh, 1);
            description.Instances = std::span(&instance, 1);
            MeshWorld.emplace(Context, description);
        }

        if (NeedsGaussians(Mode))
        {
            const rt::GaussianSceneDescription description{
                .Gaussians = scene.Gaussians,
                .SigmaCutoff = config.Gaussian.SigmaCutoff,
            };
            GaussianWorld.emplace(Context, description);
        }

        BuildAccelerationStructures(Context, MeshWorld, GaussianWorld);
        Backend = CreateBackend(Mode, Context, MeshWorld, GaussianWorld, Sensor, shaders, config.Gaussian);
    }

    SceneMode Mode;
    simulation::LidarSensor Sensor;
    std::array<double, 3> Origin{};
    std::vector<std::uint8_t> UntrustedTriangles; // one per mesh triangle; empty without Gaussians

    // Destroyed bottom-up: the backend before the scenes it traces, the scenes before the Vulkan context.
    vulkan::VulkanContext Context;
    std::optional<rt::MeshScene> MeshWorld;
    std::optional<rt::GaussianScene> GaussianWorld;
    std::unique_ptr<ScanBackend> Backend;
};

Simulator::Simulator(const SimulatorConfig &config)
{
    m_impl = std::make_unique<Impl>(config, FindShaders(config, config.ShaderDirectory / kMeshTraceShader));
}

Simulator::Simulator(const std::filesystem::path &meshPath, const std::filesystem::path &traceShaderPath,
                     const SensorConfig &sensor)
{
    SimulatorConfig config;
    config.Mode = SceneMode::Mesh;
    config.MeshPath = meshPath;
    config.ShaderDirectory = traceShaderPath.parent_path();
    config.Sensor = sensor;
    config.PreferGpuRayGeneration = false;
    m_impl = std::make_unique<Impl>(config, FindShaders(config, traceShaderPath));
}

Simulator::~Simulator() = default;

Simulator::Simulator(Simulator &&) noexcept = default;

Simulator &Simulator::operator=(Simulator &&) noexcept = default;

ScanOutput Simulator::Scan(const Pose &worldFromSensor)
{
    const auto start = Clock::now();
    auto &impl = *m_impl;
    const auto sensorToWorld = simulation::ToLocalTransform(worldFromSensor, impl.Origin);

    auto output = MakeScanOutput(impl.Sensor, impl.Origin, impl.UntrustedTriangles, worldFromSensor,
                                 impl.Backend->Scan(sensorToWorld));
    output.Timings.Total = MillisecondsSince(start);
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

bool Simulator::UsesGpuRayGeneration() const noexcept
{
    return m_impl->Backend->UsesGpuRayGeneration();
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
    frame.Sources = CastEach<simulation::HitSource>(output.Sources);
    frame.Trust = CastEach<simulation::SurfaceTrust>(output.Trust);
    simulation::WriteLidarFrame(path, frame); // checks that every per-ray array has one entry per ray
}
} // namespace raylidar
