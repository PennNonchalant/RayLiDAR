#pragma once

#include "GpuRayGenerator.h"
#include "LidarSensor.h"
#include "rt/RayQueryTracer.h"
#include "rt/TraceTypes.h"
#include "vulkan/Buffer.h"
#include "vulkan/CommandContext.h"
#include "vulkan/Shader.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace raylidar::rt
{
class MeshScene;
class GaussianScene;
} // namespace raylidar::rt

namespace raylidar::simulation
{
// Per-stage cost of one Scan() call, in milliseconds.
struct ScanTimings
{
    double Upload = 0.0;     // CPU: copy rays into the upload buffer
    double RayGen = 0.0;     // GPU: ray generation dispatch, only when rays are generated on the GPU
    double Trace = 0.0;      // GPU: ray query dispatch (timestamps)
    double CopyHits = 0.0;   // GPU: hit buffer -> readback buffer (timestamps)
    double SubmitWait = 0.0; // CPU: record, submit and wait for the GPU
    double Readback = 0.0;   // CPU: read hits and build points
    double Routing = 0.0;    // CPU, hybrid only: pick the rays for the Gaussian pass and merge both passes
    double Total = 0.0;      // CPU: the whole Scan() call
};

struct ScanResult
{
    std::vector<rt::Hit> Hits;

    std::vector<std::array<float, 3>> Points;

    ScanTimings Timings;
};

class LidarSimulator
{
  public:
    LidarSimulator(const vulkan::VulkanContext &context, const std::filesystem::path &traceShaderPath,
                   std::uint32_t maxRayCount, rt::RayQueryMode mode = rt::RayQueryMode::Mesh);

    LidarSimulator(const LidarSimulator &) = delete;
    LidarSimulator &operator=(const LidarSimulator &) = delete;

    LidarSimulator(LidarSimulator &&) = delete;
    LidarSimulator &operator=(LidarSimulator &&) = delete;

    ~LidarSimulator() = default;

    [[nodiscard]]
    ScanResult Scan(const rt::MeshScene &scene, std::span<const rt::Ray> rays);

    [[nodiscard]]
    ScanResult Scan(const rt::GaussianScene &scene, std::span<const rt::Ray> rays, float alphaThreshold);

    // Creates the GPU ray generator and a device-local ray buffer. Required before Scan(scene, sensor, pose).
    void EnableGpuRayGeneration(const std::filesystem::path &rayGenShaderPath);

    // Generates the sensor's rays on the GPU and traces them in the same submission: no CPU ray generation and no
    // ray upload. Points are rebuilt on the CPU from the sensor directions and the pose.
    [[nodiscard]]
    ScanResult Scan(const rt::MeshScene &scene, const LidarSensor &sensor, const Transform &sensorToWorld);

  private:
    // Traces rayCount rays already in rays (or written there by recordRays in the same submission) and reads the
    // hits back. Does not build points.
    template <typename SceneType, typename Parameters, typename RecordRays>
    ScanResult TraceRays(const SceneType &scene, const vulkan::Buffer &rays, std::uint32_t rayCount,
                         Parameters parameters, const RecordRays &recordRays);

    // Uploads rays from the CPU, traces them and builds points.
    template <typename SceneType, typename Parameters>
    ScanResult ScanUploadedRays(const SceneType &scene, std::span<const rt::Ray> rays, Parameters parameters);

    const vulkan::VulkanContext &m_context;
    std::uint32_t m_maxRayCount = 0;

    vulkan::Shader m_shader;
    rt::RayQueryTracer m_tracer;

    vulkan::Buffer m_rayBuffer;
    vulkan::Buffer m_hitBuffer;
    vulkan::Buffer m_readbackBuffer;

    vulkan::CommandContext m_commands;
    vk::raii::QueryPool m_timestampPool = nullptr;

    std::optional<vulkan::Buffer> m_deviceRayBuffer;
    std::optional<GpuRayGenerator> m_rayGenerator;
};
} // namespace raylidar::simulation
