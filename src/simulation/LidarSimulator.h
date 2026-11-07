#pragma once

#include "rt/RayQueryTracer.h"
#include "rt/TraceTypes.h"
#include "vulkan/Buffer.h"
#include "vulkan/CommandContext.h"
#include "vulkan/Shader.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace raylidar::rt
{
class AccelerationStructure;
} // namespace raylidar::rt

namespace raylidar::simulation
{
// Per-stage cost of one Scan() call, in milliseconds.
struct ScanTimings
{
    double Upload = 0.0;     // CPU: copy rays into the upload buffer
    double Trace = 0.0;      // GPU: ray query dispatch (timestamps)
    double CopyHits = 0.0;   // GPU: hit buffer -> readback buffer (timestamps)
    double SubmitWait = 0.0; // CPU: record, submit and wait for the GPU
    double Readback = 0.0;   // CPU: read hits and build points
    double Total = 0.0;      // CPU: the whole Scan() call
};

struct ScanResult
{
    // One hit per input ray, in input order.
    std::vector<rt::Hit> Hits;

    // World-space points of valid hits only.
    std::vector<std::array<float, 3>> Points;

    ScanTimings Timings;
};

class LidarSimulator
{
  public:
    LidarSimulator(const vulkan::VulkanContext &context, const std::filesystem::path &traceShaderPath,
                   std::uint32_t maxRayCount);

    LidarSimulator(const LidarSimulator &) = delete;
    LidarSimulator &operator=(const LidarSimulator &) = delete;

    LidarSimulator(LidarSimulator &&) = delete;
    LidarSimulator &operator=(LidarSimulator &&) = delete;

    ~LidarSimulator() = default;

    [[nodiscard]]
    ScanResult Scan(const rt::AccelerationStructure &tlas, std::span<const rt::Ray> rays);

  private:
    const vulkan::VulkanContext &m_context;
    std::uint32_t m_maxRayCount = 0;

    vulkan::Shader m_shader;
    rt::RayQueryTracer m_tracer;

    vulkan::Buffer m_rayBuffer;
    vulkan::Buffer m_hitBuffer;
    vulkan::Buffer m_readbackBuffer;

    vulkan::CommandContext m_commands;
    vk::raii::QueryPool m_timestampPool = nullptr;
};
} // namespace raylidar::simulation
