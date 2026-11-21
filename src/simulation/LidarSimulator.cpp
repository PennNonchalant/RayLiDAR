#include "LidarSimulator.h"

#include "rt/GaussianScene.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace raylidar::simulation
{
namespace
{
using Clock = std::chrono::steady_clock;

// Start, after ray generation, after trace, after hit copy.
constexpr std::uint32_t kTimestampCount = 4;

double MillisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

vulkan::BufferDescription MakeBufferDescription(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                vulkan::BufferMemoryUsage memoryUsage)
{
    vulkan::BufferDescription description;
    description.Size = size;
    description.Usage = usage;
    description.MemoryUsage = memoryUsage;
    return description;
}
} // namespace

LidarSimulator::LidarSimulator(const vulkan::VulkanContext &context, const std::filesystem::path &traceShaderPath,
                               std::uint32_t maxRayCount, rt::RayQueryMode mode, std::uint32_t gaussianHitBatchSize)
    : m_context(context), m_maxRayCount(maxRayCount), m_shader(context, traceShaderPath),
      m_tracer(context, m_shader, mode, gaussianHitBatchSize),
      m_rayBuffer(context, MakeBufferDescription(maxRayCount * sizeof(rt::Ray), vk::BufferUsageFlagBits::eStorageBuffer,
                                                 vulkan::BufferMemoryUsage::Upload)),
      m_hitBuffer(context,
                  MakeBufferDescription(maxRayCount * sizeof(rt::Hit),
                                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                                        vulkan::BufferMemoryUsage::Device)),
      m_readbackBuffer(context,
                       MakeBufferDescription(maxRayCount * sizeof(rt::Hit), vk::BufferUsageFlagBits::eTransferDst,
                                             vulkan::BufferMemoryUsage::Readback)),
      m_commands(context)
{
    if (maxRayCount == 0)
    {
        throw std::invalid_argument("LidarSimulator requires a non-zero ray capacity.");
    }

    const vk::QueryPoolCreateInfo queryPoolInfo({}, vk::QueryType::eTimestamp, kTimestampCount);
    m_timestampPool = vk::raii::QueryPool(context.Device(), queryPoolInfo);
}

template <typename SceneType, typename Parameters, typename RecordRays>
ScanResult LidarSimulator::TraceRays(const SceneType &scene, const vulkan::Buffer &rays, std::uint32_t rayCount,
                                     Parameters parameters, const RecordRays &recordRays)
{
    ScanResult result;
    parameters.RayCount = rayCount;
    const vk::DeviceSize hitBytes = static_cast<vk::DeviceSize>(rayCount) * sizeof(rt::Hit);

    const auto submitStart = Clock::now();
    m_tracer.SetResources(scene, rays, m_hitBuffer);

    const auto &commandBuffer = m_commands.Begin();

    commandBuffer.resetQueryPool(*m_timestampPool, 0, kTimestampCount);
    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *m_timestampPool, 0);

    recordRays(commandBuffer);
    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *m_timestampPool, 1);

    m_tracer.RecordTrace(commandBuffer, parameters);
    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *m_timestampPool, 2);

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                                vk::AccessFlagBits2::eTransferRead);

    commandBuffer.copyBuffer(m_hitBuffer.Handle(), m_readbackBuffer.Handle(), vk::BufferCopy(0, 0, hitBytes));
    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eCopy, *m_timestampPool, 3);

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);

    m_commands.End();
    m_commands.SubmitAndWait();
    result.Timings.SubmitWait = MillisecondsSince(submitStart);

    const auto [queryResult, timestamps] = m_timestampPool.getResults<std::uint64_t>(
        0, kTimestampCount, kTimestampCount * sizeof(std::uint64_t), sizeof(std::uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

    if (queryResult == vk::Result::eSuccess)
    {
        const double period = m_context.PhysicalDeviceProperties().limits.timestampPeriod;
        result.Timings.RayGen = static_cast<double>(timestamps[1] - timestamps[0]) * period / 1.0e6;
        result.Timings.Trace = static_cast<double>(timestamps[2] - timestamps[1]) * period / 1.0e6;
        result.Timings.CopyHits = static_cast<double>(timestamps[3] - timestamps[2]) * period / 1.0e6;
    }

    const auto readbackStart = Clock::now();
    result.Hits.resize(rayCount);
    m_readbackBuffer.Read(std::as_writable_bytes(std::span(result.Hits)));
    result.Timings.Readback = MillisecondsSince(readbackStart);

    return result;
}

template <typename SceneType, typename Parameters>
ScanResult LidarSimulator::ScanUploadedRays(const SceneType &scene, std::span<const rt::Ray> rays,
                                            Parameters parameters)
{
    if (rays.size() > m_maxRayCount)
    {
        throw std::out_of_range("Ray count exceeds LidarSimulator capacity.");
    }

    if (rays.empty())
    {
        return {};
    }

    const auto scanStart = Clock::now();

    const auto uploadStart = Clock::now();
    m_rayBuffer.Write(std::as_bytes(rays));
    const double upload = MillisecondsSince(uploadStart);

    auto result = TraceRays(scene, m_rayBuffer, static_cast<std::uint32_t>(rays.size()), parameters,
                            [](const vk::raii::CommandBuffer &) {});
    result.Timings.Upload = upload;

    const auto pointsStart = Clock::now();
    result.Points.reserve(rays.size());

    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const auto &hit = result.Hits[index];

        if (hit.Valid == 0)
        {
            continue;
        }

        const auto &ray = rays[index];
        result.Points.push_back({
            ray.Origin[0] + hit.T * ray.Direction[0],
            ray.Origin[1] + hit.T * ray.Direction[1],
            ray.Origin[2] + hit.T * ray.Direction[2],
        });
    }

    result.Timings.Readback += MillisecondsSince(pointsStart);
    result.Timings.Total = MillisecondsSince(scanStart);
    return result;
}

ScanResult LidarSimulator::Scan(const rt::MeshScene &scene, std::span<const rt::Ray> rays)
{
    return ScanUploadedRays(scene, rays, rt::MeshTraceParameters{});
}

ScanResult LidarSimulator::Scan(const rt::GaussianScene &scene, std::span<const rt::Ray> rays, float alphaThreshold)
{
    const rt::GaussianTraceParameters parameters{
        .GaussianIndexStride = scene.GaussianIndexStride(),
        .SigmaCutoff = scene.SigmaCutoff(),
        .AlphaThreshold = alphaThreshold,
    };
    return ScanUploadedRays(scene, rays, parameters);
}

void LidarSimulator::EnableGpuRayGeneration(const std::filesystem::path &rayGenShaderPath)
{
    m_deviceRayBuffer.emplace(m_context, MakeBufferDescription(m_maxRayCount * sizeof(rt::Ray),
                                                               vk::BufferUsageFlagBits::eStorageBuffer,
                                                               vulkan::BufferMemoryUsage::Device));
    m_rayGenerator.emplace(m_context, rayGenShaderPath);
    m_rayGenerator->SetOutput(*m_deviceRayBuffer);
}

ScanResult LidarSimulator::Scan(const rt::MeshScene &scene, const LidarSensor &sensor, const Transform &sensorToWorld)
{
    if (!m_rayGenerator)
    {
        throw std::logic_error("Call EnableGpuRayGeneration before scanning with GPU-generated rays.");
    }
    if (sensor.RayCount() > m_maxRayCount)
    {
        throw std::out_of_range("Ray count exceeds LidarSimulator capacity.");
    }

    const auto scanStart = Clock::now();
    const auto &description = sensor.Description();

    auto result = TraceRays(scene, *m_deviceRayBuffer, sensor.RayCount(), rt::MeshTraceParameters{},
                            [this, &description, &sensorToWorld](const vk::raii::CommandBuffer &commandBuffer) {
                                m_rayGenerator->Record(commandBuffer, description, sensorToWorld);
                                vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                                            vk::AccessFlagBits2::eShaderWrite,
                                                            vk::PipelineStageFlagBits2::eComputeShader,
                                                            vk::AccessFlagBits2::eShaderRead);
                            });

    // The rays never reach the CPU, so rebuild each hit point from the sensor direction, the same way the GPU did.
    const auto pointsStart = Clock::now();
    const auto directions = sensor.LocalDirections();
    const auto &m = sensorToWorld;
    result.Points.reserve(result.Hits.size());

    for (std::size_t index = 0; index < result.Hits.size(); ++index)
    {
        const auto &hit = result.Hits[index];
        if (hit.Valid == 0)
        {
            continue;
        }

        const auto &d = directions[index];
        std::array<float, 3> world = {
            m[0][0] * d[0] + m[0][1] * d[1] + m[0][2] * d[2],
            m[1][0] * d[0] + m[1][1] * d[1] + m[1][2] * d[2],
            m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2],
        };
        const float scale = hit.T / std::sqrt(world[0] * world[0] + world[1] * world[1] + world[2] * world[2]);

        result.Points.push_back({
            m[0][3] + scale * world[0],
            m[1][3] + scale * world[1],
            m[2][3] + scale * world[2],
        });
    }

    result.Timings.Readback += MillisecondsSince(pointsStart);
    result.Timings.Total = MillisecondsSince(scanStart);
    return result;
}

} // namespace raylidar::simulation
