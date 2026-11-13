#include "LidarSimulator.h"

#include "rt/GaussianScene.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <chrono>
#include <cstddef>
#include <stdexcept>

namespace raylidar::simulation
{
namespace
{
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kTimestampCount = 3;

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
                               std::uint32_t maxRayCount, rt::RayQueryMode mode)
    : m_context(context), m_maxRayCount(maxRayCount), m_shader(context, traceShaderPath),
      m_tracer(context, m_shader, mode),
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

ScanResult LidarSimulator::Scan(const rt::MeshScene &scene, std::span<const rt::Ray> rays)
{
    if (rays.size() > m_maxRayCount)
    {
        throw std::out_of_range("Ray count exceeds LidarSimulator capacity.");
    }

    return ScanImpl(scene, rays, rt::MeshTraceParameters{});
}

ScanResult LidarSimulator::Scan(const rt::GaussianScene &scene, std::span<const rt::Ray> rays, float alphaThreshold)
{
    const rt::GaussianTraceParameters parameters{
        .GaussiansPerBatch = scene.GaussiansPerBatch(),
        .SigmaCutoff = scene.SigmaCutoff(),
        .AlphaThreshold = alphaThreshold,
    };
    return ScanImpl(scene, rays, parameters);
}

template <typename SceneType, typename Parameters>
ScanResult LidarSimulator::ScanImpl(const SceneType &scene, std::span<const rt::Ray> rays, Parameters parameters)
{
    const auto scanStart = Clock::now();

    ScanResult result;

    if (rays.empty())
    {
        return result;
    }

    parameters.RayCount = static_cast<std::uint32_t>(rays.size());
    const vk::DeviceSize hitBytes = rays.size() * sizeof(rt::Hit);

    const auto uploadStart = Clock::now();
    m_rayBuffer.Write(std::as_bytes(rays));
    result.Timings.Upload = MillisecondsSince(uploadStart);

    const auto submitStart = Clock::now();
    m_tracer.SetResources(scene, m_rayBuffer, m_hitBuffer);

    const auto &commandBuffer = m_commands.Begin();

    commandBuffer.resetQueryPool(*m_timestampPool, 0, kTimestampCount);
    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *m_timestampPool, 0);

    m_tracer.RecordTrace(commandBuffer, parameters);

    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *m_timestampPool, 1);

    vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eCopy,
                                vk::AccessFlagBits2::eTransferRead);

    commandBuffer.copyBuffer(m_hitBuffer.Handle(), m_readbackBuffer.Handle(), vk::BufferCopy(0, 0, hitBytes));
    commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eCopy, *m_timestampPool, 2);

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
        result.Timings.Trace = static_cast<double>(timestamps[1] - timestamps[0]) * period / 1.0e6;
        result.Timings.CopyHits = static_cast<double>(timestamps[2] - timestamps[1]) * period / 1.0e6;
    }

    const auto readbackStart = Clock::now();
    result.Hits.resize(rays.size());
    m_readbackBuffer.Read(std::as_writable_bytes(std::span(result.Hits)));

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

    result.Timings.Readback = MillisecondsSince(readbackStart);
    result.Timings.Total = MillisecondsSince(scanStart);

    return result;
}

} // namespace raylidar::simulation
