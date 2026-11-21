#include "LidarSimulator.h"

#include "rt/GaussianScene.h"
#include "vulkan/Synchronization.h"
#include "vulkan/VulkanContext.h"

#include <array>
#include <chrono>
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
rt::GaussianTraceParameters GaussianParameters(const rt::GaussianScene &scene, float alphaThreshold)
{
    return {.GaussianIndexStride = scene.GaussianIndexStride(),
            .SigmaCutoff = scene.SigmaCutoff(),
            .AlphaThreshold = alphaThreshold};
}

template <typename GetRay> void BuildPoints(ScanResult &result, const GetRay &getRay)
{
    const auto start = Clock::now();
    result.Points.reserve(result.Hits.size());
    for (std::size_t index = 0; index < result.Hits.size(); ++index)
    {
        const auto &hit = result.Hits[index];
        if (hit.Valid == 0)
        {
            continue;
        }

        const auto &ray = getRay(index);
        result.Points.push_back({
            ray.Origin[0] + hit.T * ray.Direction[0],
            ray.Origin[1] + hit.T * ray.Direction[1],
            ray.Origin[2] + hit.T * ray.Direction[2],
        });
    }
    result.Timings.Readback += MillisecondsSince(start);
}
} // namespace

ScanTimings &ScanTimings::operator+=(const ScanTimings &other)
{
    Upload += other.Upload;
    RayGen += other.RayGen;
    Trace += other.Trace;
    CopyHits += other.CopyHits;
    SubmitWait += other.SubmitWait;
    Readback += other.Readback;
    Routing += other.Routing;
    Total += other.Total;
    return *this;
}

LidarSimulator::LidarSimulator(const vulkan::VulkanContext &context, const std::filesystem::path &traceShaderPath,
                               std::uint32_t maxRayCount, rt::RayQueryMode mode,
                               rt::GaussianTraceOptions gaussianOptions)
    : m_context(context), m_maxRayCount(maxRayCount), m_shader(context, traceShaderPath),
      m_tracer(context, m_shader, mode, gaussianOptions),
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

    BuildPoints(result, [rays](std::size_t index) -> const rt::Ray & { return rays[index]; });
    result.Timings.Total = MillisecondsSince(scanStart);
    return result;
}

ScanResult LidarSimulator::Scan(const rt::MeshScene &scene, std::span<const rt::Ray> rays)
{
    return ScanUploadedRays(scene, rays, rt::MeshTraceParameters{});
}

ScanResult LidarSimulator::Scan(const rt::GaussianScene &scene, std::span<const rt::Ray> rays, float alphaThreshold)
{
    return ScanUploadedRays(scene, rays, GaussianParameters(scene, alphaThreshold));
}

void LidarSimulator::EnableGpuRayGeneration(const std::filesystem::path &rayGenShaderPath, const LidarSensor &sensor)
{
    m_deviceRayBuffer.emplace(m_context, MakeBufferDescription(m_maxRayCount * sizeof(rt::Ray),
                                                               vk::BufferUsageFlagBits::eStorageBuffer,
                                                               vulkan::BufferMemoryUsage::Device));
    m_rayGenerator.emplace(m_context, rayGenShaderPath, sensor);
    m_rayGenerator->SetBuffers(*m_deviceRayBuffer, m_rayBuffer);
}

ScanResult LidarSimulator::Scan(const rt::MeshScene &scene, const LidarSensor &sensor, const Transform &sensorToWorld)
{
    return ScanGeneratedRays(scene, {sensor, sensorToWorld}, rt::MeshTraceParameters{}, 0, sensor.RayCount());
}

ScanResult LidarSimulator::Scan(const rt::GaussianScene &scene, const SensorRays &rays, float alphaThreshold,
                                std::size_t first, std::size_t count)
{
    return ScanGeneratedRays(scene, rays, GaussianParameters(scene, alphaThreshold), first, count);
}

template <typename SceneType, typename Parameters>
ScanResult LidarSimulator::ScanGeneratedRays(const SceneType &scene, const SensorRays &rays, Parameters parameters,
                                             std::size_t first, std::size_t count)
{
    if (!m_rayGenerator)
    {
        throw std::logic_error("Call EnableGpuRayGeneration before scanning with GPU-generated rays.");
    }
    if (count > m_maxRayCount)
    {
        throw std::out_of_range("Ray count exceeds LidarSimulator capacity.");
    }

    const auto scanStart = Clock::now();
    const auto rayCount = static_cast<std::uint32_t>(count);
    const bool useRayIndices = !rays.Indices.empty();
    double upload = 0.0;
    if (useRayIndices)
    {
        const auto uploadStart = Clock::now();
        m_rayBuffer.Write(std::as_bytes(rays.Indices.subspan(first, count)));
        upload = MillisecondsSince(uploadStart);
    }

    auto result =
        TraceRays(scene, *m_deviceRayBuffer, rayCount, parameters, [&](const vk::raii::CommandBuffer &commandBuffer) {
            m_rayGenerator->Record(commandBuffer, rays.Sensor.Description(), rays.SensorToWorld,
                                   static_cast<std::uint32_t>(first), rayCount, useRayIndices);
            vulkan::RecordMemoryBarrier(commandBuffer, vk::PipelineStageFlagBits2::eComputeShader,
                                        vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader,
                                        vk::AccessFlagBits2::eShaderRead);
        });
    result.Timings.Upload = upload;
    BuildPoints(result, [&](std::size_t index) {
        const auto rayIndex = useRayIndices ? rays.Indices[first + index] : first + index;
        return rays.Sensor.GenerateRay(rayIndex, rays.SensorToWorld);
    });
    result.Timings.Total = MillisecondsSince(scanStart);
    return result;
}

} // namespace raylidar::simulation
