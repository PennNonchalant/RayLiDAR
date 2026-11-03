#include "RayQueryTracer.h"

#include "AccelerationStructure.h"
#include "TraceTypes.h"
#include "vulkan/Buffer.h"
#include "vulkan/VulkanContext.h"

#include <algorithm>
#include <stdexcept>

namespace raylidar::rt
{
namespace
{
constexpr std::uint32_t kWorkgroupSize = 64;

vulkan::ComputePipelineDescription MakePipelineDescription()
{
    vulkan::ComputePipelineDescription description;
    description.PushConstantSize = sizeof(TraceParameters);
    description.DescriptorSets.resize(1);

    description.DescriptorSets[0].Bindings = {
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };

    return description;
}
} // namespace

RayQueryTracer::RayQueryTracer(const vulkan::VulkanContext &context, const vulkan::Shader &shader)
    : m_context(context), m_pipeline(context, shader, MakePipelineDescription()), m_bindings(context, m_pipeline)
{
}

void RayQueryTracer::SetResources(const AccelerationStructure &tlas, const vulkan::Buffer &rays,
                                  const vulkan::Buffer &hits)
{
    if (tlas.Type() != vk::AccelerationStructureTypeKHR::eTopLevel)
    {
        throw std::invalid_argument("RayQueryTracer requires a top-level acceleration structure.");
    }

    if (!(rays.Usage() & vk::BufferUsageFlagBits::eStorageBuffer) ||
        !(hits.Usage() & vk::BufferUsageFlagBits::eStorageBuffer))
    {
        throw std::invalid_argument("Ray and hit buffers require storage buffer usage.");
    }

    const auto maxStorageBufferRange = m_context.PhysicalDeviceProperties().limits.maxStorageBufferRange;

    if (rays.Size() > maxStorageBufferRange || hits.Size() > maxStorageBufferRange)
    {
        throw std::invalid_argument("Ray or hit buffer exceeds maxStorageBufferRange.");
    }

    const auto rayCapacity = std::min(rays.Size() / sizeof(Ray), hits.Size() / sizeof(Hit));

    m_bindings.WriteAccelerationStructure(0, 0, tlas.Handle());
    m_bindings.WriteBuffer(0, 1, rays);
    m_bindings.WriteBuffer(0, 2, hits);

    m_rayCapacity = static_cast<std::uint32_t>(rayCapacity);
}

void RayQueryTracer::RecordTrace(const vk::raii::CommandBuffer &commandBuffer, const TraceParameters &parameters)
{
    const std::uint32_t rayCount = parameters.RayCount;

    if (rayCount == 0)
    {
        return;
    }

    if (rayCount > m_rayCapacity)
    {
        throw std::out_of_range("Ray count exceeds the bound buffer capacity. Call SetResources first.");
    }

    const std::uint32_t groupCount = (rayCount + kWorkgroupSize - 1) / kWorkgroupSize;

    if (groupCount > m_context.PhysicalDeviceProperties().limits.maxComputeWorkGroupCount[0])
    {
        throw std::out_of_range("Ray count exceeds the one-dimensional dispatch limit.");
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline.Handle());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipeline.Layout(), 0, m_bindings.Handles(), {});
    commandBuffer.pushConstants(m_pipeline.Layout(), vk::ShaderStageFlagBits::eCompute, 0,
                                static_cast<std::uint32_t>(sizeof(TraceParameters)), &parameters);
    commandBuffer.dispatch(groupCount, 1, 1);
}
} // namespace raylidar::rt
