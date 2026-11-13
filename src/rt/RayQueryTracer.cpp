#include "RayQueryTracer.h"

#include "AccelerationStructure.h"
#include "GaussianScene.h"
#include "MeshScene.h"
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

vulkan::ComputePipelineDescription MakePipelineDescription(RayQueryMode mode)
{
    vulkan::ComputePipelineDescription description;
    description.DescriptorSets.resize(1);
    description.DescriptorSets[0].Bindings = {
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };

    if (mode == RayQueryMode::Mesh)
    {
        description.PushConstantSize = sizeof(MeshTraceParameters);
    }
    else
    {
        description.PushConstantSize = sizeof(GaussianTraceParameters);
        description.DescriptorSets[0].Bindings.emplace_back(3, vk::DescriptorType::eStorageBuffer, 1,
                                                            vk::ShaderStageFlagBits::eCompute);
    }
    return description;
}
} // namespace

RayQueryTracer::RayQueryTracer(const vulkan::VulkanContext &context, const vulkan::Shader &shader, RayQueryMode mode)
    : m_context(context), m_pipeline(context, shader, MakePipelineDescription(mode)), m_bindings(context, m_pipeline)
{
}

void RayQueryTracer::SetResources(const MeshScene &scene, const vulkan::Buffer &rays, const vulkan::Buffer &hits)
{
    WriteTraceBindings(scene.Tlas(), rays, hits);
}

void RayQueryTracer::SetResources(const GaussianScene &scene, const vulkan::Buffer &rays, const vulkan::Buffer &hits)
{
    WriteTraceBindings(scene.Tlas(), rays, hits);
    m_bindings.WriteBuffer(0, 3, scene.GaussianBuffer());
}

void RayQueryTracer::RecordTrace(const vk::raii::CommandBuffer &commandBuffer, const MeshTraceParameters &parameters)
{
    RecordDispatch(commandBuffer, parameters.RayCount, std::as_bytes(std::span(&parameters, 1)));
}

void RayQueryTracer::RecordTrace(const vk::raii::CommandBuffer &commandBuffer,
                                 const GaussianTraceParameters &parameters)
{
    RecordDispatch(commandBuffer, parameters.RayCount, std::as_bytes(std::span(&parameters, 1)));
}

void RayQueryTracer::WriteTraceBindings(const AccelerationStructure &tlas, const vulkan::Buffer &rays,
                                        const vulkan::Buffer &hits)
{
    m_bindings.WriteAccelerationStructure(0, 0, tlas.Handle());
    m_bindings.WriteBuffer(0, 1, rays);
    m_bindings.WriteBuffer(0, 2, hits);
}

void RayQueryTracer::RecordDispatch(const vk::raii::CommandBuffer &commandBuffer, std::uint32_t rayCount,
                                    std::span<const std::byte> constants)
{
    if (rayCount == 0)
    {
        return;
    }

    const std::uint32_t groupCount = (rayCount + kWorkgroupSize - 1) / kWorkgroupSize;
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline.Handle());
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipeline.Layout(), 0, m_bindings.Handles(), {});
    commandBuffer.pushConstants(m_pipeline.Layout(), vk::ShaderStageFlagBits::eCompute, 0,
                                static_cast<std::uint32_t>(constants.size()), constants.data());
    commandBuffer.dispatch(groupCount, 1, 1);
}
} // namespace raylidar::rt
