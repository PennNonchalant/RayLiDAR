#pragma once

#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorBindings.h"

#include <cstdint>

namespace raylidar::vulkan
{
class Buffer;
class Shader;
class VulkanContext;
} // namespace raylidar::vulkan

namespace raylidar::rt
{
class AccelerationStructure;

struct TraceParameters
{
    std::uint32_t RayCount = 0;
};

class RayQueryTracer
{
  public:
    RayQueryTracer(const vulkan::VulkanContext &context, const vulkan::Shader &shader);
    ~RayQueryTracer() = default;

    RayQueryTracer(const RayQueryTracer &) = delete;
    RayQueryTracer &operator=(const RayQueryTracer &) = delete;

    RayQueryTracer(RayQueryTracer &&) noexcept = default;
    RayQueryTracer &operator=(RayQueryTracer &&) = delete;

    void SetResources(const AccelerationStructure &tlas, const vulkan::Buffer &rays, const vulkan::Buffer &hits);

    void RecordTrace(const vk::raii::CommandBuffer &commandBuffer, const TraceParameters &parameters);

  private:
    const vulkan::VulkanContext &m_context;

    vulkan::ComputePipeline m_pipeline;
    vulkan::DescriptorBindings m_bindings;

    std::uint32_t m_rayCapacity = 0;
};
} // namespace raylidar::rt
