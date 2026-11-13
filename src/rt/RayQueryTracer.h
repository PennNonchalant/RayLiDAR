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

struct MeshTraceParameters
{
    std::uint32_t RayCount = 0;
};

struct GaussianTraceParameters
{
    std::uint32_t RayCount = 0;
    std::uint32_t GaussiansPerBatch = 0;
    float SigmaCutoff = 0.0f;
    float AlphaThreshold = 0.0f;
};

enum class RayQueryMode
{
    Mesh,
    Gaussian
};

class MeshScene;
class GaussianScene;

class RayQueryTracer
{
  public:
    RayQueryTracer(const vulkan::VulkanContext &context, const vulkan::Shader &shader,
                   RayQueryMode mode = RayQueryMode::Mesh);
    ~RayQueryTracer() = default;

    RayQueryTracer(const RayQueryTracer &) = delete;
    RayQueryTracer &operator=(const RayQueryTracer &) = delete;

    RayQueryTracer(RayQueryTracer &&) noexcept = default;
    RayQueryTracer &operator=(RayQueryTracer &&) = delete;

    void SetResources(const MeshScene &scene, const vulkan::Buffer &rays, const vulkan::Buffer &hits);
    void SetResources(const GaussianScene &scene, const vulkan::Buffer &rays, const vulkan::Buffer &hits);

    void RecordTrace(const vk::raii::CommandBuffer &commandBuffer, const MeshTraceParameters &parameters);
    void RecordTrace(const vk::raii::CommandBuffer &commandBuffer, const GaussianTraceParameters &parameters);

  private:
    void WriteTraceBindings(const AccelerationStructure &tlas, const vulkan::Buffer &rays, const vulkan::Buffer &hits);
    void RecordDispatch(const vk::raii::CommandBuffer &commandBuffer, std::uint32_t groupCount,
                        std::span<const std::byte> constants);

    const vulkan::VulkanContext &m_context;

    vulkan::ComputePipeline m_pipeline;
    vulkan::DescriptorBindings m_bindings;
};
} // namespace raylidar::rt
