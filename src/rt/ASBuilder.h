#pragma once

#include "AccelerationStructure.h"
#include "vulkan/Buffer.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace raylidar::vulkan
{
class VulkanContext;
} // namespace raylidar::vulkan

namespace raylidar::rt
{
struct TriangleGeometryView
{
    const vulkan::Buffer &VertexBuffer;
    const vulkan::Buffer &IndexBuffer;

    std::uint32_t VertexCount = 0;
    std::uint32_t IndexCount = 0;

    vk::DeviceSize VertexOffset = 0;
    vk::DeviceSize IndexOffset = 0;
    vk::DeviceSize VertexStride = 3 * sizeof(float);
};

class ASBuilder
{
  public:
    explicit ASBuilder(const vulkan::VulkanContext &context);
    ~ASBuilder() = default;

    ASBuilder(const ASBuilder &) = delete;
    ASBuilder &operator=(const ASBuilder &) = delete;

    ASBuilder(ASBuilder &&) = delete;
    ASBuilder &operator=(ASBuilder &&) = delete;

    [[nodiscard]]
    AccelerationStructure RecordBlas(const vk::raii::CommandBuffer &commandBuffer,
                                     const TriangleGeometryView &geometry);

    [[nodiscard]]
    AccelerationStructure RecordTlas(const vk::raii::CommandBuffer &commandBuffer,
                                     std::span<const vk::AccelerationStructureInstanceKHR> instances);

  private:
    AccelerationStructure RecordBuild(const vk::raii::CommandBuffer &commandBuffer,
                                      vk::AccelerationStructureTypeKHR type,
                                      const vk::AccelerationStructureGeometryKHR &geometry,
                                      std::uint32_t primitiveCount);

    const vulkan::VulkanContext &m_context;

    std::vector<vulkan::Buffer> m_buildBuffers;
};
} // namespace raylidar::rt
