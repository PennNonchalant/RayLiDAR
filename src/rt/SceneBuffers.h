#pragma once

#include "vulkan/Buffer.h"

#include <cstddef>
#include <span>

namespace raylidar::rt
{
[[nodiscard]]
inline vulkan::Buffer MakeTriangleBuildInputBuffer(const vulkan::VulkanContext &context, vk::DeviceSize size)
{
    vulkan::BufferDescription description;
    description.Size = size;
    description.Usage = vk::BufferUsageFlagBits::eTransferDst |
                        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                        vk::BufferUsageFlagBits::eShaderDeviceAddress;
    description.MinimumAlignment = 4;
    return vulkan::Buffer(context, description);
}

[[nodiscard]]
inline vulkan::Buffer MakeStagingBuffer(const vulkan::VulkanContext &context, std::span<const std::byte> bytes)
{
    vulkan::BufferDescription description;
    description.Size = bytes.size();
    description.Usage = vk::BufferUsageFlagBits::eTransferSrc;
    description.MemoryUsage = vulkan::BufferMemoryUsage::Upload;

    vulkan::Buffer buffer(context, description);
    buffer.Write(bytes);
    return buffer;
}
} // namespace raylidar::rt