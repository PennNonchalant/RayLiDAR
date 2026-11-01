#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include <cstddef>
#include <span>

namespace raylidar::vulkan
{
class VulkanContext;

enum class BufferMemoryUsage
{
    Device,
    Upload,
    Readback
};

struct BufferDescription
{
    vk::DeviceSize Size = 0;
    vk::BufferUsageFlags Usage{};
    BufferMemoryUsage MemoryUsage = BufferMemoryUsage::Device;
    vk::DeviceSize MinimumAlignment = 0;
};

class Buffer
{
public:
    Buffer(const VulkanContext &context, const BufferDescription &description);
    ~Buffer() noexcept;

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&other) noexcept;
    Buffer &operator=(Buffer &&other) noexcept;

    [[nodiscard]]
    vk::Buffer Handle() const noexcept
    {
        return m_buffer;
    }

    [[nodiscard]]
    vk::DeviceSize Size() const noexcept
    {
        return m_size;
    }

    [[nodiscard]]
    vk::BufferUsageFlags Usage() const noexcept
    {
        return m_usage;
    }

    [[nodiscard]]
    vk::DeviceAddress DeviceAddress() const;

    void Write(std::span<const std::byte> data, vk::DeviceSize offset = 0);
    void Read(std::span<std::byte> destination, vk::DeviceSize offset = 0) const;

private:
    void Initialize(const VulkanContext &context, const BufferDescription &description);
    void ValidateRange(std::size_t byteCount, vk::DeviceSize offset) const;
    void Release() noexcept;

    VmaAllocator m_allocator = nullptr;

    vk::Buffer m_buffer = nullptr;
    VmaAllocation m_allocation = nullptr;

    vk::DeviceSize m_size = 0;
    vk::BufferUsageFlags m_usage{};
    BufferMemoryUsage m_memoryUsage = BufferMemoryUsage::Device;
    vk::DeviceAddress m_deviceAddress = 0;
};
} // namespace raylidar::vulkan
