#include "Buffer.h"

#include "VulkanContext.h"

#include <bit>
#include <stdexcept>
#include <string>
#include <utility>

namespace raylidar::vulkan
{
Buffer::Buffer(const VulkanContext &context, const BufferDescription &description)
{
    Initialize(context, description);
}

Buffer::~Buffer() noexcept
{
    Release();
}

Buffer::Buffer(Buffer &&other) noexcept
    : m_allocator(std::exchange(other.m_allocator, nullptr)), m_buffer(std::exchange(other.m_buffer, vk::Buffer{})),
      m_allocation(std::exchange(other.m_allocation, nullptr)), m_size(std::exchange(other.m_size, vk::DeviceSize{0})),
      m_usage(std::exchange(other.m_usage, vk::BufferUsageFlags{})),
      m_memoryUsage(std::exchange(other.m_memoryUsage, BufferMemoryUsage::Device)),
      m_deviceAddress(std::exchange(other.m_deviceAddress, vk::DeviceAddress{0}))
{
}

Buffer &Buffer::operator=(Buffer &&other) noexcept
{
    if (this != &other)
    {
        Release();

        m_allocator = std::exchange(other.m_allocator, nullptr);
        m_buffer = std::exchange(other.m_buffer, vk::Buffer{});
        m_allocation = std::exchange(other.m_allocation, nullptr);
        m_size = std::exchange(other.m_size, vk::DeviceSize{0});
        m_usage = std::exchange(other.m_usage, vk::BufferUsageFlags{});
        m_memoryUsage = std::exchange(other.m_memoryUsage, BufferMemoryUsage::Device);
        m_deviceAddress = std::exchange(other.m_deviceAddress, vk::DeviceAddress{0});
    }

    return *this;
}

void Buffer::Initialize(const VulkanContext &context, const BufferDescription &description)
{
    if (description.Size == 0)
    {
        throw std::invalid_argument("Buffer size must be greater than zero.");
    }

    if (!description.Usage)
    {
        throw std::invalid_argument("Buffer usage flags must not be empty.");
    }

    if (description.MinimumAlignment != 0 && !std::has_single_bit(description.MinimumAlignment))
    {
        throw std::invalid_argument("Buffer minimum alignment must be zero or a power of two.");
    }

    const VmaAllocator allocator = context.Allocator().get();

    if (!allocator)
    {
        throw std::invalid_argument("Cannot create a buffer without a valid VMA allocator.");
    }

    VkBufferCreateInfo bufferInformation{};
    bufferInformation.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInformation.size = description.Size;
    bufferInformation.usage = static_cast<VkBufferUsageFlags>(description.Usage);
    bufferInformation.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInformation{};

    switch (description.MemoryUsage)
    {
    case BufferMemoryUsage::Device:
        allocationInformation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;

    case BufferMemoryUsage::Upload:
        allocationInformation.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInformation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        break;

    case BufferMemoryUsage::Readback:
        allocationInformation.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInformation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        break;

    default:
        throw std::invalid_argument("Unsupported buffer memory usage.");
    }

    VkBuffer buffer{};
    VmaAllocation allocation = nullptr;

    VkResult result = VK_SUCCESS;

    if (description.MinimumAlignment == 0)
    {
        result = vmaCreateBuffer(allocator, &bufferInformation, &allocationInformation, &buffer, &allocation, nullptr);
    }
    else
    {
        result = vmaCreateBufferWithAlignment(allocator, &bufferInformation, &allocationInformation,
                                              description.MinimumAlignment, &buffer, &allocation, nullptr);
    }

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create buffer: size=" + std::to_string(description.Size) +
                                 ", usage=" + std::to_string(static_cast<VkBufferUsageFlags>(description.Usage)) +
                                 ", VkResult=" + std::to_string(static_cast<int>(result)) + ".");
    }

    vk::DeviceAddress deviceAddress = 0;

    try
    {
        if (description.Usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)
        {
            vk::BufferDeviceAddressInfo addressInformation;
            addressInformation.setBuffer(vk::Buffer(buffer));

            deviceAddress = context.Device().getBufferAddress(addressInformation);

            if (deviceAddress == 0)
            {
                throw std::runtime_error("Buffer device address is zero.");
            }

            if (description.MinimumAlignment != 0 && deviceAddress % description.MinimumAlignment != 0)
            {
                throw std::runtime_error("Buffer device address does not satisfy the requested minimum alignment.");
            }
        }
    }
    catch (...)
    {
        vmaDestroyBuffer(allocator, buffer, allocation);
        throw;
    }

    m_allocator = allocator;
    m_buffer = buffer;
    m_allocation = allocation;
    m_size = description.Size;
    m_usage = description.Usage;
    m_memoryUsage = description.MemoryUsage;
    m_deviceAddress = deviceAddress;
}

vk::DeviceAddress Buffer::DeviceAddress() const
{
    if (!m_buffer)
    {
        throw std::logic_error("Cannot query the device address of an empty or moved-from buffer.");
    }

    if (!(m_usage & vk::BufferUsageFlagBits::eShaderDeviceAddress))
    {
        throw std::logic_error("Buffer was not created with shader device address usage.");
    }

    return m_deviceAddress;
}

void Buffer::Write(std::span<const std::byte> data, vk::DeviceSize offset)
{
    ValidateRange(data.size(), offset);

    if (m_memoryUsage != BufferMemoryUsage::Upload)
    {
        throw std::logic_error("Buffer::Write requires Upload memory usage.");
    }

    if (data.empty())
    {
        return;
    }

    const vk::DeviceSize byteCount = static_cast<vk::DeviceSize>(data.size());

    const VkResult result = vmaCopyMemoryToAllocation(m_allocator, data.data(), m_allocation, offset, byteCount);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(
            "Failed to write buffer memory. VkResult: " + std::to_string(static_cast<int>(result)) + ".");
    }
}

void Buffer::Read(std::span<std::byte> destination, vk::DeviceSize offset) const
{
    ValidateRange(destination.size(), offset);

    if (m_memoryUsage != BufferMemoryUsage::Readback)
    {
        throw std::logic_error("Buffer::Read requires Readback memory usage.");
    }

    if (destination.empty())
    {
        return;
    }

    const vk::DeviceSize byteCount = static_cast<vk::DeviceSize>(destination.size());

    const VkResult result = vmaCopyAllocationToMemory(m_allocator, m_allocation, offset, destination.data(), byteCount);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to read buffer memory. VkResult: " + std::to_string(static_cast<int>(result)) +
                                 ".");
    }
}

void Buffer::ValidateRange(std::size_t byteCount, vk::DeviceSize offset) const
{
    if (!m_buffer)
    {
        throw std::logic_error("Cannot access an empty or moved-from buffer.");
    }

    if (offset > m_size || std::cmp_greater(byteCount, m_size - offset))
    {
        throw std::out_of_range("Requested byte range exceeds the buffer size.");
    }
}

void Buffer::Release() noexcept
{
    if (m_buffer)
    {
        vmaDestroyBuffer(m_allocator, static_cast<VkBuffer>(m_buffer), m_allocation);
    }

    m_allocator = nullptr;
    m_buffer = nullptr;
    m_allocation = nullptr;
    m_size = 0;
    m_usage = vk::BufferUsageFlags{};
    m_memoryUsage = BufferMemoryUsage::Device;
    m_deviceAddress = 0;
}
} // namespace raylidar::vulkan
