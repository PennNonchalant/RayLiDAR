#pragma once

#include "vulkan/Buffer.h"

#include <vulkan/vulkan_raii.hpp>

namespace raylidar::vulkan
{
class VulkanContext;
} // namespace raylidar::vulkan

namespace raylidar::rt
{
struct AccelerationStructureDescription
{
    vk::AccelerationStructureTypeKHR Type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    vk::DeviceSize Size = 0;
};

class AccelerationStructure
{
  public:
    AccelerationStructure(const vulkan::VulkanContext &context, const AccelerationStructureDescription &description);
    ~AccelerationStructure() noexcept = default;

    AccelerationStructure(const AccelerationStructure &) = delete;
    AccelerationStructure &operator=(const AccelerationStructure &) = delete;

    AccelerationStructure(AccelerationStructure &&other) noexcept = default;
    AccelerationStructure &operator=(AccelerationStructure &&other) noexcept;

    [[nodiscard]]
    vk::AccelerationStructureKHR Handle() const noexcept
    {
        return *m_accelerationStructure;
    }

    [[nodiscard]]
    vk::DeviceAddress DeviceAddress() const noexcept
    {
        return m_deviceAddress;
    }

    [[nodiscard]]
    vk::AccelerationStructureTypeKHR Type() const noexcept
    {
        return m_type;
    }

    [[nodiscard]]
    vk::DeviceSize Size() const noexcept
    {
        return m_buffer.Size();
    }

  private:
    vulkan::Buffer m_buffer;
    vk::raii::AccelerationStructureKHR m_accelerationStructure = nullptr;

    vk::AccelerationStructureTypeKHR m_type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    vk::DeviceAddress m_deviceAddress = 0;
};
} // namespace raylidar::rt