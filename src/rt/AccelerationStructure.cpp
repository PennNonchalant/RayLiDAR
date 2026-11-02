#include "AccelerationStructure.h"

#include "vulkan/VulkanContext.h"

#include <stdexcept>
#include <utility>

namespace raylidar::rt
{
namespace
{
vulkan::BufferDescription MakeBufferDescription(const AccelerationStructureDescription &description)
{
    if (description.Type != vk::AccelerationStructureTypeKHR::eBottomLevel &&
        description.Type != vk::AccelerationStructureTypeKHR::eTopLevel)
    {
        throw std::invalid_argument("Acceleration structure type must be bottom-level or top-level.");
    }

    vulkan::BufferDescription bufferDescription;
    bufferDescription.Size = description.Size;
    bufferDescription.Usage =
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
    bufferDescription.MemoryUsage = vulkan::BufferMemoryUsage::Device;

    return bufferDescription;
}
} // namespace

AccelerationStructure &AccelerationStructure::operator=(AccelerationStructure &&other) noexcept
{
    if (this != &other)
    {
        m_accelerationStructure.clear();

        m_buffer = std::move(other.m_buffer);
        m_accelerationStructure = std::move(other.m_accelerationStructure);

        m_type = other.m_type;
        m_deviceAddress = std::exchange(other.m_deviceAddress, vk::DeviceAddress{0});
    }

    return *this;
}

AccelerationStructure::AccelerationStructure(const vulkan::VulkanContext &context,
                                             const AccelerationStructureDescription &description)
    : m_buffer(context, MakeBufferDescription(description)), m_type(description.Type)
{
    vk::AccelerationStructureCreateInfoKHR createInfo;
    createInfo.setBuffer(m_buffer.Handle()).setOffset(0).setSize(description.Size).setType(m_type);

    m_accelerationStructure = vk::raii::AccelerationStructureKHR(context.Device(), createInfo);

    vk::AccelerationStructureDeviceAddressInfoKHR addressInformation;
    addressInformation.setAccelerationStructure(*m_accelerationStructure);

    m_deviceAddress = context.Device().getAccelerationStructureAddressKHR(addressInformation);
}
} // namespace raylidar::rt