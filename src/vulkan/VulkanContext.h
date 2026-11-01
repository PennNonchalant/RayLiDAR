#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace raylidar::vulkan
{
struct VmaAllocatorDeleter
{
    void operator()(VmaAllocator allocator) const noexcept
    {
        if (allocator)
        {
            vmaDestroyAllocator(allocator);
        }
    }
};

using UniqueVmaAllocator = std::unique_ptr<VmaAllocator_T, VmaAllocatorDeleter>;

struct ContextDescription
{
    std::string ApplicationName = "RayLiDAR";
    bool EnableValidationLayers = true;
    std::optional<std::uint32_t> PhysicalDeviceIndex;
    std::vector<std::string> AdditionalInstanceExtensions;
};

class VulkanContext
{
  public:
    explicit VulkanContext(const ContextDescription &description = {});

    ~VulkanContext() = default;

    VulkanContext(const VulkanContext &) = delete;
    VulkanContext &operator=(const VulkanContext &) = delete;

    VulkanContext(VulkanContext &&) = delete;
    VulkanContext &operator=(VulkanContext &&) = delete;

    [[nodiscard]]
    const vk::raii::Device &Device() const noexcept
    {
        return m_device;
    }

    [[nodiscard]]
    VmaAllocator Allocator() const noexcept
    {
        return m_allocator.get();
    }

    [[nodiscard]]
    const vk::raii::Queue &ComputeQueue() const noexcept
    {
        return m_computeQueue;
    }

    [[nodiscard]]
    std::uint32_t ComputeQueueFamilyIndex() const noexcept
    {
        return m_computeQueueFamilyIndex;
    }

    [[nodiscard]]
    const vk::PhysicalDeviceProperties &PhysicalDeviceProperties() const noexcept
    {
        return m_physicalDeviceProperties;
    }

    [[nodiscard]]
    const vk::PhysicalDeviceAccelerationStructurePropertiesKHR &AccelerationStructureProperties() const noexcept
    {
        return m_accelerationStructureProperties;
    }

    void WaitIdle() const;

  private:
    void Initialize(const ContextDescription &description);

    void CreateInstance(const ContextDescription &description);
    void SetupDebugMessenger(bool enableValidationLayers);
    void PickPhysicalDevice(std::optional<std::uint32_t> requestedDeviceIndex);
    void CreateLogicalDevice();
    void CreateAllocator();

    vk::raii::Context m_context;
    vk::raii::Instance m_instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT m_debugMessenger = nullptr;

    vk::raii::PhysicalDevice m_physicalDevice = nullptr;
    vk::PhysicalDeviceProperties m_physicalDeviceProperties{};
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR m_accelerationStructureProperties{};

    std::uint32_t m_computeQueueFamilyIndex = std::numeric_limits<std::uint32_t>::max();

    vk::raii::Device m_device = nullptr;
    vk::raii::Queue m_computeQueue = nullptr;

    UniqueVmaAllocator m_allocator = nullptr;
};
} // namespace raylidar::vulkan
