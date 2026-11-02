#pragma once

#include "ComputePipeline.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace raylidar::vulkan
{
class Buffer;
class VulkanContext;

class DescriptorBindings
{
  public:
    DescriptorBindings(const VulkanContext &context, const ComputePipeline &pipeline);

    ~DescriptorBindings() = default;

    DescriptorBindings(const DescriptorBindings &) = delete;
    DescriptorBindings &operator=(const DescriptorBindings &) = delete;

    DescriptorBindings(DescriptorBindings &&other) noexcept;
    DescriptorBindings &operator=(DescriptorBindings &&other) noexcept;

    [[nodiscard]]
    vk::DescriptorSet Handle(std::uint32_t setIndex) const
    {
        return m_descriptorSets.at(setIndex);
    }

    [[nodiscard]]
    std::span<const vk::DescriptorSet> Handles() const noexcept
    {
        return m_descriptorSets;
    }

    void WriteBuffer(std::uint32_t setIndex, std::uint32_t binding, const Buffer &buffer, vk::DeviceSize offset = 0,
                     vk::DeviceSize range = VK_WHOLE_SIZE, std::uint32_t arrayElement = 0);

    void WriteAccelerationStructure(std::uint32_t setIndex, std::uint32_t binding,
                                    vk::AccelerationStructureKHR accelerationStructure, std::uint32_t arrayElement = 0);

  private:
    void Initialize(const VulkanContext &context, const ComputePipeline &pipeline);

    [[nodiscard]]
    const vk::DescriptorSetLayoutBinding &FindBinding(std::uint32_t setIndex, std::uint32_t binding,
                                                      std::uint32_t arrayElement) const;

    const vk::raii::Device *m_device = nullptr;

    std::vector<DescriptorSetDescription> m_setDescriptions;

    vk::raii::DescriptorPool m_descriptorPool = nullptr;
    std::vector<vk::DescriptorSet> m_descriptorSets;
};
} // namespace raylidar::vulkan
