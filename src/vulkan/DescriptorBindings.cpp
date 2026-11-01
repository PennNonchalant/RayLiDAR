#include "DescriptorBindings.h"

#include "Buffer.h"
#include "VulkanContext.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace raylidar::vulkan
{
DescriptorBindings::DescriptorBindings(const VulkanContext &context, const ComputePipeline &pipeline)
{
    Initialize(context, pipeline);
}

DescriptorBindings::DescriptorBindings(DescriptorBindings &&other) noexcept
{
    *this = std::move(other);
}

DescriptorBindings &DescriptorBindings::operator=(DescriptorBindings &&other) noexcept
{
    if (this != &other)
    {
        m_descriptorPool = std::move(other.m_descriptorPool);

        m_device = std::exchange(other.m_device, nullptr);
        m_setDescriptions = std::move(other.m_setDescriptions);
        m_descriptorSets = std::move(other.m_descriptorSets);
    }

    return *this;
}

void DescriptorBindings::Initialize(const VulkanContext &context, const ComputePipeline &pipeline)
{
    const auto &device = context.Device();
    const std::uint32_t setCount = pipeline.DescriptorSetCount();

    m_device = &device;

    if (setCount == 0)
    {
        return;
    }

    m_setDescriptions.resize(setCount);
    m_descriptorSets.resize(setCount);

    std::vector<vk::DescriptorSetLayout> layouts;
    layouts.reserve(setCount);

    std::vector<vk::DescriptorPoolSize> poolSizes;

    for (std::uint32_t setIndex = 0; setIndex < setCount; ++setIndex)
    {
        layouts.push_back(pipeline.DescriptorSetLayout(setIndex));

        const auto bindings = pipeline.Bindings(setIndex);
        m_setDescriptions[setIndex].Bindings.assign(bindings.begin(), bindings.end());

        for (const auto &binding : bindings)
        {
            switch (binding.descriptorType)
            {
            case vk::DescriptorType::eUniformBuffer:
            case vk::DescriptorType::eStorageBuffer:
            case vk::DescriptorType::eAccelerationStructureKHR:
                break;

            default:
                throw std::invalid_argument("Unsupported descriptor type for DescriptorBindings.");
            }

            const auto iterator = std::ranges::find(poolSizes, binding.descriptorType, &vk::DescriptorPoolSize::type);

            if (iterator != poolSizes.end())
            {
                iterator->descriptorCount += binding.descriptorCount;
            }
            else
            {
                poolSizes.emplace_back(binding.descriptorType, binding.descriptorCount);
            }
        }
    }

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(setCount).setPoolSizes(poolSizes);

    m_descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocationInformation;
    allocationInformation.setDescriptorPool(*m_descriptorPool).setSetLayouts(layouts);

    const vk::Result result =
        (*device).allocateDescriptorSets(&allocationInformation, m_descriptorSets.data(), *device.getDispatcher());

    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error(
            "Failed to allocate descriptor sets. VkResult: " + std::to_string(static_cast<int>(result)) + ".");
    }
}

void DescriptorBindings::WriteBuffer(std::uint32_t setIndex, std::uint32_t binding, const Buffer &buffer,
                                     vk::DeviceSize offset, vk::DeviceSize range, std::uint32_t arrayElement)
{
    const auto &layoutBinding = FindBinding(setIndex, binding, arrayElement);

    if (layoutBinding.descriptorType != vk::DescriptorType::eUniformBuffer &&
        layoutBinding.descriptorType != vk::DescriptorType::eStorageBuffer)
    {
        throw std::invalid_argument("WriteBuffer requires a uniform buffer or storage buffer binding.");
    }

    if (offset >= buffer.Size() || range == 0 || (range != VK_WHOLE_SIZE && range > buffer.Size() - offset))
    {
        throw std::out_of_range("Descriptor buffer range must be nonempty and within the buffer.");
    }

    vk::DescriptorBufferInfo bufferInformation;
    bufferInformation.setBuffer(buffer.Handle()).setOffset(offset).setRange(range);

    vk::WriteDescriptorSet write;
    write.setDstSet(m_descriptorSets[setIndex])
        .setDstBinding(binding)
        .setDstArrayElement(arrayElement)
        .setDescriptorType(layoutBinding.descriptorType)
        .setBufferInfo(bufferInformation);

    m_device->updateDescriptorSets(write, {});
}

void DescriptorBindings::WriteAccelerationStructure(std::uint32_t setIndex, std::uint32_t binding,
                                                    vk::AccelerationStructureKHR accelerationStructure,
                                                    std::uint32_t arrayElement)
{
    const auto &layoutBinding = FindBinding(setIndex, binding, arrayElement);

    if (layoutBinding.descriptorType != vk::DescriptorType::eAccelerationStructureKHR)
    {
        throw std::invalid_argument("WriteAccelerationStructure requires an acceleration structure binding.");
    }

    vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructureInformation;
    accelerationStructureInformation.setAccelerationStructures(accelerationStructure);

    vk::WriteDescriptorSet write;
    write.setDstSet(m_descriptorSets[setIndex])
        .setDstBinding(binding)
        .setDstArrayElement(arrayElement)
        .setDescriptorType(layoutBinding.descriptorType)
        .setDescriptorCount(1)
        .setPNext(&accelerationStructureInformation);

    m_device->updateDescriptorSets(write, {});
}

const vk::DescriptorSetLayoutBinding &DescriptorBindings::FindBinding(std::uint32_t setIndex, std::uint32_t binding,
                                                                      std::uint32_t arrayElement) const
{
    const auto &bindings = m_setDescriptions.at(setIndex).Bindings;
    const auto iterator = std::ranges::find(bindings, binding, &vk::DescriptorSetLayoutBinding::binding);

    if (iterator == bindings.end())
    {
        throw std::invalid_argument("Descriptor binding does not exist in the set layout.");
    }

    if (arrayElement >= iterator->descriptorCount)
    {
        throw std::out_of_range("Descriptor array element exceeds the binding descriptor count.");
    }

    return *iterator;
}
} // namespace raylidar::vulkan