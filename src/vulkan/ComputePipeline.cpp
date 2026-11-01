#include "ComputePipeline.h"

#include "Shader.h"
#include "VulkanContext.h"

#include <stdexcept>
#include <vector>

namespace raylidar::vulkan
{
ComputePipeline::ComputePipeline(const VulkanContext &context, const Shader &shader,
                                 const ComputePipelineDescription &description)
{
    Initialize(context, shader, description);
}

void ComputePipeline::Initialize(const VulkanContext &context, const Shader &shader,
                                 const ComputePipelineDescription &description)
{
    if (description.EntryPoint.empty())
    {
        throw std::invalid_argument("Compute pipeline entry point must not be empty.");
    }

    const auto &limits = context.PhysicalDeviceProperties().limits;

    if (description.DescriptorSets.size() > limits.maxBoundDescriptorSets)
    {
        throw std::invalid_argument("Descriptor set count exceeds the device limit.");
    }

    if (description.PushConstantSize % 4 != 0 || description.PushConstantSize > limits.maxPushConstantsSize)
    {
        throw std::invalid_argument("Push constant size must be a multiple of four and within the device limit.");
    }

    m_descriptorSetDescriptions = description.DescriptorSets;
    m_descriptorSetLayouts.reserve(m_descriptorSetDescriptions.size());

    std::vector<vk::DescriptorSetLayout> layoutHandles;
    layoutHandles.reserve(m_descriptorSetDescriptions.size());

    for (const auto &setDescription : m_descriptorSetDescriptions)
    {
        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.setBindings(setDescription.Bindings);

        auto &layout = m_descriptorSetLayouts.emplace_back(context.Device(), layoutInfo);

        layoutHandles.push_back(*layout);
    }

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(layoutHandles);

    const vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, description.PushConstantSize);

    if (description.PushConstantSize != 0)
    {
        pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);
    }

    m_pipelineLayout = vk::raii::PipelineLayout(context.Device(), pipelineLayoutInfo);

    vk::PipelineShaderStageCreateInfo stageInformation;
    stageInformation.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(shader.Handle())
        .setPName(description.EntryPoint.c_str());

    vk::ComputePipelineCreateInfo pipelineInformation;
    pipelineInformation.setStage(stageInformation).setLayout(*m_pipelineLayout);

    m_pipeline = vk::raii::Pipeline(context.Device(), nullptr, pipelineInformation);
}

vk::DescriptorSetLayout ComputePipeline::DescriptorSetLayout(std::uint32_t setIndex) const
{
    return *m_descriptorSetLayouts.at(setIndex);
}

std::span<const vk::DescriptorSetLayoutBinding> ComputePipeline::Bindings(std::uint32_t setIndex) const
{
    return m_descriptorSetDescriptions.at(setIndex).Bindings;
}
} // namespace raylidar::vulkan