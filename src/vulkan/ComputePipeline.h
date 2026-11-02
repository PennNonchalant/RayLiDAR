#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace raylidar::vulkan
{
class Shader;
class VulkanContext;

struct DescriptorSetDescription
{
    std::vector<vk::DescriptorSetLayoutBinding> Bindings;
};

struct ComputePipelineDescription
{
    std::string EntryPoint = "main";
    std::vector<DescriptorSetDescription> DescriptorSets;
    std::uint32_t PushConstantSize = 0;
};

class ComputePipeline
{
  public:
    ComputePipeline(const VulkanContext &context, const Shader &shader,
                    const ComputePipelineDescription &description = {});

    ~ComputePipeline() = default;

    ComputePipeline(const ComputePipeline &) = delete;
    ComputePipeline &operator=(const ComputePipeline &) = delete;

    ComputePipeline(ComputePipeline &&) noexcept = default;
    ComputePipeline &operator=(ComputePipeline &&) noexcept = default;

    [[nodiscard]]
    vk::Pipeline Handle() const noexcept
    {
        return *m_pipeline;
    }

    [[nodiscard]]
    vk::PipelineLayout Layout() const noexcept
    {
        return *m_pipelineLayout;
    }

    [[nodiscard]]
    std::uint32_t DescriptorSetCount() const noexcept
    {
        return static_cast<std::uint32_t>(m_descriptorSetLayouts.size());
    }

    [[nodiscard]]
    vk::DescriptorSetLayout DescriptorSetLayout(std::uint32_t setIndex) const;

    [[nodiscard]]
    std::span<const vk::DescriptorSetLayoutBinding> Bindings(std::uint32_t setIndex) const;

  private:
    void Initialize(const VulkanContext &context, const Shader &shader, const ComputePipelineDescription &description);

    std::vector<DescriptorSetDescription> m_descriptorSetDescriptions;
    std::vector<vk::raii::DescriptorSetLayout> m_descriptorSetLayouts;
    vk::raii::PipelineLayout m_pipelineLayout = nullptr;
    vk::raii::Pipeline m_pipeline = nullptr;
};
} // namespace raylidar::vulkan