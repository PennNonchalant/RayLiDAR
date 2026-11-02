#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace raylidar::vulkan
{
class VulkanContext;

class CommandContext
{
  public:
    explicit CommandContext(const VulkanContext &context);
    ~CommandContext() noexcept;

    CommandContext(const CommandContext &) = delete;
    CommandContext &operator=(const CommandContext &) = delete;

    CommandContext(CommandContext &&) = delete;
    CommandContext &operator=(CommandContext &&) = delete;

    [[nodiscard]]
    const vk::raii::CommandBuffer &Begin();

    void End();
    void Submit();
    void Wait();
    void SubmitAndWait();

  private:
    enum class State
    {
        Idle,
        Recording,
        Ready,
        Pending
    };

    void Initialize();

    const VulkanContext &m_context;

    vk::raii::CommandPool m_commandPool = nullptr;
    vk::raii::CommandBuffer m_commandBuffer = nullptr;
    vk::raii::Fence m_fence = nullptr;

    State m_state = State::Idle;
};
} // namespace raylidar::vulkan