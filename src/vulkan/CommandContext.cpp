#include "CommandContext.h"

#include "VulkanContext.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace raylidar::vulkan
{
CommandContext::CommandContext(const VulkanContext &context) : m_context(context)
{
    Initialize();
}

CommandContext::~CommandContext() noexcept
{
    if (m_state != State::Pending)
    {
        return;
    }

    try
    {
        Wait();
    }
    catch (const vk::DeviceLostError &)
    {
        // Device loss permits cleanup of the submitted resources.
    }
    catch (...)
    {
        // Completion is unknown; destroying potentially pending resources is unsafe.
        std::terminate();
    }
}

void CommandContext::Initialize()
{
    const auto &device = m_context.Device();

    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(m_context.ComputeQueueFamilyIndex());

    m_commandPool = vk::raii::CommandPool(device, poolInfo);

    vk::CommandBufferAllocateInfo allocationInformation;
    allocationInformation.setCommandPool(*m_commandPool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);

    auto commandBuffers = device.allocateCommandBuffers(allocationInformation);
    m_commandBuffer = std::move(commandBuffers.front());

    m_fence = vk::raii::Fence(device, vk::FenceCreateInfo{});
}

const vk::raii::CommandBuffer &CommandContext::Begin()
{
    if (m_state != State::Idle && m_state != State::Ready)
    {
        throw std::logic_error("Begin requires an idle context or an unsubmitted batch.");
    }

    m_state = State::Idle;
    m_commandPool.reset({});

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    m_commandBuffer.begin(beginInfo);
    m_state = State::Recording;

    return m_commandBuffer;
}

void CommandContext::End()
{
    if (m_state != State::Recording)
    {
        throw std::logic_error("End requires an actively recording command buffer.");
    }

    m_state = State::Idle;
    m_commandBuffer.end();
    m_state = State::Ready;
}

void CommandContext::Submit()
{
    if (m_state != State::Ready)
    {
        throw std::logic_error("Submit requires a completed, unsubmitted command buffer.");
    }

    vk::CommandBufferSubmitInfo commandBufferInfo;
    commandBufferInfo.setCommandBuffer(*m_commandBuffer);

    vk::SubmitInfo2 submitInfo;
    submitInfo.setCommandBufferInfos(commandBufferInfo);

    m_context.Device().resetFences(*m_fence);
    m_context.ComputeQueue().submit2(submitInfo, *m_fence);

    m_state = State::Pending;
}

void CommandContext::Wait()
{
    if (m_state == State::Idle)
    {
        return;
    }

    if (m_state != State::Pending)
    {
        throw std::logic_error("Wait requires a submitted batch.");
    }

    const vk::Result result =
        m_context.Device().waitForFences(*m_fence, true, std::numeric_limits<std::uint64_t>::max());

    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Waiting for command submission timed out.");
    }

    m_state = State::Idle;
}

void CommandContext::SubmitAndWait()
{
    Submit();
    Wait();
}
} // namespace raylidar::vulkan