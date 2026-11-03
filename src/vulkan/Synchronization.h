#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace raylidar::vulkan
{
void RecordMemoryBarrier(const vk::raii::CommandBuffer &commandBuffer, vk::PipelineStageFlags2 sourceStage,
                         vk::AccessFlags2 sourceAccess, vk::PipelineStageFlags2 destinationStage,
                         vk::AccessFlags2 destinationAccess);
}
