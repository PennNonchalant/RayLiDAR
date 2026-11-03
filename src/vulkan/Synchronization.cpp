#pragma once

#include "Synchronization.h"

namespace raylidar::vulkan
{
void RecordMemoryBarrier(const vk::raii::CommandBuffer &commandBuffer, vk::PipelineStageFlags2 sourceStage,
                         vk::AccessFlags2 sourceAccess, vk::PipelineStageFlags2 destinationStage,
                         vk::AccessFlags2 destinationAccess)
{
    vk::MemoryBarrier2 barrier;
    barrier.setSrcStageMask(sourceStage)
        .setSrcAccessMask(sourceAccess)
        .setDstStageMask(destinationStage)
        .setDstAccessMask(destinationAccess);

    vk::DependencyInfo dependency;
    dependency.setMemoryBarriers(barrier);

    commandBuffer.pipelineBarrier2(dependency);
}
} // namespace raylidar::vulkan
