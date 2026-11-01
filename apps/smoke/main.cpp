#include "vulkan/Buffer.h"
#include "vulkan/ComputePipeline.h"
#include "vulkan/DescriptorBindings.h"
#include "vulkan/Shader.h"
#include "vulkan/VulkanContext.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 2)
        {
            throw std::invalid_argument("Usage: raylidar_smoke <shader.spv>");
        }

        {
            raylidar::vulkan::ContextDescription description;
            description.ApplicationName = "RayLiDAR Context Smoke";
            description.EnableValidationLayers = true;

            const raylidar::vulkan::VulkanContext context(description);

            if (!*context.Device() || !*context.ComputeQueue() || !context.Allocator())
            {
                throw std::runtime_error("Context contains a null device, queue, or allocator.");
            }

            if (context.ComputeQueueFamilyIndex() == std::numeric_limits<std::uint32_t>::max())
            {
                throw std::runtime_error("Compute queue family index was not initialized.");
            }

            const auto &properties = context.PhysicalDeviceProperties();

            std::cout << "Device: " << properties.deviceName.data() << '\n'
                      << "Vulkan: " << VK_API_VERSION_MAJOR(properties.apiVersion) << '.'
                      << VK_API_VERSION_MINOR(properties.apiVersion) << '.'
                      << VK_API_VERSION_PATCH(properties.apiVersion) << '\n'
                      << "Compute queue family: " << context.ComputeQueueFamilyIndex() << '\n'
                      << "AS scratch alignment: "
                      << context.AccelerationStructureProperties().minAccelerationStructureScratchOffsetAlignment
                      << '\n';

            {
                const std::array<std::uint32_t, 4> data = {1, 2, 3, 4};
                const auto bytes = std::as_bytes(std::span(data));

                raylidar::vulkan::BufferDescription bufferDescription;
                bufferDescription.Size = static_cast<vk::DeviceSize>(bytes.size());
                bufferDescription.Usage = vk::BufferUsageFlagBits::eTransferSrc;
                bufferDescription.MemoryUsage = raylidar::vulkan::BufferMemoryUsage::Upload;

                raylidar::vulkan::Buffer buffer(context, bufferDescription);
                buffer.Write(bytes);

                const vk::Buffer originalHandle = buffer.Handle();

                // Move construction.
                raylidar::vulkan::Buffer movedBuffer(std::move(buffer));

                // Move assignment also releases the destination's previous allocation.
                raylidar::vulkan::Buffer assignedBuffer(context, bufferDescription);
                assignedBuffer = std::move(movedBuffer);

                if (!assignedBuffer.Handle() || assignedBuffer.Handle() != originalHandle ||
                    assignedBuffer.Size() != bufferDescription.Size)
                {
                    throw std::runtime_error("Buffer initialization or move failed.");
                }

                // Confirm the destination remains usable after moving.
                assignedBuffer.Write(bytes);
            }

            std::cout << "Buffer creation, write, move, and destruction completed.\n";

            {
                const raylidar::vulkan::Shader shader(context, argv[1]);

                raylidar::vulkan::ComputePipelineDescription pipelineDescription;
                pipelineDescription.DescriptorSets.resize(2);

                for (auto &setDescription : pipelineDescription.DescriptorSets)
                {
                    setDescription.Bindings.emplace_back(0, vk::DescriptorType::eStorageBuffer, 1,
                                                         vk::ShaderStageFlagBits::eCompute);
                }

                const raylidar::vulkan::ComputePipeline pipeline(context, shader, pipelineDescription);

                if (!shader.Handle() || !pipeline.Handle() || !pipeline.Layout())
                {
                    throw std::runtime_error("Shader or compute pipeline initialization failed.");
                }

                raylidar::vulkan::BufferDescription bufferDescription;
                bufferDescription.Size = 16;
                bufferDescription.Usage = vk::BufferUsageFlagBits::eStorageBuffer;

                const raylidar::vulkan::Buffer buffer(context, bufferDescription);

                raylidar::vulkan::DescriptorBindings bindings(context, pipeline);

                if (bindings.Handles().size() != 2 || !bindings.Handle(0) || !bindings.Handle(1))
                {
                    throw std::runtime_error("Descriptor set initialization failed.");
                }

                bindings.WriteBuffer(0, 0, buffer);
                bindings.WriteBuffer(1, 0, buffer);
            }

            std::cout << "Shader, compute pipeline, and descriptor bindings creation and destruction completed.\n"
                      << "Two descriptor sets allocated and written.\n";

            context.WaitIdle();
        }

        // Context destruction has completed.
        std::cout << "Context initialization and destruction completed.\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Vulkan smoke failed: " << exception.what() << '\n';
        return 1;
    }
}