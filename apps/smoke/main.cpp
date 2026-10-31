#include "vulkan/VulkanContext.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

int main()
{
    try
    {
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

            context.WaitIdle();
        }

        // Context destruction has completed.
        std::cout << "Context initialization and destruction completed.\n";
        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "Context smoke failed: " << exception.what() << '\n';
        return 1;
    }
}