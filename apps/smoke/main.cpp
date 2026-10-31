#include <vulkan/vulkan_raii.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

std::vector<std::uint32_t> ReadShader(const char *path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        throw std::runtime_error("Cannot open SPIR-V file.");
    }

    const auto size = stream.tellg();
    if (size <= 0 || static_cast<std::size_t>(size) % sizeof(std::uint32_t) != 0)
    {
        throw std::runtime_error("Invalid SPIR-V file size.");
    }

    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(code.data()), size))
    {
        throw std::runtime_error("Cannot read SPIR-V file.");
    }
    return code;
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 2)
        {
            throw std::runtime_error("Usage: raylidar_smoke <shader.spv>");
        }

        const auto code = ReadShader(argv[1]);
        const vk::raii::Context context;
        if (context.enumerateInstanceVersion() < VK_API_VERSION_1_4)
        {
            throw std::runtime_error("Vulkan 1.4 loader required.");
        }

        const vk::ApplicationInfo application("RayLiDAR Smoke", 1, nullptr, 0, VK_API_VERSION_1_4);
        const vk::raii::Instance instance(context, vk::InstanceCreateInfo({}, &application));
        const std::vector<const char *> extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                                      VK_KHR_RAY_QUERY_EXTENSION_NAME,
                                                      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};

        for (const auto &physicalDevice : instance.enumeratePhysicalDevices())
        {
            const auto properties = physicalDevice.getProperties();
            const auto available = physicalDevice.enumerateDeviceExtensionProperties();
            std::cout << "Device: " << properties.deviceName.data() << " | Vulkan "
                      << VK_API_VERSION_MAJOR(properties.apiVersion) << '.'
                      << VK_API_VERSION_MINOR(properties.apiVersion) << '.'
                      << VK_API_VERSION_PATCH(properties.apiVersion) << '\n';
            if (properties.apiVersion < VK_API_VERSION_1_4)
            {
                std::cout << "  Skipped: Vulkan 1.4 required.\n";
                continue;
            }

            bool hasExtensions = true;
            for (const char *name : extensions)
            {
                if (!std::ranges::any_of(available, [&](const auto &extension) {
                        return std::string_view(extension.extensionName.data()) == name;
                    }))
                {
                    std::cout << "  Missing extension: " << name << '\n';
                    hasExtensions = false;
                }
            }
            if (!hasExtensions)
            {
                continue;
            }

            const auto features = physicalDevice.getFeatures2<
                vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR>();
            if (!features.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress ||
                !features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering ||
                !features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 ||
                !features.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure ||
                !features.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery)
            {
                std::cout << "  Required features: bufferDeviceAddress="
                          << features.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress
                          << ", dynamicRendering="
                          << features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering
                          << ", synchronization2="
                          << features.get<vk::PhysicalDeviceVulkan13Features>().synchronization2
                          << ", accelerationStructure="
                          << features.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure
                          << ", rayQuery=" << features.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery << '\n';
                continue;
            }

            const auto queues = physicalDevice.getQueueFamilyProperties();
            const auto queue = std::ranges::find_if(queues, [](const auto &family) {
                return family.queueCount > 0 && (family.queueFlags & vk::QueueFlagBits::eCompute);
            });
            if (queue == queues.end())
            {
                std::cout << "  Skipped: no compute queue.\n";
                continue;
            }

            const auto queueIndex = static_cast<std::uint32_t>(queue - queues.begin());
            const float priority = 1.0f;
            const vk::DeviceQueueCreateInfo queueInfo({}, queueIndex, 1, &priority);
            vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceVulkan12Features,
                               vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                               vk::PhysicalDeviceRayQueryFeaturesKHR>
                deviceInfo;
            deviceInfo.get<vk::DeviceCreateInfo>().setQueueCreateInfos(queueInfo).setPEnabledExtensionNames(extensions);
            deviceInfo.get<vk::PhysicalDeviceVulkan12Features>().setBufferDeviceAddress(true);
            deviceInfo.get<vk::PhysicalDeviceVulkan13Features>().setDynamicRendering(true).setSynchronization2(true);
            deviceInfo.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().setAccelerationStructure(true);
            deviceInfo.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().setRayQuery(true);

            const vk::raii::Device device(physicalDevice, deviceInfo.get<vk::DeviceCreateInfo>());
            const vk::raii::ShaderModule shader(device, vk::ShaderModuleCreateInfo({}, code));
            const vk::raii::PipelineLayout layout(device, vk::PipelineLayoutCreateInfo());
            const vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, *shader, "main");
            const vk::raii::Pipeline pipeline(device, nullptr, vk::ComputePipelineCreateInfo({}, stage, *layout));
            const vk::raii::CommandPool pool(device, vk::CommandPoolCreateInfo({}, queueIndex));
            auto commands = device.allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(*pool, vk::CommandBufferLevel::ePrimary, 1));
            commands.front().begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            commands.front().bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
            commands.front().dispatch(1, 1, 1);
            commands.front().end();
            const vk::raii::Queue computeQueue(device, queueIndex, 0);
            const vk::CommandBufferSubmitInfo commandInfo(*commands.front());
            vk::SubmitInfo2 submitInfo;
            submitInfo.setCommandBufferInfos(commandInfo);
            computeQueue.submit2(submitInfo);
            computeQueue.waitIdle();

            std::cout << "PASS: " << properties.deviceName.data()
                      << " | Vulkan 1.4, ray query, dynamic rendering, Slang compute dispatch\n";
            return 0;
        }

        throw std::runtime_error("No Vulkan 1.4 device with the required ray query features and compute queue found.");
    }
    catch (const std::exception &exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
