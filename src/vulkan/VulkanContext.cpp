#include "VulkanContext.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace raylidar::vulkan
{
namespace
{
constexpr std::uint32_t RequiredApiVersion = VK_API_VERSION_1_4;
constexpr std::uint32_t ApplicationVersion =
    VK_MAKE_API_VERSION(0, RAYLIDAR_VERSION_MAJOR, RAYLIDAR_VERSION_MINOR, RAYLIDAR_VERSION_PATCH);

constexpr const char *ValidationLayerName = "VK_LAYER_KHRONOS_validation";

constexpr std::array<const char *, 3> RequiredDeviceExtensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                                                  VK_KHR_RAY_QUERY_EXTENSION_NAME,
                                                                  VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};

VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                               vk::DebugUtilsMessageTypeFlagsEXT,
                                               const vk::DebugUtilsMessengerCallbackDataEXT *callbackData,
                                               void *) noexcept
{
    const char *severity = "info";

    if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
    {
        severity = "error";
    }
    else if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
        severity = "warning";
    }

    const char *message = callbackData && callbackData->pMessage ? callbackData->pMessage : "(no validation message)";

    std::fprintf(stderr, "[Vulkan][%s] %s\n", severity, message);

    return VK_FALSE;
}

vk::DebugUtilsMessengerCreateInfoEXT MakeDebugMessengerCreateInfo()
{
    vk::DebugUtilsMessengerCreateInfoEXT information;

    information
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(DebugCallback);

    return information;
}
} // namespace

VulkanContext::VulkanContext(const ContextDescription &description)
{
    Initialize(description);
}

void VulkanContext::WaitIdle() const
{
    m_device.waitIdle();
}

void VulkanContext::Initialize(const ContextDescription &description)
{
    CreateInstance(description);
    SetupDebugMessenger(description.EnableValidationLayers);
    PickPhysicalDevice(description.PhysicalDeviceIndex);
    CreateLogicalDevice();
    CreateAllocator();
}

void VulkanContext::CreateInstance(const ContextDescription &description)
{
    // Vulkan 1.0 loaders may not expose this function.
    if (!m_context.getDispatcher()->vkEnumerateInstanceVersion ||
        m_context.enumerateInstanceVersion() < RequiredApiVersion)
    {
        throw std::runtime_error("RayLiDAR requires a Vulkan 1.4 loader.");
    }

    // Validate the requested layers.
    std::vector<const char *> enabledLayers;

    if (description.EnableValidationLayers)
    {
        const auto availableLayers = m_context.enumerateInstanceLayerProperties();

        const bool validationAvailable =
            std::ranges::any_of(availableLayers, [](const vk::LayerProperties &properties) {
                return std::string_view(properties.layerName.data()) == ValidationLayerName;
            });

        if (!validationAvailable)
        {
            throw std::runtime_error("VK_LAYER_KHRONOS_validation was requested but is "
                                     "unavailable. Install the validation layers or set "
                                     "EnableValidationLayers to false.");
        }

        enabledLayers.push_back(ValidationLayerName);
    }

    // Collect and deduplicate instance extension names.
    std::vector<std::string> extensionNames = description.AdditionalInstanceExtensions;

    if (description.EnableValidationLayers)
    {
        extensionNames.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::ranges::sort(extensionNames);

    extensionNames.erase(std::unique(extensionNames.begin(), extensionNames.end()), extensionNames.end());

    auto availableExtensions = m_context.enumerateInstanceExtensionProperties();

    // Enabled layers may provide additional instance extensions.
    for (const char *layerName : enabledLayers)
    {
        const std::string layerNameString(layerName);

        const auto layerExtensions = m_context.enumerateInstanceExtensionProperties(layerNameString);

        availableExtensions.insert(availableExtensions.end(), layerExtensions.begin(), layerExtensions.end());
    }

    for (const auto &requiredName : extensionNames)
    {
        const bool available =
            std::ranges::any_of(availableExtensions, [&requiredName](const vk::ExtensionProperties &properties) {
                return std::string_view(properties.extensionName.data()) == requiredName;
            });

        if (!available)
        {
            throw std::runtime_error("Required instance extension is unavailable: " + requiredName);
        }
    }

    // Keep extensionNames unchanged while these pointers are in use.
    std::vector<const char *> enabledExtensions;
    enabledExtensions.reserve(extensionNames.size());

    for (const auto &name : extensionNames)
    {
        enabledExtensions.push_back(name.c_str());
    }

    vk::ApplicationInfo applicationInformation;
    applicationInformation.setPApplicationName(description.ApplicationName.c_str())
        .setApplicationVersion(ApplicationVersion)
        .setPEngineName("RayLiDAR")
        .setEngineVersion(ApplicationVersion)
        .setApiVersion(RequiredApiVersion);

    auto debugInformation = MakeDebugMessengerCreateInfo();

    vk::InstanceCreateInfo instanceInformation;
    instanceInformation.setPApplicationInfo(&applicationInformation)
        .setPEnabledLayerNames(enabledLayers)
        .setPEnabledExtensionNames(enabledExtensions);

    if (description.EnableValidationLayers)
    {
        // Capture validation messages during instance creation.
        instanceInformation.setPNext(&debugInformation);
    }

    m_instance = vk::raii::Instance(m_context, instanceInformation);
}

void VulkanContext::SetupDebugMessenger(bool enableValidationLayers)
{
    if (!enableValidationLayers)
    {
        return;
    }

    const auto debugInformation = MakeDebugMessengerCreateInfo();

    m_debugMessenger = vk::raii::DebugUtilsMessengerEXT(m_instance, debugInformation);
}

void VulkanContext::PickPhysicalDevice(std::optional<std::uint32_t> requestedDeviceIndex)
{
    auto physicalDevices = m_instance.enumeratePhysicalDevices();

    if (physicalDevices.empty())
    {
        throw std::runtime_error("No Vulkan physical devices were found.");
    }

    if (requestedDeviceIndex && *requestedDeviceIndex >= physicalDevices.size())
    {
        throw std::runtime_error(
            "PhysicalDeviceIndex " + std::to_string(*requestedDeviceIndex) +
            " is out of range. Enumerated device count: " + std::to_string(physicalDevices.size()) + ".");
    }

    std::optional<std::size_t> selectedDeviceIndex;
    std::uint32_t selectedQueueFamilyIndex = std::numeric_limits<std::uint32_t>::max();
    int selectedDevicePriority = -1;

    std::ostringstream rejectionDetails;

    for (std::size_t deviceIndex = 0; deviceIndex < physicalDevices.size(); ++deviceIndex)
    {
        if (requestedDeviceIndex && deviceIndex != *requestedDeviceIndex)
        {
            continue;
        }

        const auto &physicalDevice = physicalDevices[deviceIndex];
        const auto properties = physicalDevice.getProperties();

        std::vector<std::string> rejectionReasons;

        if (properties.apiVersion < RequiredApiVersion)
        {
            rejectionReasons.emplace_back("Vulkan 1.4 is not supported");
        }

        const auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

        for (const char *requiredExtension : RequiredDeviceExtensions)
        {
            const bool available = std::ranges::any_of(
                availableExtensions, [requiredExtension](const vk::ExtensionProperties &extensionProperties) {
                    return std::string_view(extensionProperties.extensionName.data()) == requiredExtension;
                });

            if (!available)
            {
                rejectionReasons.emplace_back(std::string("missing device extension ") + requiredExtension);
            }
        }

        std::optional<std::uint32_t> computeQueueFamilyIndex;

        const auto queueFamilies = physicalDevice.getQueueFamilyProperties();

        for (std::size_t familyIndex = 0; familyIndex < queueFamilies.size(); ++familyIndex)
        {
            const auto &family = queueFamilies[familyIndex];

            if (family.queueCount > 0 && (family.queueFlags & vk::QueueFlagBits::eCompute))
            {
                computeQueueFamilyIndex = static_cast<std::uint32_t>(familyIndex);
                break;
            }
        }

        if (!computeQueueFamilyIndex)
        {
            rejectionReasons.emplace_back("no compute-capable queue family");
        }

        // The API and extension requirements must pass before
        // querying the corresponding feature chain.
        if (rejectionReasons.empty())
        {
            const auto supportedFeatures = physicalDevice.getFeatures2<
                vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR>();

            const auto &features12 = supportedFeatures.get<vk::PhysicalDeviceVulkan12Features>();

            const auto &features13 = supportedFeatures.get<vk::PhysicalDeviceVulkan13Features>();

            const auto &accelerationStructureFeatures =
                supportedFeatures.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();

            const auto &rayQueryFeatures = supportedFeatures.get<vk::PhysicalDeviceRayQueryFeaturesKHR>();

            if (!features12.bufferDeviceAddress)
            {
                rejectionReasons.emplace_back("bufferDeviceAddress is not supported");
            }

            if (!features13.synchronization2)
            {
                rejectionReasons.emplace_back("synchronization2 is not supported");
            }

            // Preserve the existing smoke test's feature baseline.
            if (!features13.dynamicRendering)
            {
                rejectionReasons.emplace_back("dynamicRendering is not supported");
            }

            if (!accelerationStructureFeatures.accelerationStructure)
            {
                rejectionReasons.emplace_back("accelerationStructure is not supported");
            }

            if (!rayQueryFeatures.rayQuery)
            {
                rejectionReasons.emplace_back("rayQuery is not supported");
            }
        }

        if (!rejectionReasons.empty())
        {
            rejectionDetails << "\n  [" << deviceIndex << "] " << properties.deviceName.data() << ":";

            for (const auto &reason : rejectionReasons)
            {
                rejectionDetails << "\n    - " << reason;
            }

            continue;
        }

        // Prefer discrete GPUs. Preserve enumeration order on ties.
        const int priority = properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? 1 : 0;

        if (!selectedDeviceIndex || priority > selectedDevicePriority)
        {
            selectedDeviceIndex = deviceIndex;
            selectedQueueFamilyIndex = *computeQueueFamilyIndex;
            selectedDevicePriority = priority;
        }
    }

    if (!selectedDeviceIndex)
    {
        const std::string failureMessage = requestedDeviceIndex ? "The requested physical device does not satisfy "
                                                                  "RayLiDAR requirements."
                                                                : "No physical device satisfies RayLiDAR requirements.";

        throw std::runtime_error(failureMessage + rejectionDetails.str());
    }

    m_physicalDevice = std::move(physicalDevices[*selectedDeviceIndex]);

    m_computeQueueFamilyIndex = selectedQueueFamilyIndex;

    const auto propertyChain =
        m_physicalDevice
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();

    m_physicalDeviceProperties = propertyChain.get<vk::PhysicalDeviceProperties2>().properties;

    m_accelerationStructureProperties = propertyChain.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();

    // The cached structure must not point into a local chain.
    m_accelerationStructureProperties.pNext = nullptr;
}

void VulkanContext::CreateLogicalDevice()
{
    const float queuePriority = 1.0f;

    vk::DeviceQueueCreateInfo queueInformation;
    queueInformation.setQueueFamilyIndex(m_computeQueueFamilyIndex)
        .setQueueCount(1)
        .setPQueuePriorities(&queuePriority);

    // Enable only the features required by the project.
    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceRayQueryFeaturesKHR>
        deviceInformation;

    deviceInformation.get<vk::DeviceCreateInfo>()
        .setQueueCreateInfos(queueInformation)
        .setPEnabledExtensionNames(RequiredDeviceExtensions);

    deviceInformation.get<vk::PhysicalDeviceVulkan12Features>().setBufferDeviceAddress(true);

    deviceInformation.get<vk::PhysicalDeviceVulkan13Features>().setSynchronization2(true).setDynamicRendering(true);

    deviceInformation.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().setAccelerationStructure(true);

    deviceInformation.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().setRayQuery(true);

    m_device = vk::raii::Device(m_physicalDevice, deviceInformation.get<vk::DeviceCreateInfo>());

    m_computeQueue = vk::raii::Queue(m_device, m_computeQueueFamilyIndex, 0);
}

void VulkanContext::CreateAllocator()
{
    VmaVulkanFunctions vulkanFunctions{};

    vulkanFunctions.vkGetInstanceProcAddr = m_context.getDispatcher()->vkGetInstanceProcAddr;

    vulkanFunctions.vkGetDeviceProcAddr = m_device.getDispatcher()->vkGetDeviceProcAddr;

    if (!vulkanFunctions.vkGetInstanceProcAddr || !vulkanFunctions.vkGetDeviceProcAddr)
    {
        throw std::runtime_error("Vulkan function entry points required by VMA "
                                 "are unavailable.");
    }

    VmaAllocatorCreateInfo allocatorInformation{};
    allocatorInformation.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    allocatorInformation.physicalDevice = static_cast<VkPhysicalDevice>(*m_physicalDevice);

    allocatorInformation.device = static_cast<VkDevice>(*m_device);

    allocatorInformation.instance = static_cast<VkInstance>(*m_instance);

    allocatorInformation.vulkanApiVersion = RequiredApiVersion;
    allocatorInformation.pVulkanFunctions = &vulkanFunctions;

    VmaAllocator allocator = nullptr;

    const VkResult result = vmaCreateAllocator(&allocatorInformation, &allocator);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create VMA allocator. VkResult: " +
                                 std::to_string(static_cast<int>(result)));
    }

    m_allocator.reset(allocator);
}
} // namespace raylidar::vulkan
