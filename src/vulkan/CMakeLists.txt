add_library(raylidar_vulkan STATIC
    Vma.cpp
    VulkanContext.cpp
    Buffer.cpp
    Shader.cpp
    ComputePipeline.cpp
    DescriptorBindings.cpp
    CommandContext.cpp
    Synchronization.cpp
)

target_compile_features(raylidar_vulkan PUBLIC cxx_std_20)

target_include_directories(raylidar_vulkan PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)

target_link_libraries(raylidar_vulkan
    PUBLIC
        Vulkan::Headers
        GPUOpen::VulkanMemoryAllocator
    PRIVATE
        ${CMAKE_DL_LIBS}
)

target_compile_definitions(raylidar_vulkan
    PUBLIC
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    PRIVATE
        RAYLIDAR_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        RAYLIDAR_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        RAYLIDAR_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

set_target_properties(raylidar_vulkan PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_precompile_headers(raylidar_vulkan PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../pch.h"
)
