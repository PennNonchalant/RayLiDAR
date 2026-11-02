cmake_minimum_required(VERSION 3.29)
project(RayLiDAR VERSION 0.1.0 LANGUAGES CXX)

if(WIN32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR
        "RayLiDAR requires a 64-bit Windows build. Initialize the VS Developer "
        "PowerShell with -Arch amd64 -HostArch amd64, then run "
        "cmake --fresh --preset windows-debug.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")


find_package(VulkanHeaders 1.4 CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

enable_testing()
add_subdirectory(src)
add_subdirectory(apps/cli)

include(cmake/SlangShaders.cmake)

add_executable(raylidar_smoke
    apps/smoke/main.cpp
)

target_link_libraries(raylidar_smoke PRIVATE
    raylidar_vulkan
    raylidar_rt
)

foreach(target IN ITEMS raylidar_vulkan raylidar_rt raylidar raylidar_cli raylidar_smoke)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc /utf-8)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

set(SHADER_SOURCE_DIR "${PROJECT_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${PROJECT_BINARY_DIR}/shaders")
raylidar_add_shaders(raylidar_shaders "${SHADER_SOURCE_DIR}" "${SHADER_BINARY_DIR}")
add_dependencies(raylidar_smoke raylidar_shaders)

add_test(
    NAME vulkan_context_smoke
    COMMAND raylidar_smoke "${SHADER_BINARY_DIR}/Smoke.spv"
)

set_tests_properties(vulkan_context_smoke PROPERTIES
    TIMEOUT 30
    LABELS "gpu;vulkan"
    FAIL_REGULAR_EXPRESSION "\\[Vulkan\\]\\[error\\]"
)
