cmake_minimum_required(VERSION 3.29)
project(RayLiDAR VERSION 0.1.0 LANGUAGES CXX)

if(WIN32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR
        "RayLiDAR requires a 64-bit Windows build. Initialize the VS Developer "
        "PowerShell with -Arch amd64 -HostArch amd64, then run "
        "cmake --fresh --preset windows-debug.")
endif()

find_package(VulkanHeaders 1.4 CONFIG REQUIRED)
include(cmake/SlangShaders.cmake)

add_executable(raylidar_smoke apps/smoke/main.cpp)
target_compile_features(raylidar_smoke PRIVATE cxx_std_20)
set_target_properties(raylidar_smoke PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(raylidar_smoke PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})
if(MSVC)
    target_compile_options(raylidar_smoke PRIVATE /W4 /permissive- /EHsc /utf-8)
else()
    target_compile_options(raylidar_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(SHADER_SOURCE_DIR "${PROJECT_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${PROJECT_BINARY_DIR}/shaders")
raylidar_add_shaders(raylidar_shaders "${SHADER_SOURCE_DIR}" "${SHADER_BINARY_DIR}")
add_dependencies(raylidar_smoke raylidar_shaders)

enable_testing()
add_test(NAME vulkan_smoke COMMAND raylidar_smoke "${SHADER_BINARY_DIR}/Smoke.spv")
set_tests_properties(vulkan_smoke PROPERTIES TIMEOUT 30 LABELS gpu)
