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
find_package(nlohmann_json CONFIG REQUIRED)

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

add_executable(raylidar_lidar_test
    apps/lidar_test/main.cpp
)

target_link_libraries(raylidar_lidar_test PRIVATE
    raylidar_simulation
)

add_executable(raylidar_kitti_eval
    apps/kitti_eval/main.cpp
)

target_link_libraries(raylidar_kitti_eval PRIVATE
    raylidar_simulation
)

foreach(target IN ITEMS raylidar_vulkan raylidar_rt raylidar_simulation raylidar raylidar_cli raylidar_smoke
                        raylidar_lidar_test raylidar_kitti_eval)
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
add_dependencies(raylidar_lidar_test raylidar_shaders)
add_dependencies(raylidar_kitti_eval raylidar_shaders)

add_test(
    NAME vulkan_context_smoke
    COMMAND raylidar_smoke 
        "${SHADER_BINARY_DIR}/Smoke.spv" 
        "${SHADER_BINARY_DIR}/RayQuerySmoke.spv"
)

set_tests_properties(vulkan_context_smoke PROPERTIES
    TIMEOUT 30
    LABELS "gpu;vulkan"
    FAIL_REGULAR_EXPRESSION "\\[Vulkan\\]\\[error\\]"
)

add_test(NAME lidar_sensor COMMAND raylidar_lidar_test sensor)

set_tests_properties(lidar_sensor PROPERTIES
    TIMEOUT 30
)

add_test(
    NAME lidar_wall
    COMMAND raylidar_lidar_test wall "${SHADER_BINARY_DIR}/LidarTrace.spv"
)

set_tests_properties(lidar_wall PROPERTIES
    TIMEOUT 30
    LABELS "gpu;vulkan"
    FAIL_REGULAR_EXPRESSION "\\[Vulkan\\]\\[error\\]"
)
