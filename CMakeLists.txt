cmake_minimum_required(VERSION 3.29)
project(RayLiDAR VERSION 0.1.0 LANGUAGES CXX)

if(WIN32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR
        "RayLiDAR requires a 64-bit Windows build. Initialize the VS Developer "
        "PowerShell with -Arch amd64 -HostArch amd64, then run "
        "cmake --fresh --preset windows-debug.")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
# The project does not use C++ modules; disabling the scan keeps GCC-specific
# flags out of compile_commands.json so clang-tidy can consume it.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

# Test infrastructure. include(CTest) defines BUILD_TESTING; tests are CPU-only.
include(CTest)

# Phase 1 dependencies (pinned by vcpkg.json + builtin-baseline).
find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Stb REQUIRED)
# The vcpkg libarchive port ships only a CMake Find module (no CONFIG package),
# so try CONFIG first and fall back to the Find module. Both paths provide the
# imported target LibArchive::LibArchive on CMake >= 3.17.
find_package(LibArchive CONFIG QUIET)
if(NOT LibArchive_FOUND)
    find_package(LibArchive REQUIRED)
endif()

enable_testing()
add_subdirectory(core)
add_subdirectory(data_process)
add_subdirectory(eval)
add_subdirectory(src)

# Umbrella target kept for apps/tests that need the data pipeline. The runtime
# and eval targets are deliberately not folded into this; new code should link
# raylidar::core / raylidar::data_process / raylidar::eval / raylidar::runtime
# explicitly.
add_library(raylidar INTERFACE)
add_library(raylidar::raylidar ALIAS raylidar)
target_link_libraries(raylidar INTERFACE
    raylidar_core
    raylidar_data_process
    raylidar_runtime)

add_subdirectory(apps/cli)

find_package(VulkanHeaders 1.4 CONFIG REQUIRED)
include(cmake/SlangShaders.cmake)

add_executable(raylidar_smoke apps/smoke/main.cpp)
target_compile_features(raylidar_smoke PRIVATE cxx_std_20)
target_link_libraries(raylidar_smoke PRIVATE Vulkan::Headers ${CMAKE_DL_LIBS})

foreach(target IN ITEMS raylidar_core raylidar_data_process raylidar_eval raylidar_cli raylidar_smoke)
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

add_test(NAME vulkan_smoke COMMAND raylidar_smoke "${SHADER_BINARY_DIR}/Smoke.spv")
set_tests_properties(vulkan_smoke PROPERTIES TIMEOUT 30 LABELS gpu)

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif()
