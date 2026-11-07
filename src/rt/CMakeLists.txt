add_library(raylidar_rt STATIC
    AccelerationStructure.cpp
    ASBuilder.cpp
    MeshScene.cpp
    RayQueryTracer.cpp
)

target_compile_features(raylidar_rt PUBLIC cxx_std_20)

target_include_directories(raylidar_rt PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)

target_link_libraries(raylidar_rt PUBLIC
    raylidar_vulkan
)

set_target_properties(raylidar_rt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_precompile_headers(raylidar_rt PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../pch.h"
)