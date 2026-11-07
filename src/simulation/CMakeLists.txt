add_library(raylidar_simulation STATIC
    LidarSensor.cpp
    LidarSimulator.cpp
    PointCloudIO.cpp
)

target_compile_features(raylidar_simulation PUBLIC cxx_std_20)

target_include_directories(raylidar_simulation PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)

target_link_libraries(raylidar_simulation PUBLIC
    raylidar_rt
)

set_target_properties(raylidar_simulation PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_precompile_headers(raylidar_simulation PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../pch.h"
)
