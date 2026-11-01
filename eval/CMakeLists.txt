# Evaluation harness: exact nearest-neighbour search, range/Chamfer/P-R-F
# metrics, bucket/coverage accounting, and micro/macro aggregation. It depends
# on core for the frozen ray schema and must not depend on data_process or the
# runtime library.
file(GLOB_RECURSE RAYLIDAR_EVAL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp")

add_library(raylidar_eval SHARED ${RAYLIDAR_EVAL_SOURCES})
add_library(raylidar::eval ALIAS raylidar_eval)

target_compile_features(raylidar_eval PUBLIC cxx_std_20)
set_target_properties(raylidar_eval PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# Same export-header convention as data_process: define the core build macro so
# RAYLIDAR_EXPORT expands to dllexport while building this shared library on
# Windows (a no-op for ELF visibility on Linux).
target_compile_definitions(raylidar_eval PRIVATE raylidar_core_EXPORTS)

target_include_directories(raylidar_eval PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(raylidar_eval PUBLIC
    raylidar_core
    nanoflann::nanoflann)
