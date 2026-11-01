# Shared contracts: IO/SHA-256, artifacts, CLI dispatch, KITTI calibration, and
# the frozen gt/ray binary schemas. Both data_process and the future eval/sim
# targets depend on this library; it must not depend on either.
file(GLOB_RECURSE RAYLIDAR_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(raylidar_core SHARED ${RAYLIDAR_CORE_SOURCES})
add_library(raylidar::core ALIAS raylidar_core)

target_compile_features(raylidar_core PUBLIC cxx_std_20)
target_compile_definitions(raylidar_core PRIVATE RAYLIDAR_VERSION="${PROJECT_VERSION}")
set_target_properties(raylidar_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

include(GenerateExportHeader)
generate_export_header(raylidar_core
    EXPORT_MACRO_NAME RAYLIDAR_EXPORT
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/raylidar/export.h")

target_include_directories(raylidar_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include")

target_link_libraries(raylidar_core PUBLIC
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    OpenSSL::Crypto)
