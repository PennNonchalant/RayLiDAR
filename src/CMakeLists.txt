add_library(raylidar SHARED version.cpp)
add_library(raylidar::raylidar ALIAS raylidar)

target_compile_features(raylidar PUBLIC cxx_std_20)
target_compile_definitions(raylidar PRIVATE RAYLIDAR_VERSION="${PROJECT_VERSION}")
set_target_properties(raylidar PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

include(GenerateExportHeader)
generate_export_header(raylidar
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/raylidar/export.h")

target_include_directories(raylidar PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include")
