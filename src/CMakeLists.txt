# Glob every translation unit so later plans can add src/**/*.cpp without
# editing this file. CONFIGURE_DEPENDS re-globs on each build.
file(GLOB_RECURSE RAYLIDAR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(raylidar SHARED ${RAYLIDAR_SOURCES})
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

# PUBLIC because tests and later targets need the same headers/targets.
target_link_libraries(raylidar PUBLIC
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    OpenSSL::Crypto)

# STB is header-only: expose its include directory.
target_include_directories(raylidar PUBLIC "${Stb_INCLUDE_DIR}")

# libarchive: prefer the imported target, fall back to Find-module variables.
if(TARGET LibArchive::LibArchive)
    target_link_libraries(raylidar PUBLIC LibArchive::LibArchive)
else()
    target_link_libraries(raylidar PUBLIC ${LibArchive_LIBRARIES})
    target_include_directories(raylidar PUBLIC ${LibArchive_INCLUDE_DIRS})
endif()
