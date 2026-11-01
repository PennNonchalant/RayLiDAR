# Main runtime placeholder (Vulkan/VMA/BLAS/TLAS). Member 1's framework and
# member 2's GPU ray query implementation will add sources here. Until then this
# is an interface target so the umbrella `raylidar` target can still depend on
# a stable name.
file(GLOB_RECURSE RAYLIDAR_RUNTIME_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(RAYLIDAR_RUNTIME_SOURCES)
    add_library(raylidar_runtime SHARED ${RAYLIDAR_RUNTIME_SOURCES})
    target_compile_features(raylidar_runtime PUBLIC cxx_std_20)
    set_target_properties(raylidar_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    target_link_libraries(raylidar_runtime PUBLIC raylidar_core)
else()
    add_library(raylidar_runtime INTERFACE)
    target_compile_features(raylidar_runtime INTERFACE cxx_std_20)
    target_link_libraries(raylidar_runtime INTERFACE raylidar_core)
endif()

add_library(raylidar::runtime ALIAS raylidar_runtime)
