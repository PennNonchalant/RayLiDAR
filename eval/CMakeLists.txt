# Evaluation harness placeholder. Phase 3 adds the metrics implementation and
# the eval CLI here; it must depend on core (schemas) and not on data_process
# or the runtime library.
add_library(raylidar_eval INTERFACE)
add_library(raylidar::eval ALIAS raylidar_eval)

target_compile_features(raylidar_eval INTERFACE cxx_std_20)
target_include_directories(raylidar_eval INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(raylidar_eval INTERFACE raylidar_core)
