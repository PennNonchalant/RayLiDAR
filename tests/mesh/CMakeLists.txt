# Catch2/CTest `mesh` domain: the Phase-4 manifest ingest gate.
#
# The PLY/GLB fixture builders are shared with tests/io so the manifest tests
# prove the real byte contract instead of a writer/reader pair under test.
add_executable(raylidar_test_mesh
    ../io/fixture.cpp
    test_mesh_manifest.cpp)

target_link_libraries(raylidar_test_mesh PRIVATE
    raylidar::eval
    raylidar::data_process
    Catch2::Catch2WithMain)
target_compile_features(raylidar_test_mesh PRIVATE cxx_std_20)
target_include_directories(raylidar_test_mesh PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../io")
target_compile_definitions(raylidar_test_mesh PRIVATE
    RAYLIDAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(raylidar_test_mesh PROPERTIES LABELS mesh)
