# Catch2/CTest `io` domain: the frozen simulation I/O contract.
#
# The fixtures are hand-built byte buffers (see fixture.cpp) so the tests prove
# the documented offsets instead of only writer/reader symmetry.
add_executable(raylidar_test_io
    fixture.cpp
    test_gt_rays.cpp
    test_hits.cpp)

target_link_libraries(raylidar_test_io PRIVATE raylidar::core Catch2::Catch2WithMain)
target_compile_features(raylidar_test_io PRIVATE cxx_std_20)
target_compile_definitions(raylidar_test_io PRIVATE
    RAYLIDAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(raylidar_test_io PROPERTIES LABELS io)
