# Catch2/CTest `eval` domain: exact nearest-neighbour search, metric formulas,
# bucket/coverage accounting, aggregation, and synthetic correctness proofs.
add_executable(raylidar_test_eval
    fixture.cpp
    test_knn.cpp
    test_range.cpp
    test_miss_coverage.cpp
    test_buckets.cpp
    test_aggregate.cpp)

target_link_libraries(raylidar_test_eval PRIVATE raylidar::eval Catch2::Catch2WithMain)
target_compile_features(raylidar_test_eval PRIVATE cxx_std_20)
target_compile_definitions(raylidar_test_eval PRIVATE
    RAYLIDAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(raylidar_test_eval PROPERTIES LABELS eval)
