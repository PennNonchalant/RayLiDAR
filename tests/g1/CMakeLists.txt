add_executable(raylidar_test_g1
    test_g1.cpp
    test_g1_cli.cpp)
target_link_libraries(raylidar_test_g1 PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_g1 PRIVATE cxx_std_20)
target_compile_definitions(raylidar_test_g1 PRIVATE
    RAYLIDAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(raylidar_test_g1 PROPERTIES LABELS g1)
