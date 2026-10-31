add_executable(raylidar_test_segment
    fixture.cpp
    test_select.cpp
    test_select_cli.cpp)
target_link_libraries(raylidar_test_segment PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_segment PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_segment PROPERTIES LABELS segment)
