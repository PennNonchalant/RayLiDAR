add_executable(raylidar_test_dataset
    fixture.cpp
    test_layout.cpp
    test_preflight.cpp)
target_link_libraries(raylidar_test_dataset PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_dataset PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_dataset PROPERTIES LABELS dataset)
