add_executable(test_common
    test_io.cpp
    test_cli_options.cpp)
target_link_libraries(test_common PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(test_common PRIVATE cxx_std_20)
catch_discover_tests(test_common PROPERTIES LABELS common)
