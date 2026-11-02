add_executable(raylidar_test_prereg
    test_prereg.cpp
    test_prereg_cli.cpp)
target_link_libraries(raylidar_test_prereg PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_prereg PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_prereg PROPERTIES LABELS prereg)
