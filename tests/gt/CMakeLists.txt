add_executable(raylidar_test_gt
    fixture.cpp
    test_camera_assign.cpp
    test_cache.cpp)
target_link_libraries(raylidar_test_gt PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_gt PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_gt PROPERTIES LABELS gt)
