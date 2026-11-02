add_executable(raylidar_test_lidar
    fixture.cpp
    test_velodyne.cpp
    test_uncurl.cpp
    test_uncurl_diag.cpp)
target_link_libraries(raylidar_test_lidar PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_lidar PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_lidar PROPERTIES LABELS lidar)
