add_executable(raylidar_test_calib
    fixture.cpp
    test_frames.cpp
    test_kitti360.cpp
    test_real_calibration.cpp)
target_link_libraries(raylidar_test_calib PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_calib PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_calib PROPERTIES LABELS calib)
