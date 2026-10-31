add_executable(raylidar_test_overlay
    fixture.cpp
    test_project.cpp)
target_link_libraries(raylidar_test_overlay PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_overlay PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_overlay PROPERTIES LABELS overlay)
