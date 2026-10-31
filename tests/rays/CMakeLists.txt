add_executable(raylidar_test_rays
    fixture.cpp
    ../gt/fixture.cpp
    test_rays.cpp
    test_occlusion.cpp
    test_rays_cli.cpp
    test_waterfall.cpp
    test_freeze.cpp)
target_link_libraries(raylidar_test_rays PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_rays PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_rays PROPERTIES LABELS rays)
