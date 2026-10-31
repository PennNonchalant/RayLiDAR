add_executable(raylidar_test_masking
    fixture.cpp
    test_bbox.cpp
    test_semantics.cpp
    test_mask.cpp)
target_link_libraries(raylidar_test_masking PRIVATE raylidar::raylidar Catch2::Catch2WithMain)
target_compile_features(raylidar_test_masking PRIVATE cxx_std_20)
catch_discover_tests(raylidar_test_masking PROPERTIES LABELS masking)
