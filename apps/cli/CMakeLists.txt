add_executable(raylidar_cli main.cpp)
target_link_libraries(raylidar_cli PRIVATE raylidar::raylidar)

add_test(NAME cli_version COMMAND raylidar_cli --version)
set_tests_properties(cli_version PROPERTIES
    PASS_REGULAR_EXPRESSION "RayLiDAR ${PROJECT_VERSION}"
    TIMEOUT 10)
