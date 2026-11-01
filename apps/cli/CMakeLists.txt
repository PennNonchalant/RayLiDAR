add_executable(raylidar_cli main.cpp)
target_link_libraries(raylidar_cli PRIVATE raylidar::raylidar raylidar::eval)

# Command registrars live in shared libraries and are not referenced by the
# executable; --no-as-needed keeps those libraries loaded so their static
# registrars run. Windows import libraries are loaded because they are listed.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(raylidar_cli PRIVATE "LINKER:--no-as-needed")
endif()

add_test(NAME cli_version COMMAND raylidar_cli --version)
set_tests_properties(cli_version PROPERTIES
    PASS_REGULAR_EXPRESSION "RayLiDAR ${PROJECT_VERSION}"
    TIMEOUT 10)

add_test(NAME cli_help COMMAND raylidar_cli --help)
set_tests_properties(cli_help PROPERTIES TIMEOUT 10)

add_test(NAME cli_unknown COMMAND raylidar_cli does-not-exist)
set_tests_properties(cli_unknown PROPERTIES TIMEOUT 10 WILL_FAIL TRUE)
