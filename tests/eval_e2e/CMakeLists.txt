# Catch2/CTest `eval` domain: end-to-end proof of the frozen metrics/output
# contract, the deterministic placeholder scene, and the self-registered
# `raylidar_cli eval` command. Uses the same `eval` label as the unit tests.
add_executable(raylidar_test_eval_e2e
    test_output.cpp
    test_protocol_doc.cpp)

target_link_libraries(raylidar_test_eval_e2e PRIVATE
    raylidar::eval
    Catch2::Catch2WithMain)

target_compile_features(raylidar_test_eval_e2e PRIVATE cxx_std_20)
target_compile_definitions(raylidar_test_eval_e2e PRIVATE
    RAYLIDAR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

catch_discover_tests(raylidar_test_eval_e2e PROPERTIES LABELS eval)
