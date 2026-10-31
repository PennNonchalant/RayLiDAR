find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

foreach(domain common cli dataset calib prereg overlay g1)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${domain}/CMakeLists.txt")
        add_subdirectory(${domain})
    endif()
endforeach()
