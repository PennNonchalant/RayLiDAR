find_program(SLANGC_EXECUTABLE NAMES slangc
    PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/shader-slang"
    NO_DEFAULT_PATH REQUIRED)

function(raylidar_add_shaders target source_dir output_dir)
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS
        RELATIVE "${source_dir}" "${source_dir}/*.slang")
    # Common modules are imported by entry-point shaders, not compiled separately.
    list(FILTER sources EXCLUDE REGEX "^common/")

    set(outputs)
    foreach(source IN LISTS sources)
        cmake_path(REPLACE_EXTENSION source LAST_ONLY ".spv" OUTPUT_VARIABLE relative_output)
        set(output "${output_dir}/${relative_output}")
        get_filename_component(output_parent "${output}" DIRECTORY)
        add_custom_command(
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_parent}"
            COMMAND "${SLANGC_EXECUTABLE}" "${source_dir}/${source}"
                -I "${source_dir}" -entry main -target spirv -profile spirv_1_6
                -matrix-layout-column-major -o "${output}" -depfile "${output}.d"
            DEPENDS "${source_dir}/${source}" "${SLANGC_EXECUTABLE}"
            DEPFILE "${output}.d"
            COMMENT "Compiling Slang shader ${source}"
            VERBATIM)
        list(APPEND outputs "${output}")
    endforeach()

    add_custom_target(${target} DEPENDS ${outputs})
endfunction()
