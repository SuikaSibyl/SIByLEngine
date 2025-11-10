
set(PROJECT_NAME "pyoqmc")
nanobind_add_module(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE core)
target_sources(${PROJECT_NAME} PRIVATE
    "py.oqmc.cpp"
)
# Set properties for the Python module
set_target_properties(pyoqmc PROPERTIES
    OUTPUT_NAME "pyoqmc"  # This will be the import name in Python
)

# Determine extension based on platform
if(WIN32)
set(PY_EXT "pyd")
elseif(UNIX)
set(PY_EXT "so")
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying ${PY_EXT} files for OQMC..."
    COMMAND python ${CMAKE_SOURCE_DIR}/Engine/scripts/copy_built.py
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
        "${CMAKE_SOURCE_DIR}/release/sibylengine"
        "${PY_EXT}"
    COMMENT "Running post-build script to copy .${PY_EXT} files"
)
