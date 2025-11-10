
# Set CUDA standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================
# Find Python and nanobind
# =============================================================
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Find Python from the conda environment
find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Print Python info for debugging - VERIFY THIS SHOWS PYTHON 3.8
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")
message(STATUS "Python version: ${Python_VERSION}")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Find nanobind
# Method 1: If nanobind is installed in the conda environment
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE 
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE nanobind_RESULT
)

if(nanobind_RESULT EQUAL 0)
    message(STATUS "Found nanobind at: ${nanobind_ROOT}")
    find_package(nanobind CONFIG REQUIRED PATHS ${nanobind_ROOT})
else()
    # Method 2: If nanobind is a git submodule or external dependency
    message(STATUS "Nanobind not found via pip, looking for submodule...")
    add_subdirectory(extern/nanobind)  # Adjust path as needed
endif()

# Find PyTorch via Python
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE Torch_PYTHON_CMAKE_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Torch Python CMake path: ${Torch_PYTHON_CMAKE_PATH}")
find_package(Torch REQUIRED PATHS ${Torch_PYTHON_CMAKE_PATH})
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch library dirs: ${TORCH_LIBRARIES}")

set(PROJECT_NAME "pycore")
# Link CUDA runtime library (optional if you're only including headers)
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found: ${CUDAToolkit_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "CUDAToolkit not found.")
endif()

nanobind_add_module(${PROJECT_NAME})
include_directories(${PROJECT_NAME} PRIVATE
    "../extern",
)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    core
    CUDA::cudart
)
target_sources(${PROJECT_NAME} PRIVATE
    "py.core.cpp"
    "se.rhi.cuda.cpp"
)
# Set properties for the Python module
set_target_properties(pycore PROPERTIES
    OUTPUT_NAME "pycore"  # This will be the import name in Python
)

# Determine extension based on platform
if(WIN32)
set(PY_EXT "pyd")
elseif(UNIX)
set(PY_EXT "so")
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/release/sibylengine"
    COMMAND ${CMAKE_COMMAND} -E echo "Copying ${PY_EXT} files..."
    COMMAND python ${CMAKE_SOURCE_DIR}/Engine/scripts/copy_built.py
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
        "${CMAKE_SOURCE_DIR}/release/sibylengine"
        "${PY_EXT}"
    COMMENT "Running post-build script to copy .${PY_EXT} files"
)

add_subdirectory(oqmc)