add_subdirectory("glfw")
add_subdirectory("imgui")
add_subdirectory("imnodes")
add_subdirectory("yaml")
add_subdirectory("tinyexr")
add_subdirectory("spvreflect")
add_subdirectory("tinymitsuba")

# openqmc
# Optionally set options (they are OFF by default)
set(OPENQMC_ARCH_TYPE AVX CACHE STRING "" FORCE)
set(OPENQMC_ENABLE_BINARY ON CACHE BOOL "" FORCE)
# Load external dependencies
add_subdirectory("openqmc" EXCLUDE_FROM_ALL)
# Optionally mark options as advanced
mark_as_advanced(FORCE OPENQMC_ARCH_TYPE)
mark_as_advanced(FORCE OPENQMC_BUILD_TOOLS)
mark_as_advanced(FORCE OPENQMC_FORCE_DOWNLOAD)
mark_as_advanced(FORCE OPENQMC_ENABLE_BINARY)
mark_as_advanced(FORCE OPENQMC_FORCE_PIC)

# libdeflate
set(LIBDEFLATE_BUILD_STATIC_LIB ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("libdeflate")