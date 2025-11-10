set(PROJECT_NAME "Console")
add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE 
    core
)
include_directories(${PROJECT_NAME} PRIVATE
    "../../Engine/extern"
)
target_sources(${PROJECT_NAME} PRIVATE
    "src/main.cpp"
)

if (MSVC)
    message("COPY DLL TO:", "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${CMAKE_SOURCE_DIR}/Engine/extern/aftermath/lib-win/x64/GFSDK_Aftermath_Lib.x64.dll" 
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/"
    )
endif()