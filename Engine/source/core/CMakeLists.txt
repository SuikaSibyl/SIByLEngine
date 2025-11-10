set(PROJECT_NAME "core")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
    "include"
    "../../extern",
    "../../extern/zlib",
    "../../extern/magicenum",
   ${Vulkan_INCLUDE_DIRS},
)

target_link_libraries(${PROJECT_NAME} PUBLIC 
    Vulkan::Vulkan
    glfw
    imgui
    yaml
    tinyexr
    imnodes
    spirv-reflect
    tinymitsuba
    libdeflate::libdeflate_static
    OpenQMC::OpenQMC
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/include-win"
    )
    target_link_libraries(
        ${PROJECT_NAME} PRIVATE
        "${CMAKE_SOURCE_DIR}/Engine/extern/zlib/zlibstatic.lib"
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/lib-win/slang.lib"
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/lib-win/slang-rt.lib"
        "${CMAKE_SOURCE_DIR}/Engine/extern/aftermath/lib-win/x64/GFSDK_Aftermath_Lib.x64.lib"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${CMAKE_SOURCE_DIR}/Engine/extern/aftermath/lib-win/x64/GFSDK_Aftermath_Lib.x64.dll" 
        "${CMAKE_SOURCE_DIR}/release/sibylengine"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/bin-win/slang.dll" 
        "${CMAKE_SOURCE_DIR}/release/sibylengine"
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/include-linux"
    )
    target_link_libraries(
        ${PROJECT_NAME} PRIVATE 
        "${CMAKE_SOURCE_DIR}/Engine/extern/zlib/libz.a"
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/lib-linux/libslang.so"
        "${CMAKE_SOURCE_DIR}/Engine/extern/slang/lib-linux/libslang-rt.so"
        "${CMAKE_SOURCE_DIR}/Engine/extern/aftermath/lib-linux/x64/libGFSDK_Aftermath_Lib.x64.so"
    )
endif()

target_sources(${PROJECT_NAME} PRIVATE
    "source/se.utils.cpp"
    "source/se.math.cpp"
    "source/se.rhi.cpp"
    "source/se.editor.cpp"
    "source/se.math.cpp" 
    "source/se.gfx.resources.cpp"
    "source/se.gfx.resources.cpp" 
    "source/se.gfx.image.cpp" 
    "source/se.gfx.scene.cpp" 
    "source/se.gfx.components.cpp"
    "source/se.gfx.materials.cpp"
    "source/se.rdg.cpp" 
    "source/se.editor.cpp" 
    "source/se.editor.frags.cpp" 
    "source/se.gfx.scene-gpu.cpp" 
    "source/se.gfx.scene-gltf.cpp"
    "source/se.gfx.scene-ext.cpp"
    "source/se.init.cpp" 
    "source/se.gfx.scene-xml.cpp" 
    "source/se.gfx.scene-lightbvh.cpp" 
    "addon/bxdf-microfacet/se.bxdf.microfacet.cpp"
    "addon/bxdf-rgl/se.bxdf.rglbrdf.cpp" 
    "addon/pass-editor/ex.pass.editor.cpp" 
    "source/se.gfx.scene-pbrt.cpp" 
    "source/ex.tinyprbrtloader.cpp"
    "source/ex.openqmc.cpp")