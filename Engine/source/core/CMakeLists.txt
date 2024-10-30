# CPP library 
set(PROJECT_NAME core)
add_compile_options(-msse2 -msse3)
add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PUBLIC include
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/aftermath/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vma/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/spirv_reflect/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/slang/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/tinygltf/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/tinyobjloader/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/include"
  "${PROJECT_SOURCE_DIR}/Engine/extern/libdeflate/includes"
  "${PROJECT_SOURCE_DIR}/Engine/extern/nanovdb")
target_link_directories(${PROJECT_NAME} PUBLIC
  "${PROJECT_SOURCE_DIR}/Engine/extern/glfw/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/vulkan/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/aftermath/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/libdeflate/libs"
  "${PROJECT_SOURCE_DIR}/Engine/extern/slang/libs")
target_link_libraries(${PROJECT_NAME} PUBLIC 
  imgui yaml cnpy tinyexr tinyxml tinymitsuba)
target_link_libraries(${PROJECT_NAME} PRIVATE 
  ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} 
  "glfw3.lib" "vulkan-1.lib" "slang.lib" "slang-rt.lib" "GFSDK_Aftermath_Lib.x64.lib" "deflate.lib")
target_sources(${PROJECT_NAME} PRIVATE 
  "source/base/se.log.cpp"
  "source/base/se.math.cpp"
  "source/base/se.memory.cpp"
  "source/base/se.platform.win.cpp"
  "source/base/se.window.gltf.cpp"
  "source/rhi/se.vma.cpp"
  "source/rhi/se.rhi.impl.cpp"
  "source/rhi/se.rhi.vk.cpp"
  "source/rhi/se.rhi.cu.cpp"
  "source/rhi/se.rhi.torch.cpp"
  "source/gfx/ex.spirv_reflect.cpp"
  "source/gfx/ex.tinyprbrtloader.cpp"
  "source/gfx/se.gfx.cpp"
  "source/gfx/se.gfx.scene.cpp"
  "source/gfx/se.gfx.pmf.cpp"
  "source/gfx/se.gfx.lightbvh.cpp"
  "source/base/se.resource.cpp"
  "source/rdg/se.rdg.cpp"
  "source/editor/se.editor.cpp"
  "source/base/se.utils.cpp" 
  "source/base/se.misc.cpp"
   "source/image/se.image.cpp" 
  "source/pass/se.pass.postprocess.cpp"
  "source/pass/se.pass.editor.cpp"
  "source/pass/se.pass.rasterizer.cpp" 
  "source/pass/se.pass.cbt.cpp" 
  "source/bxdf/se.bxdf.epflbrdf.cpp" 
   "source/lights/se.lights.envmap.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20)

# PY library 
set(PROJECT_NAME pycore)
add_library(${PROJECT_NAME} SHARED)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PRIVATE core ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_sources(${PROJECT_NAME} PRIVATE "pyscript/py.core.cpp")
set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 20
    SUFFIX ".pyd")
# target_compile_options(pytorch_cmake_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++-9>)