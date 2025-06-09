# SIByLEngine 2023.1

`SIByLEngine` is a personal toy game engine by [@SuikaSibyl](https://github.com/SuikaSibyl). I am planning to keep working on it for a long time and implement lots of realtime & offline effects in it.

Previous version could be find in: [SibylEngine2021](https://github.com/SuikaSibyl/SibylEngine2021). Version 2023 is a complete refactoring including better and more extendable structure, solution, design.

![EditorDemoImg](https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/2023_1.png)

## Builds

Only support Windows platform and MSVC compiler for now.


By default, Nvidia GPU with Turing or higher architecture is required to correctly run the examples, as vulkan hardware raytracing is defaultly used. If no corresponding GPU exists, compilation should still be successful, but you should manually turn off ray-tracing-related features to run it without fatal error.


### Compiliation Instruction

The compilation is tested on Windows 11 machine with Visual Studio 2022 (11.17.2, older version should also work).

1) Download Vulkan SDK from [LunarG](https://vulkan.lunarg.com/sdk/home#windows), and install it. This is mandatory to enable Validation layer, which is helpful for potential error detection. Test compilation uses VulkanSDK-1.3.290.0 (latest release), but moderately older version should also work.
2) Open the SIByLEngine.sln solution with VisualStudio 2022.
Select "Sandbox" project as target and hit compilation. It will take some time, but ideally you should eventually be able to compile everything without getting error. You should be able to compile with any GPU, but to run the default example you will need a NV GPU supporting hardware ray tracing.
3) Find the scene binaries **Sandbox.zip** in the root folder, uncompress it in root folder so that its "Sandbox" folder overlaps with the "Sandbox" folder already exists.
4) You should be able to run the example hopefully.


## Design Decisions
- `NOT Use C++ 20 Module` for the all the modules. Although I had previously used it extensively and found it to be a great feature. Despite MSVC's support for the feature, compilation is not always stable, sometimes resulting in weird 'internal error' messages that force the process to be abandoned. Moreover, as the project grows, both IntelliSense and ReSharper struggle to provide proper code highlighting and intelligent code completions, which makes coding extremely painful. Also, analysising the project eat up all the memory quickly and get the IDE and compiler super slow... Given these challenges, I have opted to deprecate the feature.

- `Only Support Vulkan Backend` for RHI module. It seems that there are some API logistic differences between Vulkan / OpenGL / DirectX. But the main reason is probably simply because I am too lazy to implement and test for all the backends for now. As Vulkan is supporting most of the features exists, I find no good reason to support other backends at this stage.

- `Use Render Graph` to manage GPU pipeline. The main motivation is that Vulkan is too verbose about memory barriers. Render graph helps me to support automatic barrier insertion (within the graph) and resource management. It may also be a good idea to do automatic barrier insertion at a lower layer like RHI, which may be helpful to support a more flexible pipeline, but I am not sure with that for now.

## Modules
- ### Core Modules
  - [SE.Core.Log](./docs/SIByLDocument_003_ECS.md): Log print system from scratch.
  - [SE.Core.Memory](./docs/SIByLDocument_003_ECS.md): Memory management system from scratch (experimental).
  - [SE.Core.ECS](./docs/SIByLDocument_003_ECS.md): ECS system from scratch (experimental).
  - [SE.Core.Resource](./docs/SIByLDocument_004_Resource.md): Resource management system from scratch (experimental).
  - [SE.Math](./docs/SIByLDocument_002_Math.md): Basic math computation library from scratch with SIMD acceleration.
- ### Rendering Modules
  - [SE.RHI Module](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#rhi-module): Abstract layer with `WebGPU`-style API supporting `Vulkan` backend.
  - [SE.Tracer](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#tracer-module): `CPU ray tracing` implementation based on PBRT v3 (`GPU` implementation see `SE.SRenderer`)
  - [SE.Image](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#image-module): `Image` loader/storer for various format.
  - [SE.Video](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#video-module): `Video` loader/storer for various format, decoding via `ffmpeg`.
  - [SE.GFX](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Graphics-Modules#gfx-module): Higher-Level infrastructure on top pf `SE.RHI`
    - Defining `scene`, with many basic `component`s and  `resource`s.
    - Providing `serializing` & `deserializing` via `yaml-cpp`.
  - [SE.RDG](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/Render-Dependency-Graph): `Render Dependency Graph` System (experimental).
    - `Pass`: rhi-pipeline-level (rasterize/fullscreen/compute/raytrace) atomic node in graph, with automatic pipeline and resource binding layouts creation via `spirv-cross` reflection of SPIR-V shaders.
    - `Graph`: a directed acyclic graph of `Pass`es, automatic resource management and barrier insertion.
    - `Pipeline`: a interface enabling cross-frame switching between multiple  `Graph`s.
  - [SE.SRenderer](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/SIByL-Renderer): Retained mode renderer based on SE.RDG
    - `Retained mode rendering` infrastructure for both rasterization and ray tracing
    - Providing lots of `RDG Pass/Graph/Pipeline` [algorithm implementation](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/SIByL-Renderer#algorithm-implemented).

- ### Physics
  - // TODO.
- ### Editor
  - `Editor_Core module`: ImGui-Vulkan interoperate and extensible GUI framework.
  - `Editor_GFX module`: Provides many editor widgets for SIByL Engine. 
  - `Editor_RDG module`: Provides DebugDraw system via RDG and RDG Viewer (no RDG editor yet).

## Demo Gallery

<p align="center">
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/bdpt-conv.gif" height="260">
  &nbsp
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/mmlt-gif.gif" height="260">
  <br />
  <em>Interactive GPU implementation of BDPT (left)  </em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/main/Engine/Shaders/SRenderer/raytracer/bdpt">[code]</a>  <em>and MMLT (right)  </em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/main/Engine/Shaders/SRenderer/raytracer/mmlt">[code]</a>
</p>

<p align="center">
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/2022-11-09%2017-21-09%2000_00_00-00_00_06.gif" height="260">
  &nbsp
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/2022-11-09%2017-16-29%2000_00_00-00_00_05.gif" height="260">
  <br />
  <em>Axis-Aligned Filtering for soft shadow (left)  </em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/main/Engine/Shaders/RayTracing/RayTrace/src/AAF_softshadow">[code]</a>  <em>and and global illumination  </em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/main/Engine/Shaders/RayTracing/RayTrace/src/aaf_gi">[code]</a>
  <br/>
  <em>(aliasing is caused by gif compression)  </em>
</p>

<p align="center">
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/2023-06-13%2023-21-34%2000_00_03-00_00_15%2000_00_00-00_00_30.gif" width="420">
  <br />
  <em>2D FLIP Fluid Simulation</em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/main/Demo/FLIP">[code]</a>
</p>

<p align="center">
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/captalp.gif" height="260">
  &nbsp
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/rabit.gif" height="260">
  <br />
  <em>3D tetrahedral elastic body simlation</em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/main/Demo/Elastic">[code]</a>
</p>

<p align="center">
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/coordinate-network.gif" height="260">
  <br />
  <em>Vulkan tiny-MLP writen by </em>
  <a href="https://developer.nvidia.com/blog/differentiable-slang-a-shading-language-for-renderers-that-learn/">Differentiable Slang</a>
  <em> with glsl WMMA</em>
</p>

<p align="center">
  <img src="https://imagehost-suikasibyl-us.oss-us-west-1.aliyuncs.com/img/differentiable_triangle.gif" height="260">
  <br />
  <em>Calling Vulkan functionality in PyTorch in </em>
  <a href="https://github.com/SuikaSibyl/SIByLEngine2023/tree/dev-24">[SE2024]</a>
  <em>! A simple differentiable rendering demo is shown.</em>
</p>

## Selected Feature List

- ### Primitive Intersection and Sampling
  - ```Triangle Intersection Offset```: [A Fast and Robust Method for Avoiding Self-Intersection](https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.4.pdf)
  - ```Sphere Intersection Query```: [Precision Improvements for Ray/Sphere Intersection](https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.4.pdf)
  - ```Spherical Rectangle Sampling```: [An Area-Preserving Parametrization for Spherical Rectangles](https://dl.acm.org/doi/10.1111/cgf.12151)

- ### Material Models and BSDF Sampling
  - ```Lambertian BSDF```: [Photometria, sive De mensura et gradibus luminus, colorum et umbrae](https://archive.org/details/lambertsphotome00lambgoog)
  - ```RoughPlastic & Dielectric BSDF```: [Average irregularity representation of a rough surface for ray reflection](https://opg.optica.org/josa/abstract.cfm?uri=josa-65-5-531)
  - ```Disney Principled BSDF```: [Physically Based Shading at Disney](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)
  - ```Importance Sampling of GGX Microfacet Model```: [Sampling the GGX Distribution of Visible Normals](https://jcgt.org/published/0007/04/01/) 

- ### Various Formulations of Light Transport
  - ```Unidirectional Path Tracing with MIS```: [Robust - Veach's Thesis [Chapter 9 - Multiple Importance Sampling]](https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf)
  - ```Bidirectional Path Tracing```:  [Robust - Veach's Thesis [Chapter 10 - Bidirectional Path Tracing]](https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter10.pdf)
  - ```Multiplexed Metropolis Light Transport```:  [Multiplexed Metropolis Light Transport](https://cs.uwaterloo.ca/~thachisu/mmlt.pdf)
  - ```Light Tracing as VPL```:   [[wiki page]](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/SRenderer-Addons#vpl--virtual-point-light)  |  [Instant Radiosity](https://doi.org/10.1145/258734.258769)

- ### Importance Sampling for Efficient Light Transport
    - ```Screen-Space Path Guiding```:  [Real-Time Path-Guiding Based on Parametric Mixture Models](https://diglib.eg.org/bitstream/handle/10.2312/egs20221024/025-028.pdf)
    - ```Neural Importance Sampling (modified)```:  [[project page]](https://suikasibyl.github.io/CSE272-Report-WebPage/) |  [Neural Importance Sampling](https://dl.acm.org/doi/10.1145/3341156)
    - ```ReSTIR GI (Global Illumination)```:  [ReSTIR GI: Path Resampling for Real-Time Path Tracing](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)

- ### Efficient Many Lights Sampling
  - ```Real-time Stochastic Lightcuts```:  [wiki page](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/SRenderer-Addons#slc--stochastic-lightcuts) | [Real-Time Stochastic Lightcuts](https://dqlin.xyz/pubs/2020-i3d-SLC/)
  - ```ReSTIR DI (Direct Lighting)```:  ------- *Under development* -------

- ### Denoiser For Noisy Path Traced Image
  - Realtime Local Frequency Analysis Based Approaches [[project page](https://suikasibyl.github.io/CSE274-RealtimeDenoiser-WebPage/)] 
    - ```AAF Softshadow```: [Axis-Aligned Filtering for Interactive Sampled Soft Shadows](http://graphics.berkeley.edu/papers/UdayMehta-AAF-2012-12/)
    - ```AAF GI```: [Axis-Aligned Filtering for Interactive Physically-Based Diffuse Indirect Lighting](https://cseweb.ucsd.edu/~ravir/filtering_GI_final.pdf)
    - ```MAAF Combined Effects```: [Multiple Axis-Aligned Filters for Rendering of Combined
Distribution Effects](https://cseweb.ucsd.edu/~ravir/paper_maaf.pdf)
  - Spatial-Temporal Reuse Based Approaches With Moment Estimation [[wiki page](https://github.com/SuikaSibyl/SIByLEngine2023/wiki/SRenderer-Addons#a-svgf-adaptive-spatiotemporal-variance-guided-filtering)]
    - ```SVGF Denoiser```: [SVGF: Real-Time Reconstruction for Path-Traced Global Illumination](https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A)
    - ```A-SVGF Denoiser```: [Gradient Estimation for Real-Time Adaptive Temporal Filtering](https://cg.ivd.kit.edu/atf.php)

- ### Biased Global Illumination with Approximation
   - ```Voxel Global Illumination```: [Deferred voxel shading for real-time global illumination](https://ieeexplore.ieee.org/document/7833375)
     - ```Realtime Voxelization Based on Rasterization```: [The Basics of GPU Voxelization](https://developer.nvidia.com/content/basics-gpu-voxelization)
   - ```Stochastic Substitute Tree```:  [Stochastic Substitute Trees for Real-Time Global Illumination](https://dl.acm.org/doi/fullHtml/10.1145/3384382.3384521)
   - ```Screen Space Reflection / Global Illumination```: [Stochastic Screen Space Reflections](https://www.ea.com/frostbite/news/stochastic-screen-space-reflections)
   - ```DDGI```:  ------- *Under development* -------

- ### Rasterizer-Specific Rendering Algorithms
   - Shadow Mapping [Casting curved shadows on curved surfaces](https://cseweb.ucsd.edu//~ravir/274/15/papers/p270-williams.pdf)
     - ```Cascaded Shadow Maps```: [Cascaded Shadow Maps](https://developer.download.nvidia.com/SDK/10.5/opengl/src/cascaded_shadow_maps/doc/cascaded_shadow_maps.pdf)
     - ```Percentage-Closer Filtering```: [GPU Gems [Chapter 11. Shadow Map Antialiasing]](https://developer.nvidia.com/gpugems/GPUGems/gpugems_ch11.html)
       
- ### Miscellaneous Utils as GPU Parallel Computing
  - Parallel Sorting Algorithms on GPU
    - ```Parallel Bitonic Sort```: [GPU Gem 2 [Chapter 46. Improved GPU Sorting]](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)
    - ```A Faster Parallel Radix Sort```: [A Faster Radix Sort Implementation](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf)
  - Screen Space Passes and Computer Vision Processing
    - ```G-Buffer```: [Comprehensible rendering of 3-D shapes](https://dl.acm.org/doi/pdf/10.1145/97880.97901)
    - ```V-Buffer```: [The Visibility Buffer: A Cache-Friendly Approach to Deferred Shading](https://jcgt.org/published/0002/02/04/paper.pdf)
    - ```gSLICr Superpixel Segmentation```: [gSLICr: SLIC superpixels at over 250Hz](https://www.robots.ox.ac.uk/~victor/gslicr/)

## Dependencies
- `dear_imgui`: [Bloat-free Graphical User interface for C++ with minimal dependencies](https://github.com/ocornut/imgui)
- `ffmpeg`: [A complete, cross-platform solution to record, convert and stream audio and video.](https://ffmpeg.org/)
- `glad`: [Multi-Language Vulkan/GL/GLES/EGL/GLX/WGL Loader-Generator based on the official specs.](https://github.com/Dav1dde/glad)
- `glfw`: [A multi-platform library for OpenGL, OpenGL ES, Vulkan, window and input.](https://github.com/glfw/glfw)
- `slang`: [Making it easier to work with shaders.](https://github.com/shader-slang/slang)
- `spirv-cross`: [A practical tool and library for performing reflection on SPIR-V and disassembling SPIR-V back to high level languages.](https://github.com/KhronosGroup/SPIRV-Cross)
- `stb`: [stb single-file public domain libraries for C/C++](https://github.com/nothings/stb)
- `tinygltf`: [Header only C++11 tiny glTF 2.0 library](https://github.com/syoyo/tinygltf)
- `tinyobjloader`: [Tiny but powerful single file wavefront obj loader](https://github.com/tinyobjloader/tinyobjloader)
- `vma`: [Easy to integrate Vulkan memory allocation library](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- `vulkan`: [A low-overhead, cross-platform API, open standard for 3D graphics and computing](https://www.vulkan.org/)
- `yaml-cpp`: [A YAML parser and emitter in C++](https://github.com/jbeder/yaml-cpp)

## References
I learned a lot from various renderers to build up this engine, here are some great references for those who might be interested.
- [PBRT](https://github.com/mmp/pbrt-v3): Rendering bible.
- [Lajolla Renderer](https://github.com/BachiLi/lajolla_public): Well-written offline renderer, learned so much from it.
- [Hazel Engine](https://github.com/TheCherno/Hazel): Fancy usage of ImGui editor and good tutorial on engine development.
- [Falcor](https://github.com/NVIDIAGameWorks/Falcor): Easy to use, I especially imitate its RDG interface.
- [Unreal Engine](https://github.com/EpicGames/UnrealEngine): Needless to say.
- [Unity](https://www.unrealengine.com/): The most commonly used engine for me, I am deeply influenced in many ways.
