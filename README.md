# SIByLEngine 0.0.3

`SIByLEngine` is a toy renderer framework based on `Vulkan`,
designed for rapid prototyping of real-time and offline graphics algorithms requiring GPU acceleration.
It supports interoperation with `Python` and `PyTorch`, enabling fast development, differentiable computing, and integration with deep learning components.

We also provide a series of [examples](#examples) to:
1) demonstrate how to use our Python interface and develop new application
2) showcase the implementation of various graphics algorithms on the GPU.

## Requirements
Our framework currently supports only `Windows` with an `NVIDIA` GPU.<br>
To run many of our examples, the GPU must also support hardware ray tracing.

## Install precompiled binaries
Follow the steps below to install our framework. Please use the specified `Python` and `Torch` versions. <br>
If you require compatibility with other Python and Torch versions, please build from the source code.
```powershell
# create a new conda environment
conda create -n sibyl python=3.8.18
conda activate sibyl

# install pytorch version 2.1.1 with CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# install sibylengine
pip install sibylengine

# install additional packages for examples
pip install matplotlib
```

Meanwhile, we highly recommend installing the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows),
as it helps with debugging.

Our examples assume that the Vulkan SDK is properly installed.
If you prefer to work without Vulkan SDK, you need to comment out the line containing `core.rhi.EnumContextExtension.DEBUG_UTILS` in our `common.py` file.
Additionally, you may need to install [shader-slang](https://github.com/shader-slang/slang) separately.

## Examples
To run the examples, please download this repository.

For example, if you place it in `d:/GitProjects/sibylengine/`, such that:
```
sibylengine/  
  ├── examples/  
  ├── resources/  
  ├── shaders/  
  ├── ...  
```

You can open the `sibylengine/` folder using VSCode.<br>
Set the `sibyl` Conda environment as the active Python interpreter, <br>
and then you should be able to run any example scripts.

Here is a list of provided examples:
| Name | Figure | Description |
| ----------- | ----------- | ----------- |
| Walk on Sphere | todo | GPU implementation of [WoS PDE solver](https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/index.html) adapted from on [official C++ implementation](https://cs.dartmouth.edu/~wjarosz/publications/sawhneyseyb22gridfree.html) |
| Reverse Walk on Sphere | todo | GPU implementation of [reverse WoS PDE solver](https://cs.dartmouth.edu/~wjarosz/publications/qi22bidirectional.html) adapted from on [official C++ implementation](https://github.com/Argent1024/BidirectionalWoS)  |

## Debugging

**Using NVIDIA Nsight Graphics**: 
When the app is a real-time project with editor enabled, 
it is straight forward to use [NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics) for debugging and profiling.
For example, to debug the `wos-forward` project, use `Frame Debugger` with following setting:
| item | fill with (example) | comments |
| ----------- | ----------- | ----------- |
| *Application Executable* | `d:/Anaconda/envs/sibyl/python.exe` | path to your conda environment |
| *Working Directory* | `d:/GitProjects/sibylengine/` | path to the root of this project |
| *Command Line Argument* | `d:/GitProjects/sibylengine/examples/wos/wos_forward.py` | path to the executable Python |
