# SIByLEngine 0.0.5

`SIByLEngine` is a toy renderer framework based on `Vulkan`,
designed for rapid prototyping of real-time and offline graphics algorithms requiring GPU acceleration.
It supports interoperation with `Python` and `PyTorch`, enabling potential of rapid development, differentiable computing, and integration with deep learning components.

## Requirements

Our framework currently supports only `Windows` and `Linux` with an `NVIDIA` GPU.<br>
To run some of our examples, the GPU must also support hardware ray tracing.

## Install precompiled binaries

For Windows users, follow the steps below to install our framework. Please use the specified `Python` and `Torch` versions. <br>
If you require compatibility with other Python and Torch versions, please build from the source code.

```powershell
# create a new conda environment
conda create -n sibyl python=3.8.18
conda activate sibyl

# install pytorch version 2.4.0 with CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# install sibylengine
pip install sibylengine==0.0.5

# install additional packages for examples
pip install numpy==1.26.3
```

Meanwhile, we highly recommend installing the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows),
as it helps with debugging.

Our examples assume that the Vulkan SDK is properly installed.
If you prefer to work without Vulkan SDK, you need to comment out the line containing `core.rhi.EnumContextExtension.DEBUG_UTILS` in our `common.py` file.

## Build from source

1. **Clone** this repository to a local folder, then open the folder in **Visual Studio** or **VS Code**.
2. **Set up the Python environment**:

   - Ensure it has **`numpy`**, **`torch`**, and **`nanobind`** installed.
   - You can create one using `conda`:
     ```bash
     # create an env called "myenv"
     conda create -n myenv python=3.9
     conda activate myenv
     # install numpy and nanobind
     pip install nanobind numpy
     # install pytorch version 2.4.0 with CUDA 11.8
     pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
     ```
   - Update the `"Python_EXECUTABLE"` variable in `CMakePresets.json` to point to the Python executable in this environment.
3. **Configure and build** the project:

   - Select your preferred **configure preset** and build the project.
4. **Install the built package**:

   - After compilation, a `.pyd` (Windows) or `.so` (Linux) file should be created under *release\sibylengine*, so:

     ```bash
     # Navigate to the `/release` folder.
     cd release
     # Activate the target Python environment.
     conda activate myenv
     # install the package
     pip install .
     ```
