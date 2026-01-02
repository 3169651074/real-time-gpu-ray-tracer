[简体中文](./README_CN.md)

# RendererInteractive
A simple real time ray tracer implemented by pure CUDA.

## Build
The source code included in this repository can be compiled directly on Windows and Linux. Ensure that the NVIDIA graphics driver is updated to the latest version and download/install the CUDA Toolkit from the NVIDIA official website.

### Windows
1. Regardless of the tool used for compilation, compiling CUDA on Windows requires the MSVC compiler. Therefore, ensure that Visual Studio is installed. You can use the Visual Studio Installer to check the installation of the toolsets.
Ensure that the "Desktop development with C++" workload, as well as "C++ CMake tools for Windows" and the Windows 10/11 SDK in the list on the right side of this workload, are installed. It is recommended to update all toolsets to the latest version.

2. Compiling with an IDE (Visual Studio installed, no need to manually enter compilation commands in the terminal)

* Visual Studio  
Open the project folder directly, set the build option at the top to RendererParallel, and click Build.  
If CMake reports an error, go to the "Project" menu at the top --> Delete Cache and Reconfigure.

* CLion
1. Click the three horizontal lines button (Files) in the upper left corner --> Settings --> Build, Execution, Deployment.
2. Select Toolchains --> Visual Studio --> Change Architecture to "x86_amd64". (Note: If you encounter a CMake error indicating the compiler cannot compile test files, it is due to an incorrect architecture setting; do not select other architectures).
3. Select CMake under Toolchains and check the "Reload CMake project on..." option.
4. Select Visual Studio in the "Toolchain" below and set Generator to "Use Default (Ninja)" or manually specify it as Ninja.
5. Close the settings interface and click the green triangle arrow in the upper right corner of the IDE to compile and run.

### Ubuntu/Debian
1. Use the system package manager to install SDL2 dependency libraries and GCC:
```
sudo apt update && sudo apt install libsdl2-dev libsdl2-image-dev gcc g++
```

2. Building with CLion IDE:
1. Same as Windows, open Settings --> Build, Execution, Deployment --> CMake.
2. Specify the nvcc path parameter in "CMake options" under "Generator", depending on the version of the CUDA Toolkit. Example: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`. You can use the `whereis nvcc` command to check the actual installation location.
3. Keep the Toolchain as the default GCC.

* Project can be built directly in ternminal, but remember to pass CMAKE_CUDA_COMPILER argument.