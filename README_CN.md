[English](./README.md)

# RendererInteractive
一个简单的纯CUDA实时光线追踪器实现

## Build
此仓库中包含的源代码可直接在Windows和Linux上编译，确保NVIDA显卡驱动已经更新到最新版，并前往NVIDIA官网下载安装CUDA Toolkit。

### Windows
1. 无论使用什么工具编译，在Windows上编译CUDA必须使用MSVC编译器，因此需要确保Visual Studio已经安装，可以使用Visual Studio Installer查看工具集的安装情况。 
确保“使用C++的桌面开发”工具包以及此工具包右侧列表中“用于Windows的C++ CMake工具”、Windows 10/11 SDK已安装。建议将所有工具包更新到最新版。

2. 使用IDE编译（已经安装了Visual Studio，无需手动使用终端输入编译命令）

* Visual Studio  
直接打开项目文件夹，将上方生成选项设置为RendererParallel，点击生成即可   
如果CMake报错，则在上方“项目”菜单 -- > 删除缓存并重新配置

* CLion
1. 点击左上角三个横线的按钮（Files）--> Settings --> Build, Execution, Deployment
2. 选中Toolchains --> Visual Studio --> 将Architecture修改为“x86_amd64”。（注：如果遇到CMake报错，提示编译器无法编译测试文件，就是架构设置错误，不能选其他架构）
3. 选中Toolchains下方的CMake，勾选“Reload CMake project on...”选项。
4. 在下方的“Toolchain”中选择Visual Studio，将Generator设置为“Use Default（Ninja）”或手动指定为Ninja
5. 关闭设置界面，点击IDE右上角的绿色三角形箭头即可编译运行

### Ubuntu/Debian
1.使用系统包管理器安装SDL2依赖库和GCC
```
sudo apt update && sudo apt install libsdl2-dev libsdl2-image-dev gcc g++
```

2.使用CLion IDE构建
1. 同Windows，打开Settings --> Build, Execution, Deployment --> CMake
2. 在“Generator”下方的“CMake options”中指定nvcc的路径参数，具体取决于CUDA Toolkit的版本。示例：`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`。可以使用`whereis nvcc`命令查看实际安装位置
3. Toolchain保持默认的GCC即可

* 也可以直接在终端中使用CMake构建，注意传递CMAKE_CUDA_COMPILER参数即可
