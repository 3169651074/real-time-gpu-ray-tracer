cmake_minimum_required(VERSION 4.0)
set(EXECUTABLE_NAME "RendererInteractive")

#设置CUDA编译器
if (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()
project(${EXECUTABLE_NAME} LANGUAGES C CXX CUDA)

#启用调试
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -G")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -G")

#设置语言标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
include_directories("${CMAKE_SOURCE_DIR}/include")

#静态链接
if (WIN32)
    link_directories("${CMAKE_SOURCE_DIR}/lib")
else ()
    find_package(SDL2 REQUIRED)
endif ()

#查找源文件
#file(GLOB_RECURSE C_SRCS "${CMAKE_SOURCE_DIR}/src/*.c")
#file(GLOB_RECURSE CPP_SRCS "${CMAKE_SOURCE_DIR}/src/*.cpp")
#file(GLOB_RECURSE CU_SRCS "${CMAKE_SOURCE_DIR}/src/*.cu")
add_executable(${EXECUTABLE_NAME} #${C_SRCS} ${CPP_SRCS} ${CU_SRCS}
        src/Global/Main.cu
        include/Global/Global.cuh
        src/API/glad.c
        src/Global/Global.cu
        include/Basic/Vec3.cuh
        src/Basic/Vec3.cu
        include/Basic/Point3.cuh
        include/Util/Pair.cuh
        include/Util/Range.cuh
        include/Basic/Color3.cuh
        include/Basic/Ray.cuh
        include/Geometry/Sphere.cuh
        include/Material/Rough.cuh
        include/Basic/BasicTypes.cuh
        include/Geometry/Parallelogram.cuh
        include/Global/Render.cuh
        include/Material/Metal.cuh
        src/Global/RenderPin.cu
        include/Util/Matrix.cuh
        src/Util/Matrix.cu
        include/AS/BoundingBox.cuh
        src/AS/BoundingBox.cu
        src/Geometry/Sphere.cu
        src/Geometry/Parallelogram.cu
        src/Global/Kernel.cu
        include/AS/Instance.cuh
        include/AS/BLAS.cuh
        src/AS/BLAS.cu
        include/AS/TLAS.cuh
        src/AS/TLAS.cu
        src/AS/Instance.cu
        include/Global/SDL_OpenGLWindow.cuh
        src/Global/SDL_OpenGLWindow.cu
        src/Global/RenderGlob.cu
)

#MSVC编译器必须指定以UTF-8编码读取文件，否则在Windows上需要将所有文件的编码设置为UTF-8 with BOM
if (WIN32)
    target_compile_options(${EXECUTABLE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif ()

#启用多文件链接支持
set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#链接SDL2库
if (WIN32)
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC SDL2main)
endif ()
target_link_libraries(${EXECUTABLE_NAME} PUBLIC SDL2)
