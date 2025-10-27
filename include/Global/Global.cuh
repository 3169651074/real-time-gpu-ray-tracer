#ifndef RENDERERINTERACTIVE_GLOBAL_CUH
#define RENDERERINTERACTIVE_GLOBAL_CUH

/*
 * 文件目录结构：
 * include/
 *   SDL2/
 *     ...
 *   OGL/
 *     glad.h
 *     khrplatform.h
 *   Global/
 *     Global.cuh
 * src/
 *   OGL/
 *     glad.c
 *   Global/
 *     Global.cu
 */

//当前项目名字空间名称
#define project renderer

// ====== 功能开关 ======

#define SDL
#define CUDA
//#define OPTIX

#define OGL
//#define D3D11
//#define D3D12
//#define VK

// ====== 库头文件包含 ======

//Win32
#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max
#endif

//SDL
#ifdef SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#endif //SDL

//CUDA
#ifdef CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif //CUDA

//OPTIX
#if defined(CUDA) && defined(OPTIX)
#include <optix.h>
#include <optix_types.h>
#include <optix_device.h>
#include <optix_function_table.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

/*
 * optix_function_table_definition.h在全局作用域中声明了OptixFunctionTable的全局变量
 * 由于全局变量不能重复定义，因此在Global.cu中包含optix_function_table_definition.h，在此处声明
 */
extern OptixFunctionTable OPTIX_FUNCTION_TABLE_SYMBOL;
#endif //OPTIX

//OGL
#ifdef OGL
//https://glad.dav1d.de/
#include <OGL/glad.h>
#include <OGL/khrplatform.h>
#endif //OGL
//OGL+CUDA
#if defined(OGL) && defined(CUDA)
#include <cuda_gl_interop.h>
#endif //OGL+CUDA

//D3D11
#if defined(_WIN32) && defined(D3D11)
#include <d3d11.h>
#include <dxgi1_5.h>
#include <wrl.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#endif //D3D11
//D3D11+CUDA
#if defined(_WIN32) && defined(D3D11) && defined(CUDA)
#include <cuda_d3d11_interop.h>
#endif //D3D11+CUDA

//D3D12
#if defined(_WIN32) && defined(D3D12)
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl.h>
#include <D3D12/directx/d3d12.h>
#include <D3D12/directx/d3dx12.h>
#endif //D3D12

#if defined(_WIN32) && (defined(D3D11) || defined(D3D12))
using namespace DirectX;
using namespace Microsoft::WRL;
#endif

//VK
#ifdef VK
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif
#endif
//VK+SDL
#if defined(VK) && defined(SDL)
#include <SDL2/SDL_vulkan.h>
#endif //VK+SDL

//C++
#include <algorithm>
#include <string>
#include <array>
#include <vector>
#include <stack>
#include <unordered_map>

#include <cmath>
#include <cstring>
#include <cstdlib>

#include <random>
#include <limits>
#include <chrono>
#include <thread>

#undef INFINITY

namespace project {
    // ====== 数值常量 ======
    constexpr float FLOAT_ZERO_VALUE = 1e-6f;
    constexpr float INFINITY = std::numeric_limits<float>::infinity();
    constexpr float PI = M_PI;

    // ====== 数学工具函数 ======
    class MathHelper {
    public:
#ifdef CUDA
        __host__ __device__
#endif
        static float degreeToRadian(float degree) {
            return degree * PI / 180.0f;
        }

#ifdef CUDA
        __host__ __device__
#endif
        static float radianToDegree(float radian) {
            return radian * 180.0f / PI;
        }

        //判断浮点数是否接近于0
#ifdef CUDA
        __host__ __device__
#endif
        static bool floatValueNearZero(float val) {
            return abs(val) < FLOAT_ZERO_VALUE;
        }

        //判断两个浮点数是否相等
#ifdef CUDA
        __host__ __device__
#endif
        static bool floatValueEquals(float v1, float v2) {
            return abs(v1 - v2) < FLOAT_ZERO_VALUE;
        }
    };

    // ====== 随机数生成函数 ======
    class RandomGenerator {
    private:
        static inline std::random_device rd;
        static inline std::mt19937 generator{rd()};
        static inline std::uniform_real_distribution<float> distribution{0.0f, 1.0f};

    public:
        //生成一个[0, 1)的浮点随机数
        static float randomDouble() {
            return distribution(generator);
        }
#ifdef CUDA
        __device__ static float randomDouble(curandState * state) {
            return curand_uniform(state);
        }
#endif

        //生成一个[min, max)之间的浮点随机数
        static float randomDouble(float min, float max) {
            return min + (max - min) * randomDouble();
        }
#ifdef CUDA
        __device__ static float randomDouble(curandState * state, float min, float max) {
            return min + (max - min) * randomDouble(state);
        }
#endif

        //生成一个[min, max]之间的整数随机数
        template<typename T>
        static T randomInteger(T min, T max) {
            std::uniform_int_distribution<T> _distribution(min, max);
            return _distribution(generator);
        }
#ifdef CUDA
        template<typename T>
        __device__ static T randomInteger(curandState * state, T min, T max) {
            const float val = min + ((max + 1) - min) * randomDouble(state);
            return static_cast<T>(val);
        }
#endif
    };

    //文件读写函数
    class FileHelper {
    public:
        //读取整个文本文件并存入字符串
        static std::string readTextFile(const std::string & filePath);

        //将字符串中所有内容以纯文本写入到指定文件
        static void writeTextFile(const std::string & filePath, const std::string & content);
    };

    // ====== 库包装 ======

    //SDL
#ifdef SDL
#define SDL_ERROR_EXIT_CODE (-100)
    extern void _SDL_CheckErrorIntImpl(int val, const char * file, const char * function, int line);
    extern void _SDL_CheckErrorPtrImpl(const void * val, const char * file, const char * function, int line);
    extern void _SDL_CheckErrorPtrBool(SDL_bool val, const char * file, const char * function, int line);
#define SDL_CheckErrorInt(call) _SDL_CheckErrorIntImpl(call, __FILE__, __func__, __LINE__)
#define SDL_CheckErrorPtr(call) _SDL_CheckErrorPtrImpl(call, __FILE__, __func__, __LINE__)
#define SDL_CheckErrorBool(call) _SDL_CheckErrorPtrBool(call, __FILE__, __func__, __LINE__)
#endif

    //CUDA
#ifdef CUDA
#define CUDA_ERROR_EXIT_CODE (-200)
    extern void _cudaCheckError(cudaError_t err, const char * file, const char * function, int line);
#define cudaCheckError(call) _cudaCheckError(call, __FILE__, __func__, __LINE__)
#endif

    //OPTIX
#if defined(CUDA) && defined(OPTIX)
#define OPTIX_ERROR_EXIT_CODE (-300)
    extern void _optixCheckError(OptixResult result, const char * file, const char * function, int line);
#define optixCheckError(call) _optixCheckError(call, __FILE__, __func__, __LINE__)

    //检查每个调用的日志信息，如果有则打印
    //使用全局定义的optixPerCallLogBuffer字符数组和optixPerCallLogSize长度变量
#define optixCheckPerCallLog \
    do {                     \
        if (optixPerCallLogSize > 1) SDL_Log("OptiX call log: %s", optixPerCallLogBuffer);\
    } while (false)

    extern char optixPerCallLogBuffer[2048];
    extern size_t optixPerCallLogSize;
    extern void optixLogCallBackFunction(Uint32 level, const char * tag, const char * message, void * callbackData);
#endif

    //D3D
#if defined(_WIN32) && (defined(D3D11) || defined(D3D12))
#define D3D_ERROR_EXIT_CODE (-400)
    extern void _D3DPrintErrorMessage(HRESULT result, const char * file, const char * function, int line);
#define D3DCheckError(result) \
    do {                      \
        if (FAILED(result)) { \
            _D3DPrintErrorMessage(result, __FILE__, __func__, __LINE__);\
        }                     \
    } while (false)
#endif

    //VK
#ifdef VK
#define VULKAN_ERROR_EXIT_CODE (-500)
    extern void _vkCheckError(VkResult result, const char * file, const char * function, int line);
#define vkCheckError(result) _vkCheckError(result, __FILE__, __func__, __LINE__)

    extern VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugCallBackFunction(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void * pUserData);
#endif

    // ====== 辅助宏 ======
#define lengthOf(name) sizeof(name) / sizeof(name[0])
}

#endif //RENDERERINTERACTIVE_GLOBAL_CUH
