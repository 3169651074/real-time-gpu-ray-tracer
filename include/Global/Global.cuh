#ifndef RENDERERINTERACTIVE_GLOBAL_CUH
#define RENDERERINTERACTIVE_GLOBAL_CUH

#define SDL
#define CUDA
#define OGL

#ifdef SDL
#include <SDL2/SDL.h>
#endif //SDL

#ifdef CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif //CUDA

#ifdef OGL
//https://glad.dav1d.de/
#include <API/glad.h>
#include <API/khrplatform.h>
#ifdef CUDA
#include <cuda_gl_interop.h>
#endif //CUDA
#endif //OGL

#include <algorithm>
#include <string>
#include <array>
#include <vector>
#include <stack>

#include <cmath>
#include <cstring>
#include <cstdlib>

#include <random>
#include <limits>
#include <chrono>
#include <thread>

#undef INFINITY
#undef NULL

namespace renderer {
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

    // ====== 库包装 ======
#ifdef SDL
    void _SDL_CheckErrorIntImpl(int val, const char * file, const char * function, int line);
    void _SDL_CheckErrorPtrImpl(const void * val, const char * file, const char * function, int line);
#define SDL_CheckErrorInt(call) _SDL_CheckErrorIntImpl(call, __FILE__, __func__, __LINE__)
#define SDL_CheckErrorPtr(call) _SDL_CheckErrorPtrImpl(call, __FILE__, __func__, __LINE__)
#endif

#ifdef CUDA
    void _cudaCheckError(cudaError_t err, const char * file, const char * function, int line);
#define cudaCheckError(call) _cudaCheckError(call, __FILE__, __func__, __LINE__)
#endif

    // ====== 辅助宏 ======
#define lengthOf(name) sizeof(name) / sizeof(name[0])
}

#endif //RENDERERINTERACTIVE_GLOBAL_CUH
