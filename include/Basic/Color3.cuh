#ifndef RENDERERINTERACTIVE_COLOR3_CUH
#define RENDERERINTERACTIVE_COLOR3_CUH

#include <Util/Range.cuh>

namespace renderer {
    /*
     * RGB颜色类，聚合类型
     * 成员访问：
     *   通过下标访问向量分量（下标必须为0，1或2）
     *
     * 对象操作：
     *   颜色相加减，乘除（均为各分量进行对应操作）
     *   颜色数乘除，允许左操作数为实数
     *   将颜色转换为uchar4
     *
     * 随机颜色生成：
     *   生成每个分量都在指定范围内的随机颜色
     */
    typedef struct Color3 {
        float r, g, b;

        // ====== 成员访问 ======
        __host__ __device__ float operator[](size_t index) const {
            switch (index) {
                case 0: return r; case 1: return g;
                case 2: return b; default: return r;
            }
        }
        __host__ __device__ float & operator[](size_t index) {
            switch (index) {
                case 0: return r; case 1: return g;
                case 2: return b; default: return r;
            }
        }

        // ====== 对象操作 ======
        //颜色加减
        __host__ __device__ void operator+=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) += obj[i];
            }
        }
        __host__ __device__ Color3 operator+(const Color3 & obj) const {
            Color3 ret = *this; ret += obj; return ret;
        }
        __host__ __device__ void operator-=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) -= obj[i];
            }
        }
        __host__ __device__ Color3 operator-(const Color3 & obj) const {
            Color3 ret = *this; ret -= obj; return ret;
        }

        //颜色乘除
        __host__ __device__ void operator*=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) *= obj[i];
            }
        }
        __host__ __device__ Color3 operator*(const Color3 & obj) const {
            Color3 ret = *this; ret *= obj; return ret;
        }
        __host__ __device__ void operator/=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) /= obj[i];
            }
        }
        __host__ __device__ Color3 operator/(const Color3 & obj) const {
            Color3 ret = *this; ret /= obj; return ret;
        }

        //数乘除
        __host__ __device__ void operator*=(float num) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) *= num;
            }
        }
        __host__ __device__ Color3 operator*(float num) const {
            Color3 ret = *this; ret *= num; return ret;
        }
        __host__ __device__ friend Color3 operator*(float num, const Color3 & obj) {
            return obj * num;
        }
        __host__ __device__ void operator/=(float num) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) /= num;
            }
        }
        __host__ __device__ Color3 operator/(float num) const {
            Color3 ret = *this; ret /= num; return ret;
        }
        __host__ __device__ friend Color3 operator/(float num, const Color3 & obj) {
            return obj / num;
        }

        //颜色转换
        __device__ uchar4 castToUchar4(float gamma = 2.0f) const {
            //伽马校正
            const float power = 1.0f / gamma;
            const float _r = pow(r, power);
            const float _g = pow(g, power);
            const float _b = pow(b, power);

            //将[0.0, 1.0]的颜色值映射到[0, 255]
            static constexpr Range colorRange{0.0f, 0.999f};
            const auto intR = static_cast<Uint8>(256.0f * colorRange.clamp(_r));
            const auto intG = static_cast<Uint8>(256.0f * colorRange.clamp(_g));
            const auto intB = static_cast<Uint8>(256.0f * colorRange.clamp(_b));

            //存储为uchar4
            return make_uchar4(intR, intG, intB, 255);
        }

        // ====== 随机颜色生成 ======
        static Color3 randomColor(float componentMin = 0.0f, float componentMax = 1.0f) {
            Color3 ret{};
            for (size_t i = 0; i < 3; i++) {
                ret[i] = RandomGenerator::randomDouble(componentMin, componentMax);
            }
            return ret;
        }
        __device__ static Color3 randomColor(curandState * state, float componentMin = 0.0f, float componentMax = 1.0f) {
            Color3 ret{};
            for (size_t i = 0; i < 3; i++) {
                ret[i] = RandomGenerator::randomDouble(state, componentMin, componentMax);
            }
            return ret;
        }
    } Color3;
}

#endif //RENDERERINTERACTIVE_COLOR3_CUH
