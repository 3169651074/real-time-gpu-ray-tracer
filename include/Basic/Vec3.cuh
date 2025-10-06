#ifndef RENDERERINTERACTIVE_VEC3_CUH
#define RENDERERINTERACTIVE_VEC3_CUH

#include <Global/Global.cuh>

namespace renderer {
    /*
     * 三维向量类，聚合类型
     * 成员访问：
     *   通过下标访问向量分量（下标必须为0，1或2）
     *
     * 对象操作：
     *   取负向量
     *   向量加减
     *   向量数乘除，允许左操作数为实数
     *   求模长
     *   点乘，叉乘
     *   单位化
     *   绕指定轴旋转
     *
     * 随机向量生成：
     *   生成每个分量都在指定范围内的随机向量
     *   生成平面（x，y，0）上模长不大于指定长度的向量
     *   生成指定模长的空间向量
     *   生成遵守按指定轴余弦分布的随机向量（单位向量）
     */
    typedef struct Vec3 {
        float x, y, z;

        // ====== 成员访问 ======
        __host__ __device__ float operator[](size_t index) const {
            switch (index) {
                case 0: return x; case 1: return y;
                case 2: return z; default: return x;
            }
        }
        __host__ __device__ float & operator[](size_t index) {
            switch (index) {
                case 0: return x; case 1: return y;
                case 2: return z; default: return x;
            }
        }

        // ====== 对象操作 ======
        //取负
        __host__ __device__ Vec3 operator-() const {
            return {-x, -y, -z};
        }
        __host__ __device__ Vec3 negativeVector() const {
            return {-x, -y, -z};
        }
        __host__ __device__ void negate() {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) = -operator[](i);
            }
        }

        //加减
        __host__ __device__ void operator+=(const Vec3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) += obj[i];
            }
        }
        __host__ __device__ Vec3 operator+(const Vec3 & obj) const {
            Vec3 ret = *this; ret += obj; return ret;
        }
        __host__ __device__ void operator-=(const Vec3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) -= obj[i];
            }
        }
        __host__ __device__ Vec3 operator-(const Vec3 & obj) const {
            Vec3 ret = *this; ret -= obj; return ret;
        }

        //数乘除
        __host__ __device__ void operator*=(float num) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) *= num;
            }
        }
        __host__ __device__ Vec3 operator*(float num) const {
            Vec3 ret = *this; ret *= num; return ret;
        }
        __host__ __device__ void operator/=(float num) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) /= num;
            }
        }
        __host__ __device__ Vec3 operator/(float num) const {
            Vec3 ret = *this; ret /= num; return ret;
        }
        __host__ __device__ friend Vec3 operator*(float num, const Vec3 & obj) {
            return obj * num;
        }
        __host__ __device__ friend Vec3 operator/(float num, const Vec3 & obj) {
            return obj / num;
        }

        //求模长
        __host__ __device__ float lengthSquared() const {
            float sum = 0.0f;
            for (size_t i = 0; i < 3; i++) {
                sum += operator[](i) * operator[](i);
            }
            return sum;
        }
        __host__ __device__ float length() const {
            return sqrt(lengthSquared());
        }

        //点乘叉乘
        __host__ __device__ float dot(const Vec3 & obj) const {
            float sum = 0.0f;
            for (size_t i = 0; i < 3; i++) {
                sum += operator[](i) * obj[i];
            }
            return sum;
        }
        __host__ __device__ Vec3 cross(const Vec3 & obj) const {
            return {
                y * obj.z - z * obj.y,
                z * obj.x - x * obj.z,
                x * obj.y - y * obj.x
            };
        }

        //单位化
        __host__ __device__ void unitize() {
            const float factor = 1.0f / length();
            for (size_t i = 0; i < 3; i++) {
                operator[](i) *= factor;
            }
        }
        __host__ __device__ Vec3 unitVector() const {
            Vec3 ret = *this; ret.unitize(); return ret;
        }

        //返回当前向量绕axis轴（不要求单位向量）旋转angle角度后的结果，使用罗德里格旋转公式实现
        __host__ __device__ Vec3 rotate(const Vec3 & axis, float angle) const {
            //罗德里格公式要求旋转轴k是单位向量
            const Vec3 k = axis.unitVector();

            //计算三角函数值
            const float cosTheta = cos(angle);
            const float sinTheta = sin(angle);
            //v 是要旋转的向量
            const Vec3 & v = *this;

            //根据罗德里格公式计算三个部分
            //part1: v * cos(theta)，旋转向量平行于旋转轴的分量，此分量不变
            const Vec3 part1 = v * cosTheta;
            //旋转向量垂直于旋转轴的分量，需要绕轴旋转，将其分解为两个垂直分量分别计算
            //part2: (k x v) * sin(theta)
            const Vec3 part2 = k.cross(v) * sinTheta;
            //part3: k * (k . v) * (1 - cos(theta))
            const Vec3 part3 = k * k.dot(v) * (1.0f - cosTheta);

            //将三部分相加得到最终的旋转后向量
            return part1 + part2 + part3;
        }

        // ====== 随机向量生成 ======
        //生成每个分量都在指定范围内的随机向量
        static Vec3 randomVector(float componentMin, float componentMax);
        __device__ static Vec3 randomVector(curandState * state, float componentMin, float componentMax);

        //生成平面（x，y，0）上模长不大于指定长度的向量
        static Vec3 randomPlaneVector(float maxLength);
        __device__ static Vec3 randomPlaneVector(curandState * state, float maxLength);

        //生成模长为length的空间向量
        static Vec3 randomSpaceVector(float length);
        __device__ static Vec3 randomSpaceVector(curandState * state, float length);

        //生成遵守按指定轴余弦分布的随机向量，非单位向量
        static Vec3 randomCosineVector(int axis, bool toPositive);
        __device__ static Vec3 randomCosineVector(curandState * state, int axis, bool toPositive);

        //点乘和叉乘提供静态版本
        __host__ __device__ static float dot(const Vec3 & v1, const Vec3 & v2) {
            return v1.dot(v2);
        }
        __host__ __device__ static Vec3 cross(const Vec3 & v1, const Vec3 & v2) {
            return v1.cross(v2);
        }
    } Vec3;
}

#endif //RENDERERINTERACTIVE_VEC3_CUH
