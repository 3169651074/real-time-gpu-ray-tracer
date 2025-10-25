#ifndef RENDERERINTERACTIVE_POINT3_CUH
#define RENDERERINTERACTIVE_POINT3_CUH

#include <Basic/Vec3.cuh>

namespace project {
    /*
     * 空间点类，聚合类型
     * 成员访问：
     *   通过下标访问向量分量（下标必须为0，1或2）
     *
     * 对象操作：
     *   按向量偏移（点加减向量）
     *   点转向量、向量转点
     *   获取两点距离
     *   通过两点构造向量
     */
    typedef struct Point3 {
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
        //点和向量转换
        __host__ __device__ Vec3 toVector() const {
            return {x, y, z};
        }
        __host__ __device__ static Point3 toPoint(const Vec3 & vec) {
            return {vec.x, vec.y, vec.z};
        }

        //点加减向量
        __host__ __device__ void operator+=(const Vec3 & offset) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) += offset[i];
            }
        }
        __host__ __device__ Point3 operator+(const Vec3 & offset) const {
            Point3 ret = *this; ret += offset; return ret;
        }
        __host__ __device__ void operator-=(const Vec3 & offset) {
            for (size_t i = 0; i < 3; i++) {
                operator[](i) -= offset[i];
            }
        }
        __host__ __device__ Point3 operator-(const Vec3 & offset) const {
            Point3 ret = *this; ret -= offset; return ret;
        }

        //通过两点构造向量
        __host__ __device__ static Vec3 constructVector(const Point3 & from, const Point3 & to) {
            Vec3 ret{};
            for (size_t i = 0; i < 3; i++) {
                ret[i] = to[i] - from[i];
            }
            return ret;
        }
        __host__ __device__ friend Vec3 operator-(const Point3 & lhs, const Point3 & rhs) {
            return constructVector(rhs, lhs);
        }

        //获取两点间距
        __host__ __device__ float distanceSquared(const Point3 & anotherPoint) const {
            float sum = 0.0;
            for (size_t i = 0; i < 3; i++) {
                sum += (operator[](i) - anotherPoint[i]) * (operator[](i) - anotherPoint[i]);
            }
            return sum;
        }
        __host__ __device__ float distance(const Point3 & anotherPoint) const {
            return sqrt(distanceSquared(anotherPoint));
        }
    } Point3;
}

#endif //RENDERERINTERACTIVE_POINT3_CUH
