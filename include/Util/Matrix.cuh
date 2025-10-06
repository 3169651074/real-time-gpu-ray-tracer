#ifndef RENDERERINTERACTIVE_MATRIX_CUH
#define RENDERERINTERACTIVE_MATRIX_CUH

#include <Basic/Point3.cuh>

namespace renderer {
    /*
     * 4x4矩阵类，聚合类型
     * 使用二维数组存储矩阵，为行主序：一行的内存连续排列
     * 由于操作均由单个GPU线程完成，逻辑上同CPU线程，则操作均支持设备端
     *
     * 转换操作：
     *   向量和矩阵转换
     *
     * 对象操作：
     *   矩阵相加减
     *   矩阵数乘除
     *   矩阵乘法
     *   求逆矩阵
     *   求转置矩阵
     */
    typedef struct Matrix {
        float data[5][5];
        size_t row, col;

        // ====== 矩阵转换 ======
        //向量转1行4列矩阵：向量不受平移影响，w = 0
        __host__ __device__ static Matrix toMatrix(const Vec3 & vec) {
            return {
                    {
                            {0.0f, 0.0f},
                            {0.0f, vec.x},
                            {0.0f, vec.y},
                            {0.0f, vec.z},
                            {0.0f, 0.0f}
                    },
                    4, 1
            };
        }
        //点转矩阵，点受平移影响，w = 1
        __host__ __device__ static Matrix toMatrix(const Point3 & p) {
            return {
                    {
                            {0.0f, 0.0f},
                            {0.0f, p.x},
                            {0.0f, p.y},
                            {0.0f, p.z},
                            {0.0f, 1.0f}
                    },
                    4, 1
            };
        }
        //1行4列矩阵转向量和点
        __host__ __device__ Vec3 toVector() const {
            return {data[1][1], data[2][1], data[3][1]};
        }
        __host__ __device__ Point3 toPoint() const {
            return {data[1][1], data[2][1], data[3][1]};
        }

        // ====== 对象操作 ======
        //矩阵加减
        __host__ __device__ void operator+=(const Matrix & obj);
        __host__ __device__ Matrix operator+(const Matrix & obj) const;
        __host__ __device__ void operator-=(const Matrix & obj);
        __host__ __device__ Matrix operator-(const Matrix & obj) const;

        //矩阵数乘除
        __host__ __device__ void operator*=(float num);
        __host__ __device__ Matrix operator*(float num) const;
        __host__ __device__ friend Matrix operator*(float num, const Matrix & obj);
        __host__ __device__ void operator/=(float num);
        __host__ __device__ Matrix operator/(float num) const;
        __host__ __device__ friend Matrix operator/(float num, const Matrix & obj);

        //矩阵乘法，返回新的矩阵
        __host__ __device__ Matrix operator*(const Matrix & obj) const;

        //求逆矩阵，返回新的矩阵
        __host__ __device__ Matrix inverse() const;

        //求转置矩阵，返回新的矩阵
        __host__ __device__ Matrix transpose() const;

        //构造变换矩阵
        static Matrix constructScaleMatrix(const std::array<float, 3> & scale);
        static Matrix constructShiftMatrix(const std::array<float, 3> & shift);
        static Matrix constructRotateMatrix(const std::array<float, 3> & rotate);
        static Matrix constructRotateMatrix(float degree, int axis);

        std::string toString() const;
    } Matrix;
}

#endif //RENDERERINTERACTIVE_MATRIX_CUH
