#include <Util/Matrix.cuh>

namespace project {
    //辅助函数
    __host__ __device__ int eliminateBottomElements(float matrixData[5][9]) {
        //前向消元
        for (size_t i = 1; i < 5; i++) {
            //主元选择
            float main = abs(matrixData[i][i]);
            size_t maxRow = i;
            //选取当前行及以下行最大的主元
            for (size_t p = i + 1; p < 5; p++) {
                if (abs(matrixData[p][i]) > main) {
                    main = abs(matrixData[p][i]);
                    maxRow = p;
                }
            }
            //检查最大主元是否为零
            if (MathHelper::floatValueNearZero(main)) {
                return 1;
            }
            if (maxRow != i) {
                //交换第maxRow行和第i行
                float tmp[9] {};
                for (size_t j = 1; j < 9; j++) {
                    tmp[j] = matrixData[maxRow][j];
                }
                for (size_t j = 1; j < 9; j++) {
                    matrixData[maxRow][j] = matrixData[i][j];
                }
                for (size_t j = 1; j < 9; j++) {
                    matrixData[i][j] = tmp[j];
                }
            }
            //操作当前行（i）的下方所有行
            for (size_t j = i + 1; j < 5; j++) {
                const float factor = matrixData[j][i] / matrixData[i][i];
                //操作一行
                for (size_t k = i; k < 9; k++) {
                    matrixData[j][k] -= factor * matrixData[i][k];
                }
            }
        }
        return 0;
    }
    __host__ __device__ int eliminateTopElements(float matrixData[5][9]) {
        for (size_t i = 4; i >= 1; i--) {
            if (MathHelper::floatValueNearZero(matrixData[i][i])) {
                if (!MathHelper::floatValueNearZero(matrixData[i][8])) {
                    return 1; //无解
                } else {
                    return 2; //有无数个解
                }
            }
            //计算归一化系数。增广部分的所有元素都需要进行缩放
            float factor = 1.0f / matrixData[i][i];
            for (size_t p = i; p < 9; p++) {
                matrixData[i][p] *= factor;
            }
            for (size_t j = i - 1; j >= 1; j--) {
                factor = matrixData[j][i];
                for (size_t k = j; k < 9; k++) {
                    matrixData[j][k] -= factor * matrixData[i][k];
                }
            }
        }
        return 0;
    }

    //矩阵乘法
    __host__ __device__ Matrix Matrix::operator*(const Matrix & right) const {
        //创建新的矩阵，行数为左矩阵的行数，列数为右矩阵的列数
        Matrix ret{{}, row, right.col};

        //逐个元素赋值
        for (size_t i = 1; i <= ret.row; i++) {
            for (size_t j = 1; j <= ret.col; j++) {
                float sum = 0.0f;
                for (size_t n = 1; n <= col; n++) {
                    sum += data[i][n] * right.data[n][j];
                }
                ret.data[i][j] = sum;
            }
        }
        return ret;
    }

    //矩阵转置
    __host__ __device__ Matrix Matrix::transpose() const {
        Matrix ret{.row = col, .col = row};
        //将第i行变为第i列
        for (size_t i = 1; i <= row; i++) {
            for (size_t j = 1; j <= col; j++) {
                ret.data[j][i] = data[i][j];
            }
        }
        return ret;
    }

    //矩阵求逆
    __host__ __device__ Matrix Matrix::inverse() const {
        /*
         * 构造同阶单位矩阵，并将其合并到参数矩阵的右侧
         * 由于Matrix类的data限定大小为4x4，因此使用临时数组代替对象
         */
        float operateMatrix[5][9] {};
        for (size_t i = 1; i < 5; i++) {
            //将原矩阵数据填入左半部分
            for (size_t j = 1; j < 5; j++) {
                operateMatrix[i][j] = data[i][j];
            }
            //将右半部分设置为单位矩阵
            operateMatrix[i][4 + i] = 1.0f;
        }

        //对合并后的矩阵进行两次消元
        if (eliminateBottomElements(operateMatrix) != 0 || eliminateTopElements(operateMatrix) != 0) {
            //矩阵不满秩，无法求逆
            return *this;
        }

        //操作矩阵的右半部分即为所求
        Matrix ret{.row = 4, .col = 4};
        for (size_t i = 1; i < 5; i++) {
            for (size_t j = 1; j < 5; j++) {
                ret.data[i][j] = operateMatrix[i][4 + j];
            }
        }
        return ret;
    }

    //矩阵加减
    __host__ __device__ void Matrix::operator+=(const Matrix & obj) {
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++) {
                data[i][j] += obj.data[i][j];
            }
        }
    }
    __host__ __device__ Matrix Matrix::operator+(const Matrix & obj) const {
        Matrix ret = *this; ret += obj; return ret;
    }
    __host__ __device__ void Matrix::operator-=(const Matrix & obj) {
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++) {
                data[i][j] -= obj.data[i][j];
            }
        }
    }
    __host__ __device__ Matrix Matrix::operator-(const Matrix & obj) const {
        Matrix ret = *this; ret -= obj; return ret;
    }

    //矩阵数乘除
    __host__ __device__ void Matrix::operator*=(float num) {
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++) {
                data[i][j] *= num;
            }
        }
    }
    __host__ __device__ Matrix Matrix::operator*(float num) const {
        Matrix ret = *this; ret *= num; return ret;
    }
    __host__ __device__ Matrix operator*(float num, const Matrix & obj) {
        return obj * num;
    }
    __host__ __device__ void Matrix::operator/=(float num) {
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++) {
                data[i][j] /= num;
            }
        }
    }
    __host__ __device__ Matrix Matrix::operator/(float num) const {
        Matrix ret = *this; ret /= num; return ret;
    }
    __host__ __device__ Matrix operator/(float num, const Matrix & obj) {
        return obj / num;
    }

    //构造变换矩阵
    Matrix Matrix::constructShiftMatrix(const float3 & shift) {
        return {
                {
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f, shift.x,
                    0.0f, 0.0f, 1.0f, 0.0f, shift.y,
                    0.0f, 0.0f, 0.0f, 1.0f, shift.z,
                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                }, 4, 4
        };
    }

    Matrix Matrix::constructScaleMatrix(const float3 & scale) {
        return {
                {
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, scale.x, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, scale.y, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, scale.z, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                }, 4, 4
        };
    }

    Matrix Matrix::constructRotateMatrix(float degree, int axis) {
        const float theta = MathHelper::degreeToRadian(degree);
        switch (axis) {
            case 0:
                return {
                        {
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, cos(theta), -sin(theta), 0.0f,
                                0.0f, 0.0f, sin(theta), cos(theta), 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                        }, 4, 4
                };
            case 1:
                return {
                        {
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, cos(theta), 0.0f, sin(theta), 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, -sin(theta), 0.0f, cos(theta), 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                        }, 4, 4
                };
            case 2:
                return {
                        {
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, cos(theta), -sin(theta), 0.0f, 0.0f,
                                0.0f, sin(theta), cos(theta), 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                        }, 4, 4
                };
            default: return {};
        }
    }

    Matrix Matrix::constructRotateMatrix(const float3 & rotate) {
        const auto mx = constructRotateMatrix(rotate.x, 0);
        const auto my = constructRotateMatrix(rotate.y, 1);
        const auto mz = constructRotateMatrix(rotate.z, 2);
        return mx * my * mz;
    }

    std::string Matrix::toString() const {
        std::string ret("Matrix: ");
        ret += "Row = " + std::to_string(row) + ", Col = " + std::to_string(col) + "\n";
        for(size_t i = 1; i <= row; i++) {
            ret += "\t";
            for(size_t j = 1; j <= col; j++) {
                ret += std::to_string(data[i][j]) + " ";
            }
            ret += "\n";
        }
        return ret;
    }
}