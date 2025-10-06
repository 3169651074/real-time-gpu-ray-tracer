#ifndef RENDERERINTERACTIVE_RANGE_CUH
#define RENDERERINTERACTIVE_RANGE_CUH

#include <Global/Global.cuh>

namespace renderer {
    /*
     * 范围工具类，聚合类型
     * 构造操作：
     *   构造两个区间的并区间
     *
     * 对象操作：
     *   判断区间是否为空
     *   判断值是否在区间内（由参数指定端点开闭状态）
     *   将区间偏移指定值
     *   将区间扩展指定长度
     *   求区间长度
     *   裁剪值到区间范围内
     */
    typedef struct Range {
        float min, max;

        // ====== 构造操作 ======
        __host__ __device__ static Range merge(const Range & r1, const Range & r2) {
            return {r1.min < r2.min ? r1.min : r2.min, r1.max > r2.max ? r1.max : r2.max};
        }

        // ====== 对象操作 ======
        __host__ __device__ bool empty() const {
            return min >= max;
        }

        __host__ __device__ bool inRange(float val, bool leftClose = true, bool rightClose = true) const {
            const bool equalsToMin = MathHelper::floatValueEquals(val, min);
            const bool equalsToMax = MathHelper::floatValueEquals(val, max);

            //处理边界相等情况
            if (equalsToMin) return leftClose;
            if (equalsToMax) return rightClose;

            //处理严格区间
            return val > min && val < max;
        }

        __host__ __device__ void shift(float length) {
            min += length;
            max += length;
        }

        __host__ __device__ void expand(float endpointExtLength) {
            if (endpointExtLength > 0.0f) {
                min -= endpointExtLength;
                max += endpointExtLength;
            } else {
                min += endpointExtLength;
                max -= endpointExtLength;
            }
        }

        __host__ __device__ float length() const {
            if (min >= max || MathHelper::floatValueEquals(min, max)) {
                return 0.0f;
            } else {
                return max - min;
            }
        }

        __host__ __device__ float clamp(float val) const {
            if (val > max) {
                return max;
            } else if (val < min) {
                return min;
            } else {
                return val;
            }
        }
    } Range;
}

#endif //RENDERERINTERACTIVE_RANGE_CUH
