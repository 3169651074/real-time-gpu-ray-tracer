#include <AS/BoundingBox.cuh>

namespace project {
    BoundingBox BoundingBox::transformBoundingBox(const Matrix & matrix) const {
        //使用矩阵对包围盒的8个顶点进行变换
        Point3 min{INFINITY, INFINITY, INFINITY};
        Point3 max{-INFINITY, -INFINITY, -INFINITY};

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    //取出每个顶点的坐标
                    const float x = i * range[0].max + (1.0f - i) * range[0].min;
                    const float y = j * range[1].max + (1.0f - j) * range[1].min;
                    const float z = k * range[2].max + (1.0f - k) * range[2].min;

                    //计算变换后的坐标
                    const auto matrixPoint = Matrix::toMatrix(Point3{x, y, z});
                    const auto mul = matrix * matrixPoint;
                    const auto point = mul.toPoint();

                    //计算最值，保证包围盒和坐标轴对齐
                    for (int l = 0; l < 3; l++) {
                        min[l] = std::min(min[l], point[l]);
                        max[l] = std::max(max[l], point[l]);
                    }
                }
            }
        }
        //使用min和max重构包围盒
        return {min, max};
    }

    __device__ bool BoundingBox::hit(const Ray & ray, const Range & checkRange, float & t) const {
        const Point3 & rayOrigin = ray.origin;
        const Vec3 & rayDirection = ray.direction;

        Range currentRange(checkRange);
        for (Uint32 axis = 0; axis < 3; axis++) {
            const Range & axisRange = range[axis];
            const float q = rayOrigin[axis];
            const float d = rayDirection[axis];

            //光线和包围盒平行
            if (abs(d) < FLOAT_ZERO_VALUE) {
                //光线起点不在包围盒内，没有交点
                if (q < axisRange.min || q > axisRange.max) return false;
                //光线从包围盒内部发出，继续测试下一个轴
                continue;
            }

            //计算光在当前轴和边界的两个交点
            float t1 = (axisRange.min - q) / d;
            float t2 = (axisRange.max - q) / d;

            //将currentRange限制到这两个交点的范围内
            if (t1 < t2) {
                if (t1 > currentRange.min) currentRange.min = t1;
                if (t2 < currentRange.max) currentRange.max = t2;
            } else {
                if (t2 > currentRange.min) currentRange.min = t2;
                if (t1 < currentRange.max) currentRange.max = t1;
            }

            if (currentRange.empty()) {
                return false;
            }
        }

        t = currentRange.min;
        return true;
    }
}