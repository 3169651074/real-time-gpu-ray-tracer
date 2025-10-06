#ifndef RENDERERINTERACTIVE_RAY_CUH
#define RENDERERINTERACTIVE_RAY_CUH

#include <Basic/BasicTypes.cuh>
#include <Basic/Color3.cuh>
#include <Util/Matrix.cuh>

namespace renderer {
    /*
     * 光线类，聚合类型
     * 对象操作：
     *   获取光线在指定t值的位置点
     */
    typedef struct Ray {
        Point3 origin;
        Vec3 direction;

        __host__ __device__ Point3 at(float t) const {
            return origin + t * direction;
        }
    } Ray;
}

#endif //RENDERERINTERACTIVE_RAY_CUH
