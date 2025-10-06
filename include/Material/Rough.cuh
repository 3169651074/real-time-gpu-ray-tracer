#ifndef RENDERERINTERACTIVE_ROUGH_CUH
#define RENDERERINTERACTIVE_ROUGH_CUH

#include <Basic/Ray.cuh>

namespace renderer {
    /*
     * 粗糙材质类，聚合类型
     * 漫反射光线：在半球内随机选择一个方向作为出射方向
     */
    typedef struct Rough {
        Color3 albedo;

        __device__ bool scatter(curandState * state, const Ray & in, const HitRecord & record, Color3 & attenuation, Ray & out) const {
            //随机选择一个反射方向
            Vec3 reflectDirection = record.normalVector + Vec3::randomSpaceVector(state, 1.0f);

            //若随机的反射方向和法向量相互抵消，则取消随机反射
            if (MathHelper::floatValueEquals(reflectDirection.lengthSquared(), FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE)) {
                reflectDirection = record.normalVector;
            }

            //从反射点出发构造反射光线
            out = {record.hitPoint, reflectDirection};

            //颜色衰减因子为当前材质的基础颜色
            attenuation = albedo;
            return true;
        }
    } Rough;
}

#endif //RENDERERINTERACTIVE_ROUGH_CUH
