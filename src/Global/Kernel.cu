#include <Global/RendererImpl.cuh>

namespace project {
    __constant__ Camera dev_camera[1];

    __device__ Color3 rayColor(const TraverseData * dev_traverseData, const Ray * ray, curandState * state) {
        HitRecord record{};
        Ray currentRay = *ray;
        Color3 result{1.0f, 1.0f, 1.0f};
//#define NO_AS
#ifdef NO_AS
        for (size_t currentIterateDepth = 0; currentIterateDepth < dev_camera[0].rayTraceDepth; currentIterateDepth++) {
            //遍历所有物体，判断是否相交
            bool isHit = false;
            float closestT = INFINITY;

            for (size_t i = 0; i < dev_geometryData->sphereCount; i++) {
                HitRecord tempRecord;
                if (dev_geometryData->spheres[i].hit(currentRay, {0.001, closestT}, tempRecord)) {
                    isHit = true;
                    closestT = tempRecord.t;
                    record = tempRecord;
                    break;
                }
            }
            for (size_t i = 0; i < dev_geometryData->parallelogramCount; i++) {
                HitRecord tempRecord;
                if (dev_geometryData->parallelograms[i].hit(currentRay, {0.001, closestT}, tempRecord)) {
                    isHit = true;
                    closestT = tempRecord.t;
                    record = tempRecord;
                    break;
                }
            }

            if (isHit) {
                //发生碰撞，调用材质的散射函数，获取下一次迭代的光线
                Ray out;
                Color3 attenuation;

                //根据材质类型调用对应的散射函数
                switch (record.materialType) {
                    case MaterialType::ROUGH:
                        dev_materialData->roughs[record.materialIndex].scatter(state, currentRay, record, attenuation, out);
                        break;
                    case MaterialType::METAL:
                        if (!dev_materialData->metals[record.materialIndex].scatter(state, currentRay, record, attenuation, out)) {
                            return result;
                        }
                        break;
                    default:;
                }

                //更新光线和颜色
                currentRay = out;
                result *= attenuation;
            } else {
                //没有发生碰撞，将背景光颜色作为光源乘入结果并结束追踪循环
                result *= dev_camera[0].backgroundColor;
                break;
            }
        }
#else
        for (size_t currentIterateDepth = 0; currentIterateDepth < dev_camera[0].rayTraceDepth; currentIterateDepth++) {
            //通过TLAS进行碰撞测试
            static constexpr Range range = {0.001f, INFINITY};
            
            if (TLAS::hit(
                    dev_traverseData->tlasNodeArray, dev_traverseData->tlasIndexArray,
                    dev_traverseData->dev_instances, dev_traverseData->blasArray,
                    &currentRay, &range, &record,
                    dev_traverseData->dev_spheres, dev_traverseData->dev_parallelograms,
                    dev_traverseData->dev_triangles))
            {
                //发生碰撞，调用材质的散射函数，获取下一次迭代的光线
                Ray out;
                Color3 attenuation;

                //根据材质类型调用对应的散射函数
                switch (record.materialType) {
                    case MaterialType::ROUGH:
                        dev_traverseData->dev_roughs[record.materialIndex].scatter(state, currentRay, record, attenuation, out);
                        break;
                    case MaterialType::METAL:
                        if (!dev_traverseData->dev_metals[record.materialIndex].scatter(state, currentRay, record, attenuation, out)) {
                            return result;
                        }
                        break;
                    default:;
                }

                //更新光线和颜色
                currentRay = out;
                result *= attenuation;
            } else {
                //没有发生碰撞，将背景光颜色作为光源乘入结果并结束追踪循环
                result *= dev_camera[0].backgroundColor;
                break;
            }
        }
#endif
        return result;
    }

    __global__ void render(const TraverseData * dev_traverseData, cudaSurfaceObject_t surfaceObject) {
        //当前线程对应的全局像素坐标
        const Uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const Uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
        const Uint32 pixelIndex = gridDim.x * blockDim.x * y + x;
        if (x >= dev_camera[0].windowWidth || y >= dev_camera[0].windowHeight) return;

        //获取线程独立的随机数生成器
        curandState state;
        curand_init(pixelIndex ^ clock64(), pixelIndex, 0, &state);

        Color3 result{};

        //抗锯齿采样和光线颜色计算
        for (size_t sampleI = 0; sampleI < dev_camera[0].sqrtSampleCount; sampleI++) {
            for (size_t sampleJ = 0; sampleJ < dev_camera[0].sqrtSampleCount; sampleJ++) {
                const float offsetX = ((static_cast<float>(sampleJ) + RandomGenerator::randomDouble(&state)) * dev_camera[0].reciprocalSqrtSampleCount) - 0.5f;
                const float offsetY = ((static_cast<float>(sampleI) + RandomGenerator::randomDouble(&state)) * dev_camera[0].reciprocalSqrtSampleCount) - 0.5f;
                const Point3 samplePoint =
                        dev_camera[0].pixelOrigin + ((x + offsetX) * dev_camera[0].viewPortPixelDx) + ((y + offsetY) * dev_camera[0].viewPortPixelDy);

                //构造光线
                Point3 rayOrigin = dev_camera[0].cameraCenter;
                if (dev_camera[0].focusDiskRadius > 0.0f) {
                    //离焦采样：在离焦半径内随机选取一个点，以这个点发射光线
                    const Vec3 defocusVector = Vec3::randomPlaneVector(&state, dev_camera[0].focusDiskRadius);
                    //使用视口方向向量定位采样点
                    rayOrigin = dev_camera[0].cameraCenter + defocusVector[0] * dev_camera[0].cameraU + defocusVector[1] * dev_camera[0].cameraV;
                }
                const Vec3 rayDirection = Point3::constructVector(rayOrigin, samplePoint).unitVector();
                const Ray ray{rayOrigin, rayDirection};

                //发射光线
                result += rayColor(dev_traverseData, &ray, &state);
            }
        }

        //取平均值
        result *= dev_camera[0].reciprocalSqrtSampleCount * dev_camera[0].reciprocalSqrtSampleCount;

        //写入到缓冲区
        surf2Dwrite(result.castToUchar4(), surfaceObject, static_cast<int>(x * sizeof(uchar4)), static_cast<int>(y));
    }
}