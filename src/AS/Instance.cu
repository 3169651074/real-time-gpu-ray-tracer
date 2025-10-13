#include <AS/Instance.cuh>

namespace renderer {
    void Instance::updateTransformArguments(float3 shift, float3 rotate, float3 scale) {
        const auto shiftMatrix = Matrix::constructShiftMatrix(shift);
        const auto rotateMatrix = Matrix::constructRotateMatrix(rotate);
        const auto scaleMatrix = Matrix::constructScaleMatrix(scale);

        //计算变换矩阵
        transformMatrix = shiftMatrix * rotateMatrix * scaleMatrix;
        transformInverse = transformMatrix.inverse();
        normalTransformMatrix = transformInverse.transpose();

        //更新包围盒和中心点
        transformedBoundingBox = boundingBox.transformBoundingBox(transformMatrix);
        transformedCentroid = (transformMatrix * Matrix::toMatrix(centroid)).toPoint();
    }

    __device__ bool Instance::hit(
            const BLASArray * const __restrict__ blasArray,
            const Ray * ray, const Range * range, HitRecord * record,
            const Sphere * const __restrict__ spheres, const Parallelogram * const __restrict__ parallelograms,
            const Triangle * const __restrict__ triangles) const
    {
        //变换光线到局部空间
        const auto rayOrigin = (transformInverse * Matrix::toMatrix(ray->origin)).toPoint();
        const auto rayDirection = (transformInverse * Matrix::toMatrix(ray->direction)).toVector();
        const Ray transformedRay{rayOrigin, rayDirection};

        //在局部空间中和BLAS求交
        const auto instanceBlasArray = blasArray[asIndex];
        const auto instanceBlasNodeArray = instanceBlasArray.first.first;
        const auto instanceBlasIndexArray = instanceBlasArray.second.first;

        if (BLAS::hit(
                instanceBlasNodeArray, instanceBlasIndexArray, &transformedRay, range, record,
                spheres, parallelograms, triangles))
        {
            //如果有碰撞，则将将局部空间的命中记录变换回世界空间，t值和uv坐标不需要变换
            //使用正变换矩阵变换碰撞点
            record->hitPoint = (transformMatrix * Matrix::toMatrix(record->hitPoint)).toPoint();

            //使用逆转置变换矩阵变换法向量
            record->normalVector = (normalTransformMatrix * Matrix::toMatrix(record->normalVector)).toVector().unitVector();
            record->hitFrontFace = Vec3::dot(ray->direction, record->normalVector) < 0.0f;
            return true;
        } else {
            return false;
        }
    }
}