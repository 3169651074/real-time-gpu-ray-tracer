#include <AS/Instance.cuh>

namespace renderer {
    Instance::Instance(PrimitiveType primitiveType, size_t primitiveIndex, const Matrix & transformMatrix)
    : primitiveType(primitiveType), primitiveIndex(primitiveIndex), asIndex(0),
    transformMatrix(transformMatrix), transformInverse(transformMatrix.inverse()),
    normalTransformMatrix(transformInverse.transpose()), transformedBoundingBox{}, transformedCentroid{}{}

    Instance::Instance(PrimitiveType primitiveType, size_t primitiveIndex,
    const std::array<float, 3> & rotate, const std::array<float, 3> & shift, const std::array<float, 3> & scale)
    : primitiveType(primitiveType), primitiveIndex(primitiveIndex), asIndex(0),
    transformMatrix(makeTransform(rotate, shift, scale)), transformInverse(transformMatrix.inverse()),
    normalTransformMatrix(transformInverse.transpose()), transformedBoundingBox{}, transformedCentroid{} {}

    void Instance::setBoundingBoxProperties(const BoundingBox & boundingBox, const Point3 & centroid) {
        transformedBoundingBox = boundingBox.transformBoundingBox(transformMatrix);
        transformedCentroid = (transformMatrix * Matrix::toMatrix(centroid)).toPoint();
    }

    __device__ bool Instance::hit(
            const BLASArray * const __restrict__ blasArray,
            const Ray * ray, const Range * range, HitRecord * record,
            const Sphere * const __restrict__ spheres, const Parallelogram * const __restrict__ parallelograms) const
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
                spheres, parallelograms))
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

    Matrix Instance::makeTransform(const std::array<float, 3> & r, const std::array<float, 3> & s, const std::array<float, 3> & sc) {
        const auto m1 = Matrix::constructShiftMatrix(s);
        const auto m2 = Matrix::constructRotateMatrix(r);
        const auto m3 = Matrix::constructScaleMatrix(sc);
        return m1 * m2 * m3;
    }
}