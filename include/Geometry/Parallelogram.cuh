#ifndef RENDERERINTERACTIVE_PARALLELOGRAM_CUH
#define RENDERERINTERACTIVE_PARALLELOGRAM_CUH

#include <AS/BoundingBox.cuh>

namespace renderer {
    /*
     * 平行四边形类，普通类型
     * 起点为平行四边形在局部空间中的位置
     */
    class Parallelogram {
    public:
        //起点，两条边向量
        Point3 q;
        Vec3 u, v;

        //面积，所在平面法向量和平面方程的常数项参数d
        float area;
        Vec3 normalVector;
        float d;

        //材质索引
        MaterialType materialType;
        size_t materialIndex;

        Parallelogram(
                MaterialType materialType, size_t materialIndex,
                const Point3 & q, const Vec3 & u, const Vec3 & v)
                : materialType(materialType), materialIndex(materialIndex), q(q), u(u), v(v), normalVector{}
        {
            this->normalVector = Vec3::cross(u, v);
            this->area = normalVector.length();
            this->normalVector.unitize();
            float sum = 0.0;
            for (int i = 0; i < 3; i++) {
                sum += normalVector[i] * q[i];
            }
            this->d = sum;
        }

        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record) const;

        BoundingBox constructBoundingBox() const;

        Point3 centroid() const {
            return q + 0.5f * u + 0.5f * v;
        }

        size_t objectPrimitiveCount() const {
            return 1;
        }
    };
}

#endif //RENDERERINTERACTIVE_PARALLELOGRAM_CUH
