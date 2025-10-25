#ifndef RENDERERINTERACTIVE_SPHERE_CUH
#define RENDERERINTERACTIVE_SPHERE_CUH

#include <AS/BoundingBox.cuh>

namespace project {
    /*
     * 球体类，普通类型（非聚合类型）
     * 图元构造时传入的位置为图元在局部空间中的位置，只和底层BLAS有关
     * 当一个物体包含多个图元时，每个图元在局部空间的位置共同决定物体形状
     *
     * 对象操作：
     *   碰撞检测
     *   获取几何中心
     *   构造包围盒
     */
    class Sphere {
    public:
        //球心和半径
        Point3 center;
        float radius;

        //材质索引
        MaterialType materialType;
        size_t materialIndex;

        Sphere(MaterialType materialType, size_t materialIndex, const Point3 & center, float radius)
        : center(center), radius(radius), materialType(materialType), materialIndex(materialIndex) {}

        //碰撞检测
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record) const;

        //构造包围盒
        BoundingBox constructBoundingBox() const;

        //获取几何中心
        Point3 centroid() const {
            return center;
        }

        //获取物体的图元数量，Sphere为单图元物体，只有一个球体图元
        size_t objectPrimitiveCount() const {
            return 1;
        }
    };
}

#endif //RENDERERINTERACTIVE_SPHERE_CUH
