#ifndef RENDERERINTERACTIVE_INSTANCE_CUH
#define RENDERERINTERACTIVE_INSTANCE_CUH

#include <AS/BLAS.cuh>

namespace renderer {
    /*
     * 实例类，普通类型
     * 实例作为TLAS的节点，不实际存储图元，只存储对BLAS的引用
     * 实例需要存储变换矩阵及其逆矩阵和逆转置矩阵，用于在世界空间和物体局部空间中来回变换光线和碰撞点法线
     *
     * 光线在TLAS中遍历 -> 找到可能命中的实例 -> 将光线变换到该实例的局部空间 ->
     *   在该实例引用的BLAS中继续遍历 -> 找到最终命中的三角形 -> 将命中结果变换回世界空间
     *
     * 加速结构创建：
     *   1.根据物体的几何属性，不包含位置信息，在局部空间中构建BVH树作为BLAS
     *   2.使用位置信息构建变换矩阵，和BVH树的引用共同构造实例对象
     *   3.使用所有实例对象构建BVH树作为TLAS
     *
     * 构造方法：
     *   使用从局部空间到世界空间的变换矩阵构建实例
     *   指定平移，旋转和缩放参数构建实例
     */
    class Instance {
    public:
        //实例指向的物体索引，由调用者指定，asIndex在构建加速结构时计算
        PrimitiveType primitiveType;
        size_t primitiveIndex;

        /*
         * BLAS索引
         * 此索引为该实例指向的BLAS在整个BLAS列表中的索引
         * 实例作为TLAS的组成元素，使用asIndex建立和对应BLAS的关联
         */
        size_t asIndex;

        //变换矩阵
        Matrix transformMatrix;         //从局部空间变换到世界空间
        Matrix transformInverse;        //从世界空间变换到局部空间
        Matrix normalTransformMatrix;

        //变换后物体的包围盒和几何中心，用于相交测试和构建顶层加速结构
        //此参数为世界空间参数
        BoundingBox transformedBoundingBox;
        Point3 transformedCentroid;

        //使用从局部空间到世界空间的变换矩阵构建实例
        Instance(PrimitiveType primitiveType, size_t primitiveIndex, const Matrix & transformMatrix);

        //使用变换参数构建实例
        Instance(PrimitiveType primitiveType, size_t primitiveIndex,
                 const std::array<float, 3> & rotate = {}, const std::array<float, 3> & shift = {}, const std::array<float, 3> & scale = {1.0, 1.0, 1.0});

        //设置变换前的包围盒和中心点，变换后存入对象
        void setBoundingBoxProperties(const BoundingBox & boundingBox, const Point3 & centroid);

        /*
         * 实例类碰撞检测函数：先将输入光线和TLAS的包围盒在世界空间中求交测试
         *   如果发生接触，则将光线使用逆变换矩阵变换到局部空间，使用asIndex找到对应的BLAS进行求交测试
         * 能调用到此函数，说明光线已经和TLAS的叶子节点，即实例，发生了碰撞，因此需要将光线变换到局部空间后和BLAS求交
         */
        __device__ bool hit(
                const BLASArray * __restrict__ blasArray,
                const Ray * ray, const Range * range, HitRecord * record,
                const Sphere * __restrict__ spheres, const Parallelogram * __restrict__ parallelograms) const;

    private:
        static Matrix makeTransform(const std::array<float, 3> & r, const std::array<float, 3> & s, const std::array<float, 3> & sc);
    };
}

#endif //RENDERERINTERACTIVE_INSTANCE_CUH
