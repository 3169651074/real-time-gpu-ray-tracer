#ifndef RENDERERINTERACTIVE_BLAS_CUH
#define RENDERERINTERACTIVE_BLAS_CUH

#include <Geometry/Sphere.cuh>
#include <Geometry/Parallelogram.cuh>
#include <Geometry/Triangle.cuh>

namespace renderer {
    /*
     * 底层加速结构BVH树类，普通类型
     * 根据图元列表在局部空间构造线性BVH树，存储在BLASNode数组中
     * BVH树为二叉树，但是叶子节点可以有多个图元
     */
    class BLAS {
    public:
        //叶子节点包含的图元数
        static constexpr size_t PRIMITIVE_COUNT_PER_LEAF_NODE = 4;

        //BVH树节点
        typedef struct BLASNode {
            /*
             * 当前节点的包围盒
             * 直接包含PRIMITIVE_COUNT_PER_LEAF_NODE个图元或者两个子包围盒
             */
            BoundingBox boundingBox {};

            /*
             * primitiveCount用于区分节点类型：如果primitiveCount大于0，则为叶子节点
             * 如果为叶子节点，则index为图元索引数组的起始下标
             * 如果为中间节点，则index为左子节点的下标
             */
            size_t primitiveCount = 0;
            size_t index = 0;
        } BLASNode;

        typedef Pair<std::vector<BLASNode>, std::vector<Pair<PrimitiveType, size_t>>> BLASBuildResult;

        /*
         * 使用物体列表构造BVH节点数组
         * 返回值包含线性化的BVH树以及用于索引的索引信息数组
         *   索引信息数组存储指向图元的索引，PrimitiveType为图元类型，size_t为图元在原始数组中的下标（参数spheres等）
         *   索引信息数组的长度为图元总数一个图元对应一个Pair<PrimitiveType, size_t>>，此数组只为叶子节点服务
         *   若为叶子节点，BLASNode的index成员存储了当前节点对应的几个图元在索引数组中的下标范围，用于定位图元
         *
         * 构造出来的BVH树将包含传入的所有图元，则仅将需要作为一个实例的一个或一组图元传入
         */
        static Pair<std::vector<BLASNode>, std::vector<Pair<PrimitiveType, size_t>>> constructBLAS(
                const Sphere * spheres, size_t sphereStartIndex, size_t sphereCount,
                const Parallelogram * parallelograms, size_t parallelogramStartIndex, size_t parallelogramCount,
                const Triangle * triangles, size_t triangleStartIndex, size_t triangleCount);

        /*
         * BLAS相交测试，由GPU线程执行
         */
        __device__ static bool hit(
                const BLASNode * __restrict__ treeArray, const Pair<PrimitiveType, size_t> * __restrict__ indexArray,
                const Ray * ray, const Range * range, HitRecord * record,
                const Sphere * __restrict__ spheres, const Parallelogram * __restrict__ parallelograms,
                const Triangle * __restrict__ triangles);

    private:
        //构建过程的任务结构体
        typedef struct BuildingTask {
            //使用起始下标和元素个数表示当前任务包含的图元列表，当只有一个图元时成为叶子节点
            //操作索引而不是对象，不需要拷贝任务对象
            size_t primitiveStartIndex;
            size_t primitiveCount;

            //该任务对应的节点在线性数组中的位置
            size_t nodeIndex;
        } BuildingTask;

        //构造过程中统一数据表示形式
        typedef struct PrimitiveInfo {
            //图元的包围盒和重心，决定如何分割图元列表
            BoundingBox boundingBox {};
            Point3 centroid {};

            //图元类型
            PrimitiveType type {};
            //在原始数组中的引用
            size_t index {};
        } PrimitiveInfo;

        //为图元列表构造包围盒，结果为所有图元包围盒的合并
        static BoundingBox constructBoundingBoxForPrimitiveList(
                const std::vector<PrimitiveInfo> & primitives, size_t startIndex, size_t endIndex);
    };

    typedef BLAS::BLASNode BLASNode;
    typedef Pair<PrimitiveType, size_t> BLASIndex;
    typedef BLAS::BLASBuildResult BLASBuildResult;

    typedef Pair<BLASNode *, size_t> BLASNodeArray;
    typedef Pair<BLASIndex *, size_t> BLASIndexArray;
    typedef Pair<BLASNodeArray, BLASIndexArray> BLASArray;
}

#endif //RENDERERINTERACTIVE_BLAS_CUH
