#ifndef RENDERERINTERACTIVE_TLAS_CUH
#define RENDERERINTERACTIVE_TLAS_CUH

#include <AS/Instance.cuh>

namespace project {
    /*
     * 顶层加速结构BVH树类，普通类型
     * 根据底层加速结构数组构造顶层加速结构
     *
     * 底层加速结构：
     *   构造数据：一组图元
     *   构造结果：包含这组图元的BVH树以及对应的索引数组，索引数组包含每个叶子节点对原始图元的索引信息
     *
     * 顶层加速结构：
     *   构造数据：一个BLAS列表和一个实例列表
     *   构造结构：包含实例列表中所有实例的BVH树以及索引数组，索引数组包含TLAS每个叶子节点对原始实例的索引
     */
    class TLAS {
    public:
        //叶子节点包含的实例数
        static constexpr size_t INSTANCE_COUNT_PER_LEAF_NODE = 4;

        typedef struct TLASNode {
            /*
            * 当前节点的包围盒
            * 直接包含INSTANCE_COUNT_PER_LEAF_NODE个实例或者两个子包围盒
            */
            BoundingBox boundingBox {};

            /*
             * 如果instanceCount大于0，则为叶子节点
             * 如果为叶子节点，则index为索引数组的起始下标
             *   通过此起始下标+偏移量获取索引数组的元素，通过索引数组中元素找到原始实例
             * 如果为中间节点，则index为左子节点的下标
             */
            size_t instanceCount = 0;
            size_t index = 0;
        } TLASNode;

        typedef Pair<std::vector<TLASNode>, std::vector<size_t>> TLASBuildResult;

        /*
         * 使用BLAS列表和实例列表构造TLAS
         * 一个TLASNode数组对应一个TLASIndex数组，但是二者的长度不同
         */
        static TLASBuildResult constructTLAS(const Instance * instanceArray, size_t instanceCount);

        /*
         * TLAS遍历，由TLAS遍历函数调用BLAS遍历函数
         * 此函数托管了BLAS的hit函数，rayColor中将直接调用此函数而不是BLAS::hit
         */
        __device__ static bool hit(
                const TLASNode * __restrict__ treeArray, const size_t * __restrict__ indexArray,
                const Instance * __restrict__ instances, const BLASArray * __restrict__ blasArray,
                const Ray * ray, const Range * range, HitRecord * record,
                const Sphere * __restrict__ spheres, const Parallelogram * __restrict__ parallelograms,
                const Triangle * __restrict__ triangles);

    private:
        //构建过程的任务结构体
        typedef struct BuildingTask {
            //使用起始下标和元素个数表示当前任务包含的实例列表，当只有少数实例时成为叶子节点
            size_t instanceStartIndex;
            size_t instanceCount;

            //该任务对应的节点在线性数组中的位置
            size_t nodeIndex;
        } BuildingTask;

        //将实例和其在实例数组中的原始索引打包在一起
        typedef struct InstanceAndIndex {
            const Instance * instance;
            size_t originalIndex;
        } InstanceAndIndex;

        //为多个实例构建合并包围盒（世界空间）
        static BoundingBox constructBoundingBoxForInstanceList(
                const std::vector<InstanceAndIndex> & instanceAndIndexArray, size_t startIndex, size_t endIndex);
    };

    typedef TLAS::TLASNode TLASNode;
    typedef size_t TLASIndex;
    typedef TLAS::TLASBuildResult TLASBuildResult;

    //将数组和其长度打包
    typedef Pair<TLASNode *, size_t> TLASNodeArray;
    typedef Pair<TLASIndex *, size_t> TLASIndexArray;
    //打包为一个Pair
    typedef Pair<TLASNodeArray, TLASIndexArray> TLASArray;
}

#endif //RENDERERINTERACTIVE_TLAS_CUH
