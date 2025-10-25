#ifndef RENDERERINTERACTIVE_RENDERERIMPL_CUH
#define RENDERERINTERACTIVE_RENDERERIMPL_CUH

#include <AS/TLAS.cuh>

#include <Material/Rough.cuh>
#include <Material/Metal.cuh>

namespace project {
    //几何体数据
    typedef struct SceneGeometryData {
        Sphere * spheres;
        size_t sphereCount;

        Parallelogram * parallelograms;
        size_t parallelogramCount;

        Triangle * triangles;
        size_t triangleCount;
    } SceneGeometryData;

    //材质数据
    typedef struct SceneMaterialData {
        Rough * roughs;
        size_t roughCount;

        Metal * metals;
        size_t metalCount;
    } SceneMaterialData;

    //相机数据
    typedef struct Camera {
        //窗口尺寸，简化访问
        int windowWidth;
        int windowHeight;

        //相机
        Color3 backgroundColor;
        Point3 cameraCenter;
        Point3 cameraTarget;
        float fov;

        //视口
        Vec3 upDirection;
        float viewPortWidth;
        float viewPortHeight;
        Vec3 cameraU, cameraV, cameraW;
        Vec3 viewPortX, viewPortY;
        Vec3 viewPortPixelDx, viewPortPixelDy;
        Point3 viewPortOrigin;
        Point3 pixelOrigin;

        //采样
        float focusDiskRadius;
        float focusDistance;
        float sampleRange;
        size_t sampleCount;
        size_t sqrtSampleCount;
        float reciprocalSqrtSampleCount;
        size_t rayTraceDepth;
    } Camera;

    //遍历参数，在光线遍历期间，所有资源均通过单一的只读指针访问，以启用编译器优化
    typedef struct TraverseData {
        //无需保留数组长度信息
        const Sphere * const __restrict__ dev_spheres;
        const Parallelogram * const __restrict__ dev_parallelograms;
        const Triangle * const __restrict__ dev_triangles;

        const Rough * const __restrict__ dev_roughs;
        const Metal * const __restrict__ dev_metals;

        const Instance * const __restrict__ dev_instances;

        //加速结构
        const TLASNode * const __restrict__ tlasNodeArray;
        const TLASIndex * const __restrict__ tlasIndexArray;

        //为了不分配额外的全局内存用于存放指针数组，直接使用原始结果
        const BLASArray * const __restrict__ blasArray;
    } TraverseData;

    /*
     * 初始化：
     * 分配存放材质，几何体，相机，实例和BLAS的页面锁定内存
     * 在页面锁定内存中初始化材质和几何体
     *
     * 在页面锁定内存中使用几何体信息构建BLAS
     * 分配材质，几何体和BLAS的全局内存
     * 使用默认流将材质，几何体和底层加速结构信息拷贝到全局内存
     * 初始化拷贝流和计算流
     *
     * 更新：
     * 主机：
     *   在页面锁定内存中更新实例变换信息
     *   在页面锁定内存中根据实例信息分配并构建TLAS
     *
     * 拷贝流：
     *   拷贝实例和当前TLAS数据到全局内存，拷贝相机到常量内存
     * 等待拷贝流完成
     *
     * 计算流：
     *   启动渲染核函数
     * 等待计算流完成
     * 显示结果
     * 更新当前TLAS下标
     *
     * 清理：
     * 销毁流
     * 释放全局内存
     * 释放页面锁定内存
     * 释放其他资源
     */
    class RendererImpl {
    public:
        // ====== 几何体和材质 ======
        //分配页面锁定内存时，传入的结构体需要设置每种几何体或材质个数，函数设置结构体的指针指向有效内存地址

        //几何体
        static void allocGeoPinMem(SceneGeometryData & geometryDataWithPinPtr);
        static void freeGeoPinMem(SceneGeometryData & geometryDataWithPinPtr);

        static SceneGeometryData allocGeoGlobMem(cudaStream_t stream, const SceneGeometryData & geometryDataWithPinPtr);
        static void copyGeoToGlobMem(cudaStream_t stream, SceneGeometryData & geometryDataWithDevPtr, const SceneGeometryData & geometryDataWithPinPtr);
        static void freeGeoGlobMem(cudaStream_t stream, SceneGeometryData & geometryDataWithDevPtr);

        //材质
        static void allocMatPinMem(SceneMaterialData & materialDataWithPinPtr);
        static void freeMatPinMem(SceneMaterialData & materialDataWithPinPtr);

        static SceneMaterialData allocMatGlobMem(cudaStream_t stream, const SceneMaterialData & materialDataWithPinPtr);
        static void copyMatToGlobMem(cudaStream_t stream, SceneMaterialData & materialDataWithDevPtr, const SceneMaterialData & materialDataWithPinPtr);
        static void freeMatGlobMem(cudaStream_t stream, SceneMaterialData & materialDataWithDevPtr);

        // ====== 实例 ======

        //根据实例数量分配实例页面锁定内存
        static Instance * allocInstPinMem(size_t instanceCount);
        static void freeInstPinMem(Instance * & pin_instances);

        //根据实例数量分配实例全局内存
        static Instance * allocInstGlobMem(cudaStream_t stream, size_t instanceCount);
        static void freeInstGlobMem(cudaStream_t stream, Instance * & dev_instances);

        //将实例拷贝到全局内存
        static void copyInstToGlobMem(cudaStream_t stream, Instance * dev_instances, const Instance * pin_instances, size_t instanceCount);

        // ====== 相机 ======

        //分配相机页面锁定内存
        static Camera * allocCamPinMem();
        static void freeCamPinMem(Camera * & pin_camera);

        //根据相机基本参数计算相机全部参数
        static void calculateCameraProperties(Camera * pin_camera);

        //将相机拷贝到常量内存
        static void copyCamToConstMem(cudaStream_t stream, const Camera * pin_camera);

        // ====== 加速结构 ======

        /*
         * 在页面锁定内存中构建加速结构
         * 函数在普通内存中构建完加速结构后，根据加速结构的实际大小分配页面锁定内存
         *
         * 输入：物体列表和实例列表（存储于页面锁定内存中）
         * 输出：一个TLAS和一组BLAS
         *   TLAS的BVH树中每个叶子节点存储多个实例，每个实例拥有一个BLAS索引
         *   函数同时修改每个实例对象的asIndex成员
         * 返回的结构体包含指向存储了加速结构的页面锁定内存的指针
         */
        static Pair<BLASArray *, size_t> buildBLASPinMem(const SceneGeometryData & geometryDataWithPinPtr, Instance * pin_instances, size_t instanceCount);
        static void freeBLASPinMem(Pair<BLASArray *, size_t> & blasWithPinPtr);

        static TLASArray buildTLASPinMem(Instance * pin_instances, size_t instanceCount);
        static void freeTLASPinMem(TLASArray & tlasWithPinPtr);

        //将主机端加速结构拷贝到全局内存，返回包含指向加速结构全局内存的指针
        //使用CUDA 11.2 引入的流有序内存分配器（Stream-Ordered Memory Allocator），使得全局内存分配和释放操作非阻塞
        static Pair<BLASArray *, size_t> copyBLASToGlobMem(cudaStream_t stream, const Pair<BLASArray *, size_t> & pin_blas);
        static void freeBLASGlobMem(cudaStream_t stream, Pair<BLASArray *, size_t> & blasWithDevPtr);

        static TLASArray copyTLASGlobMem(cudaStream_t stream, const TLASArray & pin_tlas);
        static void freeTLASGlobMem(cudaStream_t stream, TLASArray & tlasWithDevPtr);

        // ====== 渲染 ======

        //将分配好的设备指针转换为只读指针
        static TraverseData traverseDevPtr(
                const SceneGeometryData & geometryDataWithDevPtr, const SceneMaterialData & materialDataWithDevPtr,
                const Instance * dev_instances,
                const Pair<BLASArray *, size_t> & blasWithDevPtr, const TLASArray & tlasWithDevPtr);
    };

    //常量内存声明
    extern __constant__ Camera dev_camera[1];

    //核函数声明
    extern __global__ void render(const TraverseData * dev_traverseData, cudaSurfaceObject_t surfaceObject);
}

#endif //RENDERERINTERACTIVE_RENDERERIMPL_CUH
