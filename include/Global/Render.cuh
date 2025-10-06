#ifndef RENDERERINTERACTIVE_RENDER_CUH
#define RENDERERINTERACTIVE_RENDER_CUH

#include <AS/TLAS.cuh>

#include <Material/Rough.cuh>
#include <Material/Metal.cuh>

namespace renderer {
    //几何体数据
    typedef struct SceneGeometryData {
        Sphere * spheres;
        size_t sphereCount;

        Parallelogram * parallelograms;
        size_t parallelogramCount;
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

    //加速结构构建结果，主机端使用，作为上传到显存的函数的参数
    //此结构体直接存储加速结构数据
    typedef struct ASBuildResult {
        //加速结构构建结果包含一个TLAS和一组BLAS
        TLASBuildResult tlas;
        std::vector<BLASBuildResult> blasVector;
    } ASBuildResult;

    //加速结构遍历参数，此结构体保存指向加速结构的引用，在遍历前应当指向设备内存
    typedef struct ASTraverseData {
        //实例数组，作为直接被遍历的对象
        Instance * instances;
        size_t instanceCount;

        //一个TLASNode数组对应一个TLASIndex数组，但是二者的长度不同
        //在构建时，数组和其长度合成为一个std::vector，在分配显存和遍历时数组展开为一个指针和一个size_t
        TLASArray tlasArray;

        BLASArray * blasArray;
        size_t blasArrayCount; //BLAS的数量，即blasNodeArrayArray的长度
    } ASTraverseData;

    /*
     * 初始化：
     * 分配存放材质，几何体数据和两层加速结构的页面锁定内存
     * 在页面锁定内存中初始化材质，几何体，相机
     * 构建底层加速结构，计算几何体变换矩阵，构建顶层加速结构
     * 分配全局内存，初始化流
     * 拷贝材质，几何体和两层加速结构数据到全局内存
     * 拷贝相机到常量内存
     * 启动渲染核函数
     * 显示结果
     *
     * 更新：
     * CPU：更新几何体，相机数据
     * CPU：基于新的几何体数据计算新的变换矩阵，更新顶层加速结构
     * CUDA流1：拷贝相机和新的顶层加速结构到全局内存
     * CUDA流2：启动渲染核函数并显示结果
     * 流1和流2异步执行
     *
     * 清理：
     * 释放全局内存
     * 释放页面锁定内存
     * 释放其他资源
     */
    class Renderer {
    public:
        /*
         * 分配页面锁定内存：传入的结构体需要设置每种几何体个数，函数将结构体的指针指向有效内存地址
         * 两层加速结构的内存大小：由于BVH树的一个叶子节点可以存放多个实例或图元，则节点总数小于2N - 1
         * 设有a个物体，一个物体有b个BLAS节点，则分配2a - 1个TLAS节点，a * (2b - 1)个BLAS节点
         */
        static void mallocPinnedMemory(SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & pin_camera, Instance * & pin_instances, size_t instanceCount);

        //释放页面锁定内存
        static void freePinnedMemory(SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & pin_camera, Instance * & pin_instances);

        //计算相机参数
        static void constructCamera(Camera * pin_camera);

        //更新相机位置
        static void updateCameraProperties(Camera * pin_camera, const Point3 & newCenter, const Point3 & newTarget);

        /*
         * 分配全局内存，将场景数据拷贝到全局内存，将相机拷贝到常量内存
         * 传入初始化后的场景数据结构体，结构体内指针指向有效的页面锁定内存
         * 返回新的场景数据结构体，结构体内指针指向全局内存
         */
        static Pair<SceneGeometryData, SceneMaterialData> copyToGlobalMemory(
                const SceneGeometryData & geometryData, const SceneMaterialData & materialData, const Camera * pin_camera);

        //释放全局内存
        static void freeGlobalMemory(
                SceneGeometryData & geometryDataWithDevPtr, SceneMaterialData & materialDataWithDevPtr);

        /*
         * 构建加速结构
         * 输入：物体列表和实例列表
         * 输出：一个TLAS和一组BLAS
         *   TLAS的BVH树中每个叶子节点存储多个实例，每个实例拥有一个BLAS索引
         *   函数同时修改每个实例对象的asIndex成员
         */
        static ASBuildResult buildAccelerationStructure(const SceneGeometryData & geometryDataWithPinPtr, Instance * pin_instances, size_t instanceCount);

        /*
         * 将主机端加速结构拷贝到全局内存，返回包含有效全局内存的遍历结构体
         */
        static ASTraverseData copyAccelerationStructureToGlobalMemory(
                const ASBuildResult & asBuildResult, const Instance * pin_instances, size_t instanceCount);

        //释放加速结构全局内存
        static void freeAccelerationStructureGlobalMemory(ASTraverseData & asTraverseData);

        //渲染循环
        static void renderLoop(
                const SceneGeometryData & geometryDataWithDevPtr, const SceneMaterialData & materialDataWithDevPtr,
                Camera * pin_camera, const ASTraverseData & asTraverseData);
    };

    extern __constant__ Camera dev_camera[1];
}

#endif //RENDERERINTERACTIVE_RENDER_CUH
