#ifndef RENDERERINTERACTIVE_RENDERER_CUH
#define RENDERERINTERACTIVE_RENDERER_CUH

#include <Global/RendererImpl.cuh>

namespace project {
    /*
     * 渲染器对外接口，封装RendererImpl
     *   不直接调用RendererImpl中所有结构体和函数
     *
     * 提交几何体信息（位于普通内存）
     *   分配页面锁定内存--拷贝到页面锁定内存--在页面锁定内存中构建加速结构--拷贝几何体和加速结构到全局内存
     *
     * 提交材质信息
     *   同提交几何体信息
     *
     * 提交实例信息
     *   传递实例更新函数指针
     *
     * 构建加速结构
     *   分配页面锁定内存并构建BLAS，分配全局内存并拷贝BLAS到设备
     *
     * 提交相机初始信息
     *
     * 启动渲染
     *   准备第一帧数据--启动渲染循环
     *   封装按键接收逻辑，只需选择图形API
     *
     * 清理资源
     */

    //几何体信息
    typedef struct GeometryData {
        std::vector<Sphere> spheres;
        std::vector<Parallelogram> parallelograms;
        std::vector<Triangle> triangles;
    } GeometryData;

    //材质信息
    typedef struct MaterialData {
        std::vector<Rough> roughs;
        std::vector<Metal> metals;
    } MaterialData;

    //窗口初始信息
    typedef struct WindowInput {
        Uint32 windowWidth;
        Uint32 windowHeight;

        std::string title;
    } WindowInput;

    //相机初始信息
    typedef struct CameraInput {
        Color3 backgroundColor;
        Point3 initialCenter;
        Point3 initialTarget;
        float fov;
        Vec3 upDirection;
        float focusDiskRadius;
        float sampleRange;
        size_t sampleCount;
        size_t rayTraceDepth;
    } CameraInput;

    //Impl结构体封装，包含指向页面锁定内存的和指向全局内存的
    //几何体
    typedef struct SceneGeometryDataPackage {
        SceneGeometryData pin_sceneGeometryData;
        SceneGeometryData dev_sceneGeometryData;
    } SceneGeometryDataPackage;

    //材质
    typedef struct SceneMaterialDataPackage {
        SceneMaterialData pin_sceneMaterialData;
        SceneMaterialData dev_sceneMaterialData;
    } SceneMaterialDataPackage;

    //加速结构
    typedef struct SceneAccelerationStructurePackage {
        Pair<BLASArray *, size_t> pin_blasArray;
        Pair<BLASArray *, size_t> dev_blasArray;

        std::array<TLASArray, 2> pin_tlasArrays;
        std::array<TLASArray, 2> dev_tlasArrays;
    } SceneAccelerationStructurePackage;

    //实例
    typedef struct SceneInstanceDataPackage {
        std::array<Instance *, 2> pin_instanceArrays;
        std::array<Instance *, 2> dev_instanceArrays;
        size_t instanceCount;

        void (*updateInstances)(Instance * pin_instances, size_t instanceCount, size_t frameCount);
    } SceneInstanceDataPackage;

    //VTK
    typedef struct SceneVTKDataPackage {
        //.series文件路径
        std::string seriesFilePath;

        //每个vtk文件和其对应时间，数组长度为vtk文件个数
        std::vector<Pair<std::string, float>> vtkFileAndTimeArray;

        //当前加载的VTK文件的三角形总数，用于调整手动定义的三角形实例索引
        size_t vtkTriangleCount;
    } SceneVTKDataPackage;

    class Renderer {
    public:
        //读取.series文件信息，通过.series文件预读取所有参与动画的vtk文件，必须在提交其他信息前调用
        //当前只读取.series中的第一个VTK文件
        static SceneVTKDataPackage configureVTKFiles(const std::string & seriesFilePath);

        //提交几何体信息
        static SceneGeometryDataPackage commitGeometryData(GeometryData & data, SceneVTKDataPackage * vtkData = nullptr);

        //提交材质信息
        static SceneMaterialDataPackage commitMaterialData(MaterialData & data, SceneVTKDataPackage * vtkData = nullptr);

        //设置实例数量和实例更新函数，每个实例的更新由实例更新函数处理
        //当前将VTK粒子的三角形图元插入到已有图元列表头部，使得所有三角形实例的图元索引在此函数需要被偏移
        static SceneInstanceDataPackage configureInstances(
            std::vector<Pair<PrimitiveType, size_t>> & insMapArray,
            void (*updateInstances)(Instance * pin_instances, size_t instanceCount, size_t frameCount),
            const SceneVTKDataPackage * vtkData = nullptr);

        //构建加速结构
        static SceneAccelerationStructurePackage buildAccelerationStructure(
                const SceneGeometryDataPackage & geoData, SceneInstanceDataPackage & insData);

        //提交相机基础信息，返回计算完成的，存储于页面锁定内存中的相机结构体，直接传入渲染函数
        static Camera * configureCamera(const WindowInput & windowData, const CameraInput & cameraData);

        //启动渲染，渲染函数实时修改Camera参数
        static void startRender(
                const WindowInput & windowData,
                const SceneGeometryDataPackage & geoData, const SceneMaterialDataPackage & matData,
                SceneInstanceDataPackage & insData, SceneAccelerationStructurePackage & asData,
                Camera * pin_camera);

        //清理资源，不管理输入信息中的资源
        static void cleanup(SceneGeometryDataPackage & geoData, SceneMaterialDataPackage & matData,
                     SceneAccelerationStructurePackage & asData,
                     SceneInstanceDataPackage & insData, Camera * & pin_camera);
    };
}

#endif //RENDERERINTERACTIVE_RENDERER_CUH
