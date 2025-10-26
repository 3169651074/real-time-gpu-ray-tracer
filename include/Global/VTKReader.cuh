#ifndef RENDERERINTERACTIVE_VTKREADER_CUH
#define RENDERERINTERACTIVE_VTKREADER_CUH

#include <Global/Renderer.cuh>

namespace project {
    /*
     * VTK读取器：
     */

    //VTK粒子信息
    typedef struct VTKParticle {
        //粒子ID
        size_t id;

        //速度
        Vec3 velocity;

        //包围盒范围
        std::array<Range, 3> bounds;

        //质心
        Point3 centroid;

        //组成此粒子的所有三角形顶点，保留triangle strip的格式
        std::vector<Point3> vertices;

        //每个顶点对应的法向量，和顶点一一对应
        std::vector<Vec3> verticeNormals;
    } VTKParticle;

    class VTKReader {
    public:
        //读取单个VTK文件所有粒子
        static std::vector<VTKParticle> readVTKFile(const std::string & filePath);

        //将VTK粒子原始信息转换为三角形数组和实例数组
        static Pair<std::vector<Triangle>, std::vector<Instance>> convertToRendererData(const std::vector<VTKParticle> & particles);
    };
}

//    /*
//     * VTK读取器，当需要读取新的VTK文件时，此读取器读取文件信息并更新场景数据
//     * 1.使用VTK库读取所需粒子信息
//     * 2.将粒子信息转换为三角形数组和实例数组
//     * 3.将新的几何体信息和实例信息添加到已有场景中
//     * 4.启动渲染
//     * 5.在更新VTK文件时，释放上一个VTK文件对应的几何体、实例内存
//     */
//    /*
//     * VTK场景引用信息，标记那些场景信息由VTK粒子管理，这些信息在更新VTK文件时需要被释放
//     * 在场景信息列表中，一次添加的VTK粒子信息连续排列
//     *
//     * TLAS每帧重建，且只有一个，在删除实例后自动改变
//     * BLAS由实例对象的asIndex进行索引，删除VTK粒子组后，从BLAS数组中删除对应BLAS并重新整理BLAS
//     */
//    typedef struct SceneVTKReferenceData {
//        //粒子实例在实例数组中的起始下标和数量
//        size_t instanceStartIndex;
//        size_t instanceCount;
//
//        //所有粒子的所有三角形在三角形数组中的索引范围
//        size_t triangleStartIndex;
//        size_t triangleCount;
//
//        //为渲染粒子所新创建的材质在粗糙材质数组中的索引范围
//        //在总粒子数量较多时，一个文件可以包含多层粒子，需要为每一层粒子分配不同粗糙材质
//        size_t roughStartIndex;
//        size_t roughCount;
//    } SceneVTKReferenceData;
//将读取到的单个文件的所有粒子几何和实例附加到已有几何体列表和实例列表中，并更新加速结构
//此函数将原来已经准备好渲染的数据，经过添加后再次准备好渲染
//void addVTKToScene(
//        const std::vector<VTKParticle> & particles,
//        SceneGeometryDataPackage & geoData, SceneMaterialDataPackage & matData,
//        SceneInstanceDataPackage & insData, SceneAccelerationStructurePackage & asData);

#endif //RENDERERINTERACTIVE_VTKREADER_CUH
