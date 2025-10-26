#include <Global/Renderer.cuh>
using namespace project;

namespace {
    //更新每个实例的变换信息
    void updateInstance(Instance * pin_instances, size_t instanceCount, size_t frameCount) {
        const static Point3 initialCenter = {0.0, 2.0, 0.0};
        const float radius = 2.0f;
        const float speed = 0.02f;
        const float angle = static_cast<float>(frameCount) * speed;
        const float3 newCenter = {
                initialCenter[0] + radius * std::cos(angle) * 1.5f,
                initialCenter[1] + radius * std::sin(angle) * std::cos(angle),
                initialCenter[2] + radius * std::sin(angle) * 1.5f
        };
        const float3 newCenter2 = {-newCenter.x, newCenter.y,-newCenter.z};

        pin_instances[0].updateTransformArguments(
                {0.0, -1000.0, 0.0},
                {0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0});
        pin_instances[1].updateTransformArguments(
                newCenter,
                {0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0});
        pin_instances[2].updateTransformArguments(
                {-5.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0});
        pin_instances[3].updateTransformArguments(
                newCenter2,
                {
                        static_cast<float>(frameCount) * 0.4f,
                        static_cast<float>(frameCount) * 0.4f,
                        static_cast<float>(frameCount) * 0.4f},
                {3.0, 3.0, 3.0});
    }
}

#undef main
int main(int args, char * argv[]) {
    //几何体
    const std::vector<Sphere> spheres = {
            {MaterialType::ROUGH, 3, {0.0, 0, 0.0}, 1000.0},
            {MaterialType::ROUGH, 0, {0.0, 0, 0.0}, 2.0},
    };
    const std::vector<Parallelogram> parallelograms = {
            {MaterialType::ROUGH, 1, {0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 4.0, 0.0}},
    };
    const std::vector<Triangle> triangles {
            {MaterialType::ROUGH, 2, std::array<Point3, 3>{ Point3{0.0, 0.0, 0.0}, Point3{1.0, 0.0, 1.0}, Point3{0.0, 1.0, 0.0} }},
    };
    GeometryData geoData = {
            .spheres = spheres,
            .parallelograms = parallelograms,
            .triangles = triangles
    };

    //材质
    const std::vector<Rough> roughs = {
            {.65, .05, .05},
            {.73, .73, .73},
            {.12, .45, .15},
            {.70, .60, .50}
    };
    const std::vector<Metal> metals = {
            {0.8, 0.85, 0.88, 0.0}
    };
    MaterialData matData = {
            .roughs = roughs,
            .metals = metals
    };

    //相机
    const CameraInput camData = {
            .backgroundColor = {0.7, 0.8, 0.9},
            .initialCenter = {0.0, 2.0, 10.0},
            .initialTarget = {0.0, 2.0, 0.0},
            .fov = 90,
            .upDirection = {0.0, 1.0, 0.0},
            .focusDiskRadius = 0.0,
            .sampleRange = 0.5,
            .sampleCount = 1,
            .rayTraceDepth = 10,
    };

    //实例映射
    std::vector<Pair<PrimitiveType, size_t>> insMapArray = {
            {PrimitiveType::SPHERE, 0},
            {PrimitiveType::SPHERE, 1},
            {PrimitiveType::PARALLELOGRAM, 0},
            {PrimitiveType::TRIANGLE, 0},
    };

    //窗口
    const WindowInput windowData = {
            .windowWidth = 1200,
            .windowHeight = 800,
            .title = std::string("Test")
    };

    //初始化资源
    auto vtk = Renderer::configureVTKFiles("../files/particle_mesh.vtk.series");
    auto geo = Renderer::commitGeometryData(geoData, &vtk);
    auto mat = Renderer::commitMaterialData(matData, &vtk);
    auto ins = Renderer::configureInstances(insMapArray, updateInstance, &vtk);
    auto as = Renderer::buildAccelerationStructure(geo, ins);
    auto cam = Renderer::configureCamera(windowData, camData);

    //启动渲染
    Renderer::startRender(windowData, geo, mat, ins, as, cam);

    //清理资源
    Renderer::cleanup(geo, mat, as, ins, cam);
}