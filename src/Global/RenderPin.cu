#include <Global/Render.cuh>

namespace renderer {
#define _mallocHost(type, className, arrayName, countName)       \
    do {                                                         \
        SDL_Log(#className" count: %zd", type##DataWithPinPtr.countName);\
        if (type##DataWithPinPtr.countName != 0) {               \
            cudaCheckError(cudaHostAlloc(&type##DataWithPinPtr.arrayName, type##DataWithPinPtr.countName * sizeof(className), cudaHostAllocDefault));\
        }                                                        \
    } while (false)
#define _mallocGeoHost(className, arrayName, countName) _mallocHost(geometry, className, arrayName, countName)
#define _mallocMatHost(className, arrayName, countName) _mallocHost(material, className, arrayName, countName)

#define _freeHost(type, arrayName)     \
    do {                               \
        cudaCheckError(cudaFreeHost(type##DataWithPinPtr.arrayName));\
        type##DataWithPinPtr.arrayName = nullptr;\
    } while (false)
#define _freeGeoHost(arrayName) _freeHost(geometry, arrayName)
#define _freeMatHost(arrayName) _freeHost(material, arrayName)

    // ====== 几何体 ======

    void Renderer::allocGeoPinMem(SceneGeometryData & geometryDataWithPinPtr) {
        SDL_Log("Allocating pinned memory for geometry data...");

        _mallocGeoHost(Sphere, spheres, sphereCount);
        _mallocGeoHost(Parallelogram, parallelograms, parallelogramCount);
    }
    void Renderer::freeGeoPinMem(SceneGeometryData & geometryDataWithPinPtr) {
        SDL_Log("Freeing pinned memory for geometry data...");

        _freeGeoHost(spheres);
        _freeGeoHost(parallelograms);

        geometryDataWithPinPtr = {};
    }

    // ====== 材质 ======

    void Renderer::allocMatPinMem(SceneMaterialData & materialDataWithPinPtr) {
        SDL_Log("Allocating pinned memory for material data...");

        _mallocMatHost(Rough, roughs, roughCount);
        _mallocMatHost(Metal, metals, metalCount);
    }
    void Renderer::freeMatPinMem(SceneMaterialData & materialDataWithPinPtr) {
        SDL_Log("Freeing pinned memory for material data...");

        _freeMatHost(roughs);
        _freeMatHost(metals);

        materialDataWithPinPtr = {};
    }

    // ====== 实例 ======

    Instance * Renderer::allocInstPinMem(size_t instanceCount) {
        SDL_Log("Allocating pinned memory for instance...");
        Instance * ret;
        cudaCheckError(cudaHostAlloc(&ret, instanceCount * sizeof(Instance), cudaHostAllocDefault));
        return ret;
    }
    void Renderer::freeInstPinMem(Instance * & pin_instances) {
        SDL_Log("Freeing pinned memory for instance...");
        cudaCheckError(cudaFreeHost(pin_instances));
        pin_instances = nullptr;
    }

    // ====== 相机 ======

    Camera * Renderer::allocCamPinMem() {
        SDL_Log("Allocating pinned memory for camera...");
        Camera * ret;
        cudaCheckError(cudaHostAlloc(&ret, sizeof(Camera), cudaHostAllocDefault));
        return ret;
    }
    void Renderer::freeCamPinMem(Camera * & pin_camera) {
        SDL_Log("Freeing pinned memory for camera...");
        cudaCheckError(cudaFreeHost(pin_camera));
        pin_camera = {};
    }

    void Renderer::calculateCameraProperties(Camera * pin_camera) {
        Camera & cam = *pin_camera;

        cam.focusDistance = cam.cameraCenter.distance(cam.cameraTarget);
        const float thetaFOV = MathHelper::degreeToRadian(cam.fov);
        const float vWidth = 2.0f * std::tan(thetaFOV / 2.0f) * cam.focusDistance;
        const float vHeight = vWidth / (static_cast<float>(cam.windowWidth) * 1.0f / static_cast<float>(cam.windowHeight));

        cam.viewPortWidth = vWidth;
        cam.viewPortHeight = vHeight;
        cam.cameraW = Point3::constructVector(cam.cameraCenter, cam.cameraTarget).unitVector();
        cam.cameraU = Vec3::cross(cam.cameraW, cam.upDirection).unitVector();
        cam.cameraV = Vec3::cross(cam.cameraU, cam.cameraW).unitVector();

        cam.viewPortX = vWidth * cam.cameraU;
        cam.viewPortY = vHeight * cam.cameraV;
        cam.viewPortPixelDx = cam.viewPortX / static_cast<float>(cam.windowWidth);
        cam.viewPortPixelDy = cam.viewPortY / static_cast<float>(cam.windowHeight);
        cam.viewPortOrigin = cam.cameraCenter + cam.focusDistance * cam.cameraW - cam.viewPortX * 0.5f - cam.viewPortY * 0.5f;
        cam.pixelOrigin = cam.viewPortOrigin + cam.viewPortPixelDx * 0.5f + cam.viewPortPixelDy * 0.5f;
        cam.sqrtSampleCount = static_cast<size_t>(std::sqrt(cam.sampleCount));
        cam.reciprocalSqrtSampleCount = 1.0f / static_cast<float>(cam.sqrtSampleCount);
    }

    // ====== BLAS ======

    Pair<BLASArray *, size_t> Renderer::buildBLASPinMem(
            const SceneGeometryData & geometryDataWithPinPtr, Instance * pin_instances, size_t instanceCount)
    {
        SDL_Log("Building BLAS...");

        //为每个物体创建一个的BLAS，存储到数组中，当前一个实例对应一个BLAS
        std::vector<BLASBuildResult> blasBuildResultVector;
        blasBuildResultVector.reserve(instanceCount);

        for (size_t i = 0; i < instanceCount; i++) {
            Instance & instance = pin_instances[i];

            //设置实例对应的BLAS下标
            instance.asIndex = blasBuildResultVector.size();

            switch (instance.primitiveType) {
                //构建时设置实例的变换前包围盒和几何中心
                //由于constructBLAS函数传入参数的位置不同，不使用宏进行展开
                case PrimitiveType::SPHERE:
                    //instance.setBoundingBoxProperties(geometryDataWithPinPtr.spheres[instance.primitiveIndex].constructBoundingBox(), geometryDataWithPinPtr.spheres[instance.primitiveIndex].centroid());
                    blasBuildResultVector.push_back(BLAS::constructBLAS(
                            geometryDataWithPinPtr.spheres, instance.primitiveIndex, 1,
                            geometryDataWithPinPtr.parallelograms, 0, 0));
                    break;
                case PrimitiveType::PARALLELOGRAM:
                    //instance.setBoundingBoxProperties(geometryDataWithPinPtr.parallelograms[instance.primitiveIndex].constructBoundingBox(), geometryDataWithPinPtr.parallelograms[instance.primitiveIndex].centroid());
                    blasBuildResultVector.push_back(BLAS::constructBLAS(
                            geometryDataWithPinPtr.spheres, 0, 0,
                            geometryDataWithPinPtr.parallelograms, instance.primitiveIndex, 1));
                    break;
                default:;
            }
            SDL_Log("BLAS for instance [%zd]: Node array length: %zd, index array length: %zd.", i, blasBuildResultVector[i].first.size(), blasBuildResultVector[i].second.size());
        }

        //将加速结构拷贝到页面锁定内存
        SDL_Log("Allocating and copying BLAS to pinned memory...");

        //分配指针数组内存
        Pair<BLASArray *, size_t> ret{};
        cudaCheckError(cudaHostAlloc(&ret.first, instanceCount * sizeof(BLASArray), cudaHostAllocDefault));

        //逐个BLAS拷贝数据
        for (size_t i = 0; i < instanceCount; i++) {
            const auto & blasNodeArray = blasBuildResultVector[i].first.data();
            const size_t blasNodeArrayLength = blasBuildResultVector[i].first.size();
            const auto & blasIndexArray = blasBuildResultVector[i].second.data();
            const size_t blasIndexArrayLength = blasBuildResultVector[i].second.size();

            //Node
            cudaCheckError(cudaHostAlloc(&ret.first[i].first.first, blasNodeArrayLength * sizeof(BLASNode), cudaHostAllocDefault));
            memcpy(ret.first[i].first.first, blasNodeArray, blasNodeArrayLength * sizeof(BLASNode));
            ret.first[i].first.second = blasNodeArrayLength;

            //Index
            cudaCheckError(cudaHostAlloc(&ret.first[i].second.first, blasIndexArrayLength * sizeof(BLASIndex), cudaHostAllocDefault));
            memcpy(ret.first[i].second.first, blasIndexArray, blasIndexArrayLength * sizeof(BLASIndex));
            ret.first[i].second.second = blasIndexArrayLength;

            SDL_Log("Copying BLAS [%zd] to pinned memory...", i);
        }
        ret.second = instanceCount;

        return ret;
    }
    void Renderer::freeBLASPinMem(Pair<BLASArray *, size_t> & blasWithPinPtr) {
        SDL_Log("Freeing pinned memory for BLAS...");
        for (size_t i = 0; i < blasWithPinPtr.second; i++) {
            cudaCheckError(cudaFreeHost(blasWithPinPtr.first[i].first.first));
            cudaCheckError(cudaFreeHost(blasWithPinPtr.first[i].second.first));
        }
        blasWithPinPtr = {};
    }

    // ====== TLAS ======

    TLASArray Renderer::buildTLASPinMem(Instance * pin_instances, size_t instanceCount) {
        //SDL_Log("Building TLAS...");

        const TLASBuildResult tlasBuildResult = TLAS::constructTLAS(pin_instances, instanceCount);
        //SDL_Log("TLAS: Node array length: %zd; Index array length: %zd.", tlasBuildResult.first.size(), tlasBuildResult.second.size());
        //SDL_Log("Allocating and copying TLAS to pinned memory ...");

        const auto & tlasNodeArray = tlasBuildResult.first.data();
        const size_t tlasNodeArrayLength = tlasBuildResult.first.size();
        const auto & tlasIndexArray = tlasBuildResult.second.data();
        const size_t tlasIndexArrayLength = tlasBuildResult.second.size();

        TLASArray ret{};

        //Node
        cudaCheckError(cudaHostAlloc(&ret.first.first, tlasNodeArrayLength * sizeof(TLASNode), cudaHostAllocDefault));
        memcpy(ret.first.first, tlasNodeArray, tlasNodeArrayLength * sizeof(TLASNode));
        ret.first.second = tlasNodeArrayLength;

        //Index
        cudaCheckError(cudaHostAlloc(&ret.second.first, tlasIndexArrayLength * sizeof(TLASIndex), cudaHostAllocDefault));
        memcpy(ret.second.first, tlasIndexArray, tlasIndexArrayLength * sizeof(TLASIndex));
        ret.second.second = tlasIndexArrayLength;

        return ret;
    }
    void Renderer::freeTLASPinMem(TLASArray & tlasWithPinPtr) {
        //SDL_Log("Freeing pinned memory for TLAS...");
        cudaCheckError(cudaFreeHost(tlasWithPinPtr.first.first));
        cudaCheckError(cudaFreeHost(tlasWithPinPtr.second.first));
        tlasWithPinPtr = {};
    }

    // ====== 渲染 ======

    TraverseData Renderer::traverseDevPtr(
            const SceneGeometryData & geometryDataWithDevPtr,
            const SceneMaterialData & materialDataWithDevPtr,
            const Instance * dev_instances,
            const Pair<BLASArray *, size_t> & blasWithDevPtr,
            const TLASArray & tlasWithDevPtr)
    {
        return {
                //几何体和材质
                .dev_spheres = geometryDataWithDevPtr.spheres,
                .dev_parallelograms = geometryDataWithDevPtr.parallelograms,

                .dev_roughs = materialDataWithDevPtr.roughs,
                .dev_metals = materialDataWithDevPtr.metals,

                //实例
                .dev_instances = dev_instances,

                //加速结构
                .tlasNodeArray = tlasWithDevPtr.first.first,
                .tlasIndexArray = tlasWithDevPtr.second.first,
                .blasArray = blasWithDevPtr.first
        };
    }
}