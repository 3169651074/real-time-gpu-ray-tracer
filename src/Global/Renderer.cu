#include <Global/Renderer.cuh>
#include <Global/VTKReader.cuh>
#include <JSON/json.hpp>
#include <fstream>
#include <Global/SDL_OpenGLWindow.cuh>
using json = nlohmann::json;

namespace project {
    SceneGeometryDataPackage Renderer::commitGeometryData(GeometryData & data, SceneVTKDataPackage * vtkData) {
        //读取VTK粒子几何信息
        if (vtkData != nullptr) {
            //读取第一个VTK文件
            const auto vtk = VTKReader::readVTKFile(vtkData->seriesFilePath + vtkData->vtkFileAndTimeArray[0].first);
            //转换为三角形数组
            const auto triangles = VTKReader::convertToRendererData(vtk).first;
            //将三角形数组插入到data.triangles头部
            data.triangles.insert(data.triangles.begin(), triangles.begin(), triangles.end());
            vtkData->vtkTriangleCount = triangles.size();
        }

        //为几何体分配页面锁定内存
        SDL_Log("Allocating pinned memory for geometries...");
        SceneGeometryData pin_geoData = {
                .sphereCount = data.spheres.size(),
                .parallelogramCount = data.parallelograms.size(),
                .triangleCount = data.triangles.size()
        };
        RendererImpl::allocGeoPinMem(pin_geoData);

        //写入几何体信息
        memcpy(pin_geoData.spheres, data.spheres.data(), pin_geoData.sphereCount * sizeof(Sphere));
        memcpy(pin_geoData.parallelograms, data.parallelograms.data(), pin_geoData.parallelogramCount * sizeof(Parallelogram));
        memcpy(pin_geoData.triangles, data.triangles.data(), pin_geoData.triangleCount * sizeof(Triangle));

        //拷贝到全局内存
        SDL_Log("Allocating global memory for geometries...");
        SceneGeometryData dev_geoData = RendererImpl::allocGeoGlobMem(nullptr, pin_geoData);
        SDL_Log("Copying geometry to global memory...");
        RendererImpl::copyGeoToGlobMem(nullptr, dev_geoData, pin_geoData);

        SDL_Log("Geometry data process completed.");
        return {pin_geoData, dev_geoData};
    }

    SceneMaterialDataPackage Renderer::commitMaterialData(MaterialData & data, SceneVTKDataPackage * vtkData) {
        if (vtkData != nullptr) {
            // ...
        }

        SDL_Log("Allocating pinned memory for materials...");
        SceneMaterialData pin_matData = {
                .roughCount = data.roughs.size(),
                .metalCount = data.metals.size()
        };
        RendererImpl::allocMatPinMem(pin_matData);

        memcpy(pin_matData.roughs, data.roughs.data(), pin_matData.roughCount * sizeof(Rough));
        memcpy(pin_matData.metals, data.metals.data(), pin_matData.metalCount * sizeof(Metal));

        SDL_Log("Allocating global memory for materials...");
        SceneMaterialData dev_matData = RendererImpl::allocMatGlobMem(nullptr, pin_matData);
        SDL_Log("Copying materials to global memory...");
        RendererImpl::copyMatToGlobMem(nullptr, dev_matData, pin_matData);

        SDL_Log("Material data process completed.");
        return {
            pin_matData, dev_matData
        };
    }

    SceneInstanceDataPackage Renderer::configureInstances(
        std::vector<Pair<PrimitiveType, size_t>> & insMapArray,
        void (*updateInstances)(Instance * pin_instances, size_t instanceCount, size_t frameCount),
        const SceneVTKDataPackage * vtkData)
    {
        const size_t instanceCount = insMapArray.size();
        size_t vtkInstanceCount = 0;
        std::vector<Instance> vtkInstances;

        //将VTK实例对象添加到实例数组之后，实例对象在实例数组中存放顺序无影响，存放在后面可以避免修改原有实例映射参数
        if (vtkData != nullptr) {
            const auto vtk = VTKReader::readVTKFile(vtkData->seriesFilePath + vtkData->vtkFileAndTimeArray[0].first);
            vtkInstances = VTKReader::convertToRendererData(vtk).second;
            vtkInstanceCount = vtkInstances.size();
        }

        SceneInstanceDataPackage insData = {
                .instanceCount = instanceCount + vtkInstanceCount, //VTK粒子参与分配内存，但不参与信息计算
                .updateInstances = updateInstances
        };

        //分配实例数组内存
        SDL_Log("Allocating memory for instances...");
        for (size_t i = 0; i < 2; i++) {
            insData.pin_instanceArrays[i] = RendererImpl::allocInstPinMem(instanceCount + vtkInstanceCount);
            insData.dev_instanceArrays[i] = RendererImpl::allocInstGlobMem(nullptr, instanceCount + vtkInstanceCount);
        }

        //设置每个实例和几何体的对应关系（只设置页面锁定内存中第一个缓冲实例数组，另一个直接拷贝内存）
        SDL_Log("Mapping instance and geometries...");
        for (size_t i = 0; i < instanceCount; i++) {
            insData.pin_instanceArrays[0][i].primitiveType = insMapArray[i].first;
            insData.pin_instanceArrays[0][i].primitiveIndex = insMapArray[i].second;

            //如果是三角形类型且有VTK数据，需要调整索引（因为VTK三角形被插入到数组头部）
            if (vtkData != nullptr && insMapArray[i].first == PrimitiveType::TRIANGLE) {
                insData.pin_instanceArrays[0][i].primitiveIndex += vtkData->vtkTriangleCount;
            }
        }
        //更新实例信息。instanceCount = 自定义实例数量；insData.instanceCount = 自定义实例数量 + VTK实例数量
        updateInstances(insData.pin_instanceArrays[0], instanceCount, 0);

        //拷贝VTK实例对象到页面锁定内存
        if (vtkInstanceCount > 0) {
            memcpy(insData.pin_instanceArrays[0] + instanceCount, vtkInstances.data(), vtkInstanceCount * sizeof(Instance));
        }

        //只部分初始化了实例数组，需要在构建BLAS后完成对asIndex的赋值，并需要拷贝至另一实例缓冲区和全局内存
        SDL_Log("Instance data pre-init completed.");
        return insData;
    }

    SceneAccelerationStructurePackage Renderer::buildAccelerationStructure(
            const SceneGeometryDataPackage & geoData, SceneInstanceDataPackage & insData)
    {
        SceneAccelerationStructurePackage asData{};

        SDL_Log("Building BLAS...");
        asData.pin_blasArray = RendererImpl::buildBLASPinMem(
                geoData.pin_sceneGeometryData,
                insData.pin_instanceArrays[0],
                insData.instanceCount); //为所有实例构建BLAS，包括VTK实例
        asData.dev_blasArray = RendererImpl::copyBLASToGlobMem(nullptr, asData.pin_blasArray);

        //补全实例信息
        memcpy(insData.pin_instanceArrays[1], insData.pin_instanceArrays[0], insData.instanceCount * sizeof(Instance));
        //拷贝到全局内存，此时页面锁定内存完全初始化，全局内存完全分配但只拷贝一个
        RendererImpl::copyInstToGlobMem(nullptr, insData.dev_instanceArrays[0], insData.pin_instanceArrays[0], insData.instanceCount);
        SDL_Log("Instance initialization completed.");

//        for (size_t i = 0; i < insData.instanceCount; i++) {
//            SDL_Log("Ins[%zd]:type=%zd,idx=%zd,cnt=%zd",i,
//                    insData.pin_instanceArrays[0][i].primitiveType,
//                    insData.pin_instanceArrays[0][i].primitiveIndex,
//                    insData.pin_instanceArrays[0][i].primitiveCount);
//        }

        SDL_Log("Building TLAS...");
        //TLAS双缓冲，初始时只需要在第一个缓冲区构建即可
        asData.pin_tlasArrays[0] = RendererImpl::buildTLASPinMem(insData.pin_instanceArrays[0], insData.instanceCount);
        asData.dev_tlasArrays[0] = RendererImpl::copyTLASGlobMem(nullptr, asData.pin_tlasArrays[0]);

        SDL_Log("Acceleration structure build completed.");
        return asData;
    }

    Camera * Renderer::configureCamera(const WindowInput & windowData, const CameraInput & cameraData) {
        SDL_Log("Calculating camera properties...");

        auto * pin_camera = RendererImpl::allocCamPinMem();
        pin_camera->windowWidth = windowData.windowWidth;
        pin_camera->windowHeight = windowData.windowHeight;
        pin_camera->backgroundColor = cameraData.backgroundColor;
        pin_camera->cameraCenter = cameraData.initialCenter;
        pin_camera->cameraTarget = cameraData.initialTarget;
        pin_camera->fov = cameraData.fov;
        pin_camera->upDirection = cameraData.upDirection;
        pin_camera->focusDiskRadius = cameraData.focusDiskRadius;
        pin_camera->sampleRange = cameraData.sampleRange;
        pin_camera->sampleCount = cameraData.sampleCount;
        pin_camera->rayTraceDepth = cameraData.rayTraceDepth;

        RendererImpl::calculateCameraProperties(pin_camera);
        RendererImpl::copyCamToConstMem(nullptr, pin_camera);

        SDL_Log("Camera initialization completed.");
        return pin_camera;
    }

    void Renderer::startRender(
            const WindowInput & windowData,
            const SceneGeometryDataPackage & geoData, const SceneMaterialDataPackage & matData,
            SceneInstanceDataPackage & insData, SceneAccelerationStructurePackage & asData,
            Camera * pin_camera)
    {
        SDL_Log("Preparing first frame...");

        //当前使用的缓冲索引
        size_t currentBufferIndex = 0;
        //遍历参数结构体也需要双缓冲
        TraverseData * dev_traverseData[2];
        for (auto & i : dev_traverseData) {
            cudaCheckError(cudaMalloc(&i, sizeof(TraverseData)));
        }

        //事件
        cudaEvent_t copyCompleteEvents[2];
        cudaEvent_t renderCompleteEvents[2];
        for (size_t i = 0; i < 2; i++) {
            cudaCheckError(cudaEventCreate(&copyCompleteEvents[i]));
            cudaCheckError(cudaEventCreate(&renderCompleteEvents[i]));
        }

        //拷贝流和计算流
        cudaStream_t copyStream, renderStream;
        cudaCheckError(cudaStreamCreate(&copyStream));
        cudaCheckError(cudaStreamCreate(&renderStream));

        //准备第一帧数据
        const auto traverseData = RendererImpl::traverseDevPtr(
                geoData.dev_sceneGeometryData, matData.dev_sceneMaterialData,
                insData.dev_instanceArrays[0],
                asData.dev_blasArray, asData.dev_tlasArrays[0]);
        cudaCheckError(cudaMemcpyAsync(dev_traverseData[0], &traverseData, sizeof(TraverseData), cudaMemcpyHostToDevice, copyStream));
        cudaCheckError(cudaEventRecord(copyCompleteEvents[0], copyStream));

        SDL_Log("Initializing graphics api...");
        auto sdlWindow = SDL_OpenGLWindow::createSDLGLWindow(
                windowData.title.c_str(), windowData.windowWidth, windowData.windowHeight);
        auto window = sdlWindow.first;
        auto args = SDL_OpenGLWindow::initializeOGL(window);

        SDL_OpenGLWindow::CudaArgs CudaArgs = SDL_OpenGLWindow::getCudaResource(window, args);
        SDL_OpenGLWindow::OperateArgs operateArgs = SDL_OpenGLWindow::getOperateArgs(
                120, 0.001f, 80, 2, 0.05f);
        SDL_OpenGLWindow::KeyMouseInputArgs inputArgs{};

        SDL_Log("Starting render loop...");
        size_t frameCount = 0;
        SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_TRUE));

        while (!inputArgs.keyQuit) {
            const auto frameStartTime = std::chrono::steady_clock::now();

            //确定当前帧用于渲染的缓冲区索引，和下一帧用于更新的缓冲区索引
            const size_t renderIdx = currentBufferIndex;
            const size_t updateIdx = (currentBufferIndex + 1) % 2;

            // ====== 主机：准备下一帧(updateIdx)的数据 ======

            //更新相机
            SDL_OpenGLWindow::getKeyMouseInput(inputArgs);
            if (inputArgs.keyQuit) { break; }
            if (inputArgs.mouseClick) {
                SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() ? SDL_FALSE : SDL_TRUE));
            }
            const auto newPos = SDL_OpenGLWindow::calculateNewPosition(
                    operateArgs, inputArgs,
                    pin_camera->cameraCenter, pin_camera->cameraTarget,
                    pin_camera->upDirection, pin_camera->cameraU, pin_camera->cameraV, pin_camera->cameraW);
            if (newPos.first) {
                pin_camera->cameraCenter = newPos.second.first;
                pin_camera->cameraTarget = newPos.second.second;
                RendererImpl::calculateCameraProperties(pin_camera);
            }
            if (inputArgs.dSpeed != 0) {
                if (inputArgs.dSpeed > 0) {
                    operateArgs.moveSpeed += operateArgs.moveSpeedChangeStep;
                } else {
                    operateArgs.moveSpeed = operateArgs.moveSpeed < operateArgs.moveSpeedChangeStep ? 0.0f : operateArgs.moveSpeed - operateArgs.moveSpeedChangeStep;
                }
            }

            /*
             * 更新实例
             * 此处updateInstance只更新updateIdx缓冲区的实例对象的变换信息（实例基础信息在循环前已固定）
             * 实例双缓冲使得此时设备可以读取另一个缓冲区的实例信息
             */
            insData.updateInstances(insData.pin_instanceArrays[updateIdx], insData.instanceCount, frameCount);

            //为下一帧构建TLAS（在页面锁定内存中。先释放原有内存）
            if (asData.pin_tlasArrays[updateIdx].first.first != nullptr) {
                RendererImpl::freeTLASPinMem(asData.pin_tlasArrays[updateIdx]);
            }
            asData.pin_tlasArrays[updateIdx] = RendererImpl::buildTLASPinMem(
                    insData.pin_instanceArrays[updateIdx], insData.instanceCount);

            // ====== 拷贝流：异步更新下一帧(updateIdx)的缓冲区 ======

            //等待渲染流完成对该缓冲区的渲染
            cudaCheckError(cudaStreamWaitEvent(copyStream, renderCompleteEvents[updateIdx], 0));

            //释放上一轮为这个updateIdx分配的TLAS设备内存
            if (asData.dev_tlasArrays[updateIdx].first.first != nullptr) {
                RendererImpl::freeTLASGlobMem(copyStream, asData.dev_tlasArrays[updateIdx]);
            }

            //异步拷贝实例、TLAS和相机数据到全局内存
            RendererImpl::copyInstToGlobMem(
                    copyStream, insData.dev_instanceArrays[updateIdx],
                    insData.pin_instanceArrays[updateIdx], insData.instanceCount);
            RendererImpl::copyCamToConstMem(copyStream, pin_camera);
            asData.dev_tlasArrays[updateIdx] = RendererImpl::copyTLASGlobMem(copyStream, asData.pin_tlasArrays[updateIdx]);

            //设置并异步拷贝下一帧的遍历指针结构体
            const auto nextTraverseData = RendererImpl::traverseDevPtr(
                    geoData.dev_sceneGeometryData, matData.dev_sceneMaterialData,
                    insData.dev_instanceArrays[updateIdx],
                    asData.dev_blasArray, asData.dev_tlasArrays[updateIdx]);
            cudaCheckError(cudaMemcpyAsync(dev_traverseData[updateIdx], &nextTraverseData, sizeof(TraverseData), cudaMemcpyHostToDevice, copyStream));

            //在当所有拷贝操作都提交后，在拷贝流中记录事件
            cudaCheckError(cudaEventRecord(copyCompleteEvents[updateIdx], copyStream));

            // ====== 渲染流：渲染当前帧(renderIdx) ======

            //等待“当前帧数据准备就绪”事件
            cudaCheckError(cudaStreamWaitEvent(renderStream, copyCompleteEvents[renderIdx], 0));

            //映射资源
            SDL_OpenGLWindow::mapCudaResource(renderStream, CudaArgs);
            //启动核函数，使用当前帧(renderIdx)的TraverseData
            render<<<CudaArgs.blocks, CudaArgs.threads, 0, renderStream>>>(dev_traverseData[renderIdx], CudaArgs.surfaceObject);
            //渲染完成后，清理资源 (同样在渲染流上)
            SDL_OpenGLWindow::unmapCudaResource(renderStream, CudaArgs);
            //记录当前帧(renderIdx)的渲染已在渲染流上完成
            cudaCheckError(cudaEventRecord(renderCompleteEvents[renderIdx], renderStream));

            // ====== 主机：显示与交换 ======

            SDL_OpenGLWindow::presentFrame(window, args);

            //切换缓冲区索引，为下一轮循环做准备
            currentBufferIndex = updateIdx;
            frameCount++;

            //帧数控制
            const auto workTime = std::chrono::steady_clock::now() - frameStartTime;
            if (workTime < operateArgs.targetFrameDuration) {
                const auto timeToWait = operateArgs.targetFrameDuration - workTime;
                //1. 粗略休眠
                if (timeToWait > operateArgs.sleepMargin) {
                    std::this_thread::sleep_for(timeToWait - operateArgs.sleepMargin);
                }
                //2. 精确自旋
                while (std::chrono::steady_clock::now() - frameStartTime < operateArgs.targetFrameDuration) {}
            }
        }
        SDL_Log("Render completed.");

        //等待所有工作完成
        cudaCheckError(cudaDeviceSynchronize());

        // ====== 清理临时资源 ======

        //释放SDL，OGL和CUDA资源
        SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_FALSE));
        SDL_OpenGLWindow::releaseCudaResource(CudaArgs);
        SDL_OpenGLWindow::releaseOGL(args);
        SDL_OpenGLWindow::destroySDLGLWindow(sdlWindow.first, sdlWindow.second);

        //销毁流
        cudaCheckError(cudaStreamDestroy(copyStream));
        cudaCheckError(cudaStreamDestroy(renderStream));

        //销毁事件
        for (size_t i = 0; i < 2; i++) {
            cudaCheckError(cudaEventDestroy(copyCompleteEvents[i]));
            cudaCheckError(cudaEventDestroy(renderCompleteEvents[i]));
        }

        //释放全局内存和页面锁定内存
        for (size_t i = 0; i < 2; i++) {
            cudaCheckError(cudaFree(dev_traverseData[i]));
        }
    }

    void Renderer::cleanup(SceneGeometryDataPackage & geoData, SceneMaterialDataPackage & matData,
                           SceneAccelerationStructurePackage & asData,
                           SceneInstanceDataPackage & insData, Camera * & pin_camera) {
        SDL_Log("Cleaning up resources...");

        RendererImpl::freeBLASGlobMem(nullptr, asData.dev_blasArray);
        RendererImpl::freeBLASPinMem(asData.pin_blasArray);

        for (size_t i = 0; i < 2; i++) {
            RendererImpl::freeInstGlobMem(nullptr, insData.dev_instanceArrays[i]);
            RendererImpl::freeInstPinMem(insData.pin_instanceArrays[i]);
        }
        RendererImpl::freeCamPinMem(pin_camera);
        RendererImpl::freeMatGlobMem(nullptr, matData.dev_sceneMaterialData);
        RendererImpl::freeGeoGlobMem(nullptr, geoData.dev_sceneGeometryData);
        RendererImpl::freeMatPinMem(matData.pin_sceneMaterialData);
        RendererImpl::freeGeoPinMem(geoData.pin_sceneGeometryData);

        SDL_Log("Cleanup completed.");
        geoData = {};
        matData = {};
        asData = {};
        insData = {};
        pin_camera = nullptr;
    }

    SceneVTKDataPackage Renderer::configureVTKFiles(const std::string & seriesFilePath) {
        SDL_Log("Reading series file: %s", seriesFilePath.c_str());

        //打开 JSON 文件
        std::ifstream file(seriesFilePath);
        if (!file.is_open()) {
            SDL_Log("Could not open the series file!");
            exit(-1);
        }

        //解析 JSON 文件
        json data;
        try {
            //使用 parse() 函数将文件流解析成一个 json 对象
            data = json::parse(file);
        } catch (json::parse_error & e) {
            //如果文件内容不是有效的 JSON 格式，会抛出异常
            SDL_Log("JSON parsing error: %s", e.what());
            exit(-1);
        }

        //从 JSON 对象中提取数据
        const std::string version = data["file-series-version"];
        SDL_Log("Series file version: %s", version.c_str());

        if (!(data.contains("files") && data["files"].is_array())) {
            SDL_Log("Failed to parse files array in series file!");
            exit(-1);
        }

        std::vector<Pair<std::string, float>> fileEntries;
        for (const auto & item : data["files"]) {
            //从数组中的每个对象里提取 "name" 和 "time"
            const std::string name = item["name"];
            const float time = item["time"];
            fileEntries.push_back({name, time});
        }

        const size_t entryCount = fileEntries.size();
        const size_t printEntryCount = std::min(entryCount, static_cast<size_t>(5));
        SDL_Log("Found %zd vtk entries.", entryCount);
        SDL_Log("First %zd entries:", printEntryCount);
        for (size_t i = 0; i < printEntryCount; i++) {
            SDL_Log("Time: %f --> VTK file: %s", fileEntries[i].second, fileEntries[i].first.c_str());
        }
        return {
            .seriesFilePath = seriesFilePath.substr(0, seriesFilePath.size() - std::string("particle_mesh.vtk.series").size()),
            .vtkFileAndTimeArray = fileEntries
        };
    }
}