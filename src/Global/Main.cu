#include <Global/Render.cuh>
#include <Global/SDL_OpenGLWindow.cuh>
using namespace renderer;

void initGeoAndMat(SceneGeometryData & geometryDataWithPinPtr, SceneMaterialData & materialDataWithPinPtr) {
    materialDataWithPinPtr.roughs[0] = {.65, .05, .05};
    materialDataWithPinPtr.roughs[1] = {.73, .73, .73};
    materialDataWithPinPtr.roughs[2] = {.12, .45, .15};
    materialDataWithPinPtr.roughs[3] = {.70, .60, .50};
    materialDataWithPinPtr.metals[0] = {0.8, 0.85, 0.88, 0.0};

    geometryDataWithPinPtr.spheres[0] = {MaterialType::ROUGH, 3, {0.0, 0, 0.0}, 1000.0};
    geometryDataWithPinPtr.spheres[1] = {MaterialType::ROUGH, 0, {0.0, 0, 0.0}, 2.0};
    geometryDataWithPinPtr.parallelograms[0] = {MaterialType::ROUGH, 1, {0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 4.0, 0.0}};
}

void initCamera(Camera * pin_camera) {
    pin_camera->windowWidth = 1200;
    pin_camera->windowHeight = 800;
    pin_camera->backgroundColor = {0.7, 0.8, 0.9};
    pin_camera->cameraCenter = {0.0, 2.0, 10.0};
    pin_camera->cameraTarget = {0.0, 2.0, 0.0};
    pin_camera->fov = 90;
    pin_camera->upDirection = {0.0, 1.0, 0.0};
    pin_camera->focusDiskRadius = 0.0;
    pin_camera->sampleRange = 0.5;
    pin_camera->sampleCount = 1;
    pin_camera->rayTraceDepth = 10;
}

void updateInstance(const SceneGeometryData & geometryData, Instance * pin_instances, size_t instanceCount, size_t frame) {
    pin_instances[0] = Instance(PrimitiveType::SPHERE, 0,
                                {},
                                {0.0, -1000.0, 0.0},
                                {1.0, 1.0, 1.0});
    //初始中心点
    const Point3 initialCenter = {0.0, 2.0, 0.0};
    //定义旋转半径
    const float radius = 2.0f;
    //定义旋转速度（弧度/帧）
    const float speed = 0.02f;
    //计算当前帧的角度
    const float angle = static_cast<float>(frame) * speed;
    const float3 newCenter = {
            initialCenter[0] + radius * std::cos(angle) * 1.5f,
            initialCenter[1] + radius * std::sin(angle) * std::cos(angle),
            initialCenter[2] + radius * std::sin(angle) * 1.5f
    };
    pin_instances[1] = Instance(PrimitiveType::SPHERE, 1,
                                {},
                                newCenter,
                                {1.0, 1.0, 1.0});
    pin_instances[2] = Instance(PrimitiveType::PARALLELOGRAM, 0,
                                {0.0, 0.0, 0.0},
                                {-5.0, 0.0, 0.0},
                                {1.0, 1.0, 1.0});

    //为实例准备包围盒和asIndex
    for (size_t i = 0; i < instanceCount; ++i) {
        Instance& instance = pin_instances[i];
        instance.asIndex = i; //一个实例对应一个BLAS，索引就是它自己的索引
        switch (instance.primitiveType) {
            case PrimitiveType::SPHERE:
                instance.setBoundingBoxProperties(
                        geometryData.spheres[instance.primitiveIndex].constructBoundingBox(),
                        geometryData.spheres[instance.primitiveIndex].centroid()
                );
                break;
            case PrimitiveType::PARALLELOGRAM:
                instance.setBoundingBoxProperties(
                        geometryData.parallelograms[instance.primitiveIndex].constructBoundingBox(),
                        geometryData.parallelograms[instance.primitiveIndex].centroid()
                );
                break;
            default:
                break;
        }
    }
}

#undef main
int main(int argc, char * argv[]) {
    // ====== 初始化 ======

    //几何体和材质
    SceneGeometryData geometryDataWithPinPtr = {
            .sphereCount = 2,
            .parallelogramCount = 1
    };
    SceneMaterialData materialDataWithPinPtr = {
            .roughCount = 4,
            .metalCount = 1
    };
    Renderer::allocGeoPinMem(geometryDataWithPinPtr);
    Renderer::allocMatPinMem(materialDataWithPinPtr);
    initGeoAndMat(geometryDataWithPinPtr, materialDataWithPinPtr);
    auto geometryDataWithDevPtr = Renderer::allocGeoGlobMem(nullptr, geometryDataWithPinPtr);
    auto materialDataWithDevPtr = Renderer::allocMatGlobMem(nullptr, materialDataWithPinPtr);
    Renderer::copyGeoToGlobMem(nullptr, geometryDataWithDevPtr, geometryDataWithPinPtr);
    Renderer::copyMatToGlobMem(nullptr, materialDataWithDevPtr, materialDataWithPinPtr);

    //相机
    Camera * pin_camera = Renderer::allocCamPinMem();
    initCamera(pin_camera);
    Renderer::calculateCameraProperties(pin_camera);

    //实例
    const size_t instanceCount = 3;
    Instance * pin_instances[2];
    Instance * dev_instances[2];
    for (size_t i = 0; i < lengthOf(pin_instances); i++) {
        pin_instances[i] = Renderer::allocInstPinMem(instanceCount);
        cudaCheckError(cudaMalloc(&dev_instances[i], instanceCount * sizeof(Instance)));
    }

    //BLAS
    updateInstance(geometryDataWithPinPtr, pin_instances[0], instanceCount, 0);
    auto pin_blas = Renderer::buildBLASPinMem(geometryDataWithPinPtr, pin_instances[0], instanceCount);
    auto dev_blas = Renderer::copyBLASToGlobMem(nullptr, pin_blas);

    //TLAS
    size_t currentBufferIndex = 0;
    TLASArray pin_tlas[2] {};
    TLASArray dev_tlas[2] {};

    //遍历参数结构体也需要双缓冲
    TraverseData * dev_traverseData[2];
    for (size_t i = 0; i < lengthOf(dev_traverseData); i++) {
        cudaCheckError(cudaMalloc(&dev_traverseData[i], sizeof(TraverseData)));
    }

    cudaEvent_t copyCompleteEvents[2];
    cudaEvent_t renderCompleteEvents[2];
    for (size_t i = 0; i < lengthOf(copyCompleteEvents); i++) {
        cudaCheckError(cudaEventCreate(&copyCompleteEvents[i]));
        cudaCheckError(cudaEventCreate(&renderCompleteEvents[i]));
    }

    //初始化拷贝流和计算流
    cudaStream_t copyStream, renderStream;
    cudaCheckError(cudaStreamCreate(&copyStream));
    cudaCheckError(cudaStreamCreate(&renderStream));

    //准备第一帧数据
    pin_tlas[0] = Renderer::buildTLASPinMem(pin_instances[0], instanceCount);
    dev_tlas[0] = Renderer::copyTLASGlobMem(copyStream, pin_tlas[0]);
    Renderer::copyInstToGlobMem(copyStream, dev_instances[0], pin_instances[0], instanceCount);
    Renderer::copyCamToConstMem(copyStream, pin_camera);
    const auto _traverseData = Renderer::traverseDevPtr(
            geometryDataWithDevPtr, materialDataWithDevPtr,
            dev_instances[0],
            dev_blas, dev_tlas[0]);
    cudaCheckError(cudaMemcpyAsync(dev_traverseData[0], &_traverseData, sizeof(TraverseData), cudaMemcpyHostToDevice, copyStream));
    cudaCheckError(cudaEventRecord(copyCompleteEvents[0], copyStream));

    //初始化窗口，OGL，获取CUDA资源
    auto sdlWindow = SDL_OpenGLWindow::createSDLGLWindow("Test", 1200, 800);
    auto window = sdlWindow.first;
    auto oglArgs = SDL_OpenGLWindow::initializeOGL(window);

    SDL_OpenGLWindow::CUDAArgs cudaArgs = SDL_OpenGLWindow::getCudaResource(window, oglArgs);
    SDL_OpenGLWindow::OperateArgs operateArgs = SDL_OpenGLWindow::getOperateArgs(
            120, 0.001f, 80, 0.5f, 0.05f);
    SDL_OpenGLWindow::KeyMouseInputArgs inputArgs{};

    // ====== 循环更新 ======

    size_t frameCount = 0;
    SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_TRUE));

    //初始时渲染第一帧，准备第二帧
    while (!inputArgs.keyQuit) {
        //帧数限制：控制主机提交数据的速度
        const auto frameStartTime = std::chrono::steady_clock::now();

        //确定当前帧用于渲染的缓冲区索引，和下一帧用于更新的缓冲区索引
        const size_t renderIdx = currentBufferIndex;
        const size_t updateIdx = (currentBufferIndex + 1) % 2; // !currentBufferIndex

        // ====== 主机：准备下一帧(updateIdx)的数据 ======

        //接收输入并更新相机
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
            Renderer::calculateCameraProperties(pin_camera);
        }
        if (inputArgs.dSpeed != 0) {
            if (inputArgs.dSpeed > 0) {
                operateArgs.moveSpeed += operateArgs.moveSpeedChangeStep;
            } else {
                operateArgs.moveSpeed = operateArgs.moveSpeed < operateArgs.moveSpeedChangeStep ? operateArgs.moveSpeedChangeStep * 0.5f : operateArgs.moveSpeed - operateArgs.moveSpeedChangeStep;
            }
            SDL_Log("Change move speed to %.6f", operateArgs.moveSpeed);
        }
        //在后台缓冲区(updateIdx)的页面锁定内存中更新实例和TLAS
        updateInstance(geometryDataWithPinPtr, pin_instances[updateIdx], instanceCount, frameCount);
        //释放上一轮为这个updateIdx分配的TLAS页面锁定内存
        if (pin_tlas[updateIdx].first.first != nullptr) {
            Renderer::freeTLASPinMem(pin_tlas[updateIdx]);
        }
        pin_tlas[updateIdx] = Renderer::buildTLASPinMem(pin_instances[updateIdx], instanceCount);

        // ====== 拷贝流：异步更新下一帧(updateIdx)的缓冲区 ======

        //让拷贝流等待渲染流完成对该缓冲区的渲染，才能安全地覆写它
        cudaCheckError(cudaStreamWaitEvent(copyStream, renderCompleteEvents[updateIdx], 0));
        //释放上一轮为这个updateIdx分配的TLAS设备内存
        if (dev_tlas[updateIdx].first.first != nullptr) {
            Renderer::freeTLASGlobMem(copyStream, dev_tlas[updateIdx]);
        }
        //异步拷贝实例、TLAS和相机数据到全局内存
        Renderer::copyInstToGlobMem(copyStream, dev_instances[updateIdx], pin_instances[updateIdx], instanceCount);
        Renderer::copyCamToConstMem(copyStream, pin_camera);
        dev_tlas[updateIdx] = Renderer::copyTLASGlobMem(copyStream, pin_tlas[updateIdx]);
        //设置并异步拷贝下一帧的遍历指针结构体
        const auto traverseData = Renderer::traverseDevPtr(
                geometryDataWithDevPtr, materialDataWithDevPtr,
                dev_instances[updateIdx],
                dev_blas, dev_tlas[updateIdx]);
        cudaCheckError(cudaMemcpyAsync(dev_traverseData[updateIdx], &traverseData, sizeof(TraverseData), cudaMemcpyHostToDevice, copyStream));
        //在当所有拷贝操作都提交后，在拷贝流中记录一个事件
        cudaCheckError(cudaEventRecord(copyCompleteEvents[updateIdx], copyStream));

        // ====== 渲染流：渲染当前帧(renderIdx) ======

        //让渲染流等待“当前帧数据准备就绪”的事件
        cudaCheckError(cudaStreamWaitEvent(renderStream, copyCompleteEvents[renderIdx], 0));
        //映射资源
        SDL_OpenGLWindow::mapCudaResource(renderStream, cudaArgs);
        //启动渲染内核，使用当前帧(renderIdx)的TraverseData
        render<<<cudaArgs.blocks, cudaArgs.threads, 0, renderStream>>>(dev_traverseData[renderIdx], cudaArgs.surfaceObject);
        //渲染完成后，清理资源 (同样在渲染流上)
        SDL_OpenGLWindow::unmapCudaResource(renderStream, cudaArgs);
        //记录当前帧(renderIdx)的渲染已在渲染流上完成
        cudaCheckError(cudaEventRecord(renderCompleteEvents[renderIdx], renderStream));

        // ====== 显示与交换 ======
        SDL_OpenGLWindow::presentFrame(window, oglArgs);

        //切换缓冲区索引，为下一轮循环做准备
        currentBufferIndex = updateIdx;
        frameCount++;

        const auto workTime = std::chrono::steady_clock::now() - frameStartTime;

        //如果本帧工作时间小于目标帧时长，则需要等待
        if (workTime < operateArgs.targetFrameDuration) {
            const auto timeToWait = operateArgs.targetFrameDuration - workTime;
            //1. 粗略休眠：如果需要等待的时间较长，先进行一次低CPU占用的线程休眠
            if (timeToWait > operateArgs.sleepMargin) {
                std::this_thread::sleep_for(timeToWait - operateArgs.sleepMargin);
            }
            //2. 精确自旋：在最后几毫秒进行忙等待，以达到更精确的帧同步
            while (std::chrono::steady_clock::now() - frameStartTime < operateArgs.targetFrameDuration) {}
        }
    }
    SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_FALSE));

    // ====== 清理 ======
    //等待所有工作完成再开始清理
    cudaCheckError(cudaDeviceSynchronize());

    //释放SDL，OGL和CUDA资源
    SDL_OpenGLWindow::releaseCudaResource(cudaArgs);
    SDL_OpenGLWindow::releaseOGL(oglArgs);
    SDL_OpenGLWindow::destroySDLGLWindow(sdlWindow.first, sdlWindow.second);

    //销毁流
    cudaCheckError(cudaStreamDestroy(copyStream));
    cudaCheckError(cudaStreamDestroy(renderStream));

    //销毁事件
    for (size_t i = 0; i < lengthOf(copyCompleteEvents); i++) {
        cudaCheckError(cudaEventDestroy(copyCompleteEvents[i]));
        cudaCheckError(cudaEventDestroy(renderCompleteEvents[i]));
    }

    //释放全局内存和页面锁定内存
    for (size_t i = 0; i < lengthOf(dev_traverseData); i++) {
        cudaCheckError(cudaFree(dev_traverseData[i]));
    }
    Renderer::freeBLASGlobMem(nullptr, dev_blas);
    Renderer::freeBLASPinMem(pin_blas);

    for (size_t i = 0; i < lengthOf(pin_instances); i++) {
        Renderer::freeInstGlobMem(nullptr, dev_instances[i]);
        Renderer::freeInstPinMem(pin_instances[i]);
    }
    Renderer::freeCamPinMem(pin_camera);
    Renderer::freeMatGlobMem(nullptr, materialDataWithDevPtr);
    Renderer::freeGeoGlobMem(nullptr, geometryDataWithDevPtr);
    Renderer::freeMatPinMem(materialDataWithPinPtr);
    Renderer::freeGeoPinMem(geometryDataWithPinPtr);
    return 0;
}