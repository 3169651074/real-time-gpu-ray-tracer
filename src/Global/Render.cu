#include <Global/Render.cuh>

namespace renderer {
    //核函数声明
    __global__ void render(
            SceneGeometryData * dev_geometryData, SceneMaterialData * dev_materialData,
            cudaSurfaceObject_t surfaceObject, const ASTraverseData * dev_asTraverseData);

    //分配页面锁定内存
    /*
     * cudaHostAllocWriteCombined标志：写合并
     * CPU读取这块内存的性能差，仅适用于CPU主要负责写入、GPU主要负责读取的场景
     */
#define _mallocHost(structName, className, arrayName, countName) \
    do {                                    \
        if (structName.countName != 0) {    \
            cudaCheckError(cudaHostAlloc(&structName.arrayName, structName.countName * sizeof(className), cudaHostAllocDefault));\
        }                                   \
    } while(false)
    //geometryData和materialData本质上是指针，不存放在页面锁定内存中
    void Renderer::mallocPinnedMemory(
            SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & pin_camera, Instance * & pin_instances, size_t instanceCount)
    {
        //分配几何体和材质内存
        _mallocHost(geometryData, Sphere, spheres, sphereCount);
        _mallocHost(geometryData, Parallelogram, parallelograms, parallelogramCount);

        _mallocHost(materialData, Rough, roughs, roughCount);
        _mallocHost(materialData, Metal, metals, metalCount);

        //分配相机内存
        cudaCheckError(cudaHostAlloc(&pin_camera, sizeof(Camera), cudaHostAllocDefault));

        //分配实例内存
        cudaCheckError(cudaHostAlloc(&pin_instances, instanceCount * sizeof(Instance), cudaHostAllocDefault));

        SDL_Log("Pinned memory allocated.");
    }

    //释放页面锁定内存
#define _freeHost(structName, arrayName) cudaCheckError(cudaFreeHost(structName.arrayName))
    void Renderer::freePinnedMemory(
            SceneGeometryData & geometryData, SceneMaterialData & materialData,
            Camera * & pin_camera, Instance * & pin_instances)
    {
        _freeHost(geometryData, spheres);
        _freeHost(geometryData, parallelograms);

        _freeHost(materialData, roughs);
        _freeHost(materialData, metals);

        //释放相机空间
        cudaCheckError(cudaFreeHost(pin_camera));

        //释放实例空间
        cudaCheckError(cudaFreeHost(pin_instances));

        SDL_Log("Pinned memory freed.");
    }

    //分配全局内存并拷贝数据
#define _mallocGlobalAndCopy(srcStructName, dstStructName, arrayName, countName, className)\
        do {                                        \
            if (srcStructName.countName != 0) {     \
                cudaCheckError(cudaMalloc(&dstStructName.arrayName, srcStructName.countName * sizeof(className)));\
                cudaCheckError(cudaMemcpy(dstStructName.arrayName, srcStructName.arrayName, srcStructName.countName * sizeof(className), cudaMemcpyHostToDevice));\
            }                                       \
        } while (false)
    Pair<SceneGeometryData, SceneMaterialData> Renderer::copyToGlobalMemory(
            const SceneGeometryData & geometryData, const SceneMaterialData & materialData,
            const Camera * pin_camera)
    {
        //拷贝数组长度信息
        SceneGeometryData geometryDataWithDevPtr = geometryData;
        SceneMaterialData materialDataWithDevPtr = materialData;

        _mallocGlobalAndCopy(geometryData, geometryDataWithDevPtr, spheres, sphereCount, Sphere);
        _mallocGlobalAndCopy(geometryData, geometryDataWithDevPtr, parallelograms, parallelogramCount, Parallelogram);

        _mallocGlobalAndCopy(materialData, materialDataWithDevPtr, roughs, roughCount, Rough);
        _mallocGlobalAndCopy(materialData, materialDataWithDevPtr, metals, metalCount, Metal);

        //拷贝相机到常量内存
        cudaCheckError(cudaMemcpyToSymbol(dev_camera, pin_camera, sizeof(Camera)));
        SDL_Log("Global memory allocated.");

        return {geometryDataWithDevPtr, materialDataWithDevPtr};
    }

    //释放全局内存，无需释放常量内存
#define _freeGlobal(structName, arrayName) cudaCheckError(cudaFree(structName.arrayName))
    void Renderer::freeGlobalMemory(
            SceneGeometryData & geometryDataWithDevPtr, SceneMaterialData & materialDataWithDevPtr)
    {
        _freeGlobal(geometryDataWithDevPtr, spheres);
        _freeGlobal(geometryDataWithDevPtr, parallelograms);

        _freeGlobal(materialDataWithDevPtr, roughs);
        _freeGlobal(materialDataWithDevPtr, metals);

        SDL_Log("Global memory freed");
    }

    ASBuildResult Renderer::buildAccelerationStructure(
            const SceneGeometryData & geometryDataWithPinPtr, Instance * pin_instances, size_t instanceCount)
    {
        SDL_Log("Building BLAS...");
        const Sphere * spheres = geometryDataWithPinPtr.spheres;
        const Parallelogram * parallelograms = geometryDataWithPinPtr.parallelograms;

        //构建BLAS：为每个物体创建独立的BLAS，存储到数组中
        //当前一个实例对应一个BLAS
        std::vector<BLASBuildResult> blasBuildResultVector;
        blasBuildResultVector.reserve(instanceCount);

        for (size_t i = 0; i < instanceCount; i++) {
            Instance & instance = pin_instances[i];
            //设置实例对应的BLAS下标
            instance.asIndex = blasBuildResultVector.size();

            switch (instance.primitiveType) {
                case PrimitiveType::SPHERE:
                    //设置实例的变换前包围盒和几何中心
                    instance.setBoundingBoxProperties(spheres[instance.primitiveIndex].constructBoundingBox(), spheres[instance.primitiveIndex].centroid());
                    blasBuildResultVector.push_back(BLAS::constructBLAS(
                            spheres, instance.primitiveIndex, 1,
                            parallelograms, 0, 0));
                    break;
                case PrimitiveType::PARALLELOGRAM:
                    instance.setBoundingBoxProperties(parallelograms[instance.primitiveIndex].constructBoundingBox(), parallelograms[instance.primitiveIndex].centroid());
                    blasBuildResultVector.push_back(BLAS::constructBLAS(
                            spheres, 0, 0,
                            parallelograms, instance.primitiveIndex, 1));
                    break;
                default:;
            }
        }

        //构建TLAS
        SDL_Log("Building TLAS...");
        const TLASBuildResult tlasBuildResult = TLAS::constructTLAS(pin_instances, instanceCount);

        return {tlasBuildResult, blasBuildResultVector};
    }

    ASTraverseData Renderer::copyAccelerationStructureToGlobalMemory(const ASBuildResult & asBuildResult, const Instance * pin_instances, size_t instanceCount) {
        ASTraverseData ret{};
        SDL_Log("Copying AS to global memory...");

        //拷贝实例数组
        cudaCheckError(cudaMalloc(&ret.instances, instanceCount * sizeof(Instance)));
        cudaCheckError(cudaMemcpy(ret.instances, pin_instances, instanceCount * sizeof(Instance), cudaMemcpyHostToDevice));

        //拷贝TLAS
        const auto tlasNodeArray = asBuildResult.tlas.first;
        const size_t tlasNodeArrayLength = tlasNodeArray.size();
        const auto tlasIndexArray = asBuildResult.tlas.second;
        const size_t tlasIndexArrayLength = tlasIndexArray.size();
        //拷贝Node数组
        cudaCheckError(cudaMalloc(&ret.tlasArray.first.first, tlasNodeArrayLength * sizeof(TLASNode)));
        cudaCheckError(cudaMemcpy(ret.tlasArray.first.first, tlasNodeArray.data(), tlasNodeArrayLength * sizeof(TLASNode), cudaMemcpyHostToDevice));
        //拷贝Index数组
        cudaCheckError(cudaMalloc(&ret.tlasArray.second.first, tlasIndexArrayLength * sizeof(TLASIndex)));
        cudaCheckError(cudaMemcpy(ret.tlasArray.second.first, tlasIndexArray.data(), tlasIndexArrayLength * sizeof(TLASIndex), cudaMemcpyHostToDevice));
        //赋值数组长度
        ret.tlasArray.first.second = tlasNodeArrayLength;
        ret.tlasArray.second.second = tlasIndexArrayLength;

        //拷贝BLAS
        const auto blasVector = asBuildResult.blasVector;
        const size_t blasCount = blasVector.size();
        //分配临时指针数组，BLASArray本身作为指针
        auto tempBlasArray = new BLASArray [blasCount];
        //逐个拷贝BLAS
        for (size_t i = 0; i < blasCount; i++) {
            const auto & blas = blasVector[i];
            const auto & blasNodeArray = blas.first;
            const size_t blasNodeArrayLength = blasNodeArray.size();
            const auto & blasIndexArray = blas.second;
            const size_t blasIndexArrayLength = blasIndexArray.size();

            //拷贝Node数组
            cudaCheckError(cudaMalloc(&tempBlasArray[i].first.first, blasNodeArrayLength * sizeof(BLASNode)));
            cudaCheckError(cudaMemcpy(tempBlasArray[i].first.first, blasNodeArray.data(), blasNodeArrayLength * sizeof(BLASNode), cudaMemcpyHostToDevice));

            //拷贝Index数组
            cudaCheckError(cudaMalloc(&tempBlasArray[i].second.first, blasIndexArrayLength * sizeof(BLASIndex)));
            cudaCheckError(cudaMemcpy(tempBlasArray[i].second.first, blasIndexArray.data(), blasIndexArrayLength * sizeof(BLASIndex), cudaMemcpyHostToDevice));

            //赋值数组长度
            tempBlasArray[i].first.second = blasNodeArrayLength;
            tempBlasArray[i].second.second = blasIndexArrayLength;
        }
        //将临时指针数组的指针和长度拷贝到设备
        cudaCheckError(cudaMalloc(&ret.blasArray, blasCount * sizeof(BLASArray)));
        cudaCheckError(cudaMemcpy(ret.blasArray, tempBlasArray, blasCount * sizeof(BLASArray), cudaMemcpyHostToDevice));
        ret.blasArrayCount = blasCount;
        delete[] tempBlasArray;

        SDL_Log("AS copied to global memory.");
        return ret;
    }

    void Renderer::freeAccelerationStructureGlobalMemory(ASTraverseData &asTraverseData) {
        //释放实例数组
        cudaCheckError(cudaFree(asTraverseData.instances));

        //释放TLAS
        cudaCheckError(cudaFree(asTraverseData.tlasArray.first.first));
        cudaCheckError(cudaFree(asTraverseData.tlasArray.second.first));

        //1. 在主机端创建一个临时数组来接收从设备传回的BLAS指针数组
        auto tempBlasArray = new BLASArray [asTraverseData.blasArrayCount];
        //2. 将设备端的BLAS指针数组拷贝回主机端的临时数组
        cudaCheckError(cudaMemcpy(tempBlasArray, asTraverseData.blasArray, asTraverseData.blasArrayCount * sizeof(BLASArray), cudaMemcpyDeviceToHost));

        //3. 遍历主机端的临时数组，释放其中存储的每一个设备指针
        for (size_t i = 0; i < asTraverseData.blasArrayCount; i++) {
            // 释放Node数组
            cudaCheckError(cudaFree(tempBlasArray[i].first.first));
            // 释放Index数组
            cudaCheckError(cudaFree(tempBlasArray[i].second.first));
        }
        //4. 释放设备端的BLAS指针数组本身
        cudaCheckError(cudaFree(asTraverseData.blasArray));
        //5. 释放主机端的临时数组
        delete[] tempBlasArray;

        SDL_Log("AS global memory freed");
    }

    void Renderer::renderLoop(
            const SceneGeometryData & geometryDataWithDevPtr, const SceneMaterialData & materialDataWithDevPtr,
            Camera * pin_camera, const ASTraverseData & asTraverseData)
    {
        SDL_Log("Starting render loop.");
        const int w = pin_camera->windowWidth;
        const int h = pin_camera->windowHeight;

        //SDL
        SDL_Window * window = SDL_CreateWindow(
                "Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                w, h, SDL_WINDOW_OPENGL);
        SDL_GLContext context = SDL_GL_CreateContext(window);

        //禁用垂直同步
        SDL_GL_SetSwapInterval(0);

        //OGL
        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
            SDL_Log("Failed to init GLAD!"); return;
        }
        glViewport(0, 0, w, h);
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        constexpr float vertices[] = {
                -1.0f,  1.0f,  0.0f, 1.0f,
                -1.0f, -1.0f,  0.0f, 0.0f,
                1.0f, -1.0f,  1.0f, 0.0f,
                1.0f,  1.0f,  1.0f, 1.0f
        };
        constexpr GLuint indices[] = {
                0, 1, 2,
                0, 2, 3
        };
        GLuint VAO, VBO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        constexpr const char * vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        constexpr const char * fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
        )";
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        //CUDA
        cudaGraphicsResource_t cudaResource;
        cudaCheckError(cudaGraphicsGLRegisterImage(
                &cudaResource, textureID, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard));

        // ====== 启动参数 ======
        //将结构体本身拷贝到全局内存
        SceneGeometryData * dev_geometryData;
        SceneMaterialData * dev_materialData;
        cudaCheckError(cudaMalloc(&dev_geometryData, sizeof(SceneGeometryData)));
        cudaCheckError(cudaMalloc(&dev_materialData, sizeof(SceneMaterialData)));
        cudaCheckError(cudaMemcpyAsync(dev_geometryData, &geometryDataWithDevPtr, sizeof(SceneGeometryData), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(dev_materialData, &materialDataWithDevPtr, sizeof(SceneMaterialData), cudaMemcpyHostToDevice));

        ASTraverseData * dev_asTraverseData;
        cudaCheckError(cudaMalloc(&dev_asTraverseData, sizeof(ASTraverseData)));
        cudaCheckError(cudaMemcpy(dev_asTraverseData, &asTraverseData, sizeof(ASTraverseData), cudaMemcpyHostToDevice));

        const dim3 blocks(w % 16 == 0 ? w / 16 : w / 16 + 1,
                          h % 16 == 0 ? h / 16 : h / 16 + 1, 1);
        const dim3 threads(16, 16, 1);

        // ====== 渲染循环 ======
        bool quit = false;
        bool isReceiveInput = true;
        SDL_Event event;

        SDL_SetRelativeMouseMode(SDL_TRUE);
        bool key_w_pressed = false;
        bool key_a_pressed = false;
        bool key_s_pressed = false;
        bool key_d_pressed = false;
        bool key_space_pressed = false;
        bool key_lshift_pressed = false;

        constexpr double MOUSE_SENSITIVITY = 0.001;
        constexpr double PITCH_LIMIT_RADIAN = PI / 2.1;
        constexpr double MOVE_SPEED = 0.1;

        constexpr double TARGET_FPS = 70.0;
        constexpr auto TARGET_FRAME_DURATION = std::chrono::microseconds(static_cast<Sint64>(1000000.0 / TARGET_FPS));
        constexpr auto SLEEP_MARGIN = std::chrono::milliseconds(2);

        std::chrono::time_point<std::chrono::steady_clock> frameStartTime;

        while (!quit) {
            //帧开始时间点
            frameStartTime = std::chrono::steady_clock::now();

            int dx = 0;
            int dy = 0;
            Point3 newCameraCenter = pin_camera->cameraCenter;
            Point3 newCameraTarget = pin_camera->cameraTarget;

            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = true;
                    break;
                }
                if (event.type == SDL_KEYDOWN) {
                    switch (event.key.keysym.sym) {
                        case SDLK_w: key_w_pressed = true; break;
                        case SDLK_a: key_a_pressed = true; break;
                        case SDLK_s: key_s_pressed = true; break;
                        case SDLK_d: key_d_pressed = true; break;
                        case SDLK_SPACE: key_space_pressed = true; break;
                        case SDLK_LSHIFT: key_lshift_pressed = true; break;
                    }
                }
                if (event.type == SDL_KEYUP) {
                    switch (event.key.keysym.sym) {
                        case SDLK_w: key_w_pressed = false; break;
                        case SDLK_a: key_a_pressed = false; break;
                        case SDLK_s: key_s_pressed = false; break;
                        case SDLK_d: key_d_pressed = false; break;
                        case SDLK_SPACE: key_space_pressed = false; break;
                        case SDLK_LSHIFT: key_lshift_pressed = false; break;
                    }
                }
                if (event.type == SDL_MOUSEBUTTONDOWN) {
                    SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() == SDL_TRUE ? SDL_FALSE : SDL_TRUE);
                    isReceiveInput = !isReceiveInput;
                }
                if (event.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode() == SDL_TRUE) {
                    dx += event.motion.xrel;
                    dy += event.motion.yrel;
                }
            }
            if (quit) break;

            //鼠标移动
            if (dx != 0 || dy != 0) {
                const Vec3 viewDirection = Point3::constructVector(pin_camera->cameraCenter, pin_camera->cameraTarget);

                //获取当前相机的方向向量
                Vec3 W = pin_camera->cameraW.unitVector();
                Vec3 U = pin_camera->cameraU.unitVector();
                Vec3 V = pin_camera->cameraV.unitVector();

                //左右旋转 (Yaw)
                //将视线向量(W) 绕着上方向量(V) 进行旋转，实现视角左右旋转
                const double yawAngle = -dx * MOUSE_SENSITIVITY;
                W = W.rotate(V, yawAngle);

                //上下旋转 (Pitch)
                //将已经左右旋转过的视线向量(W) 绕着右方向量(U) 进行旋转
                double pitchAngle = -dy * MOUSE_SENSITIVITY;
                W = W.rotate(U, pitchAngle);

                //限制俯仰角超过限制
                //从已经左右旋转过的W向量中获取当前俯仰角
                //W向量的Y分量是俯仰角(pitch)的正弦值，所以可以用asin来获取
                double newPitch = std::asin(W[1]);

                bool needsCorrection = false;
                if (newPitch > PITCH_LIMIT_RADIAN) {
                    newPitch = PITCH_LIMIT_RADIAN;
                    needsCorrection = true;
                } else if (newPitch < -PITCH_LIMIT_RADIAN) {
                    newPitch = -PITCH_LIMIT_RADIAN;
                    needsCorrection = true;
                }
                //如果超限了，就根据限制角度重新构建W向量
                if (needsCorrection) {
                    //获取水平方向
                    const Vec3 horizontalDir = Vec3{W[0], 0.0, W[2]}.unitVector();
                    //用被钳制后的俯仰角 newPitch 重新计算W
                    const double horizontalMagnitude = std::cos(newPitch);
                    W = horizontalDir * horizontalMagnitude + Vec3{0.0, std::sin(newPitch), 0.0};
                }
                //更新目标点。鼠标移动只改变看向的位置
                newCameraTarget = newCameraCenter + W * viewDirection.length();
            }

            //键盘按键
            Vec3 movementDirection{};

            //使得键盘按键总是在水平平面上移动，移除方向向量的竖直分量（取平面投影）
            const Vec3 forwardHorizontal = Vec3{pin_camera->cameraW[0], 0.0, pin_camera->cameraW[2]}.unitVector();
            if (key_w_pressed) movementDirection += forwardHorizontal; // Forward
            if (key_s_pressed) movementDirection -= forwardHorizontal; // Backward
            if (key_d_pressed) movementDirection += pin_camera->cameraU; // Right (Strafe)
            if (key_a_pressed) movementDirection -= pin_camera->cameraU; // Left (Strafe)

            //上下移动
            if (key_space_pressed) movementDirection += pin_camera->upDirection;  // Up
            if (key_lshift_pressed) movementDirection -= pin_camera->upDirection; // Down

            if (movementDirection.lengthSquared() > 0.0) {
                //将移动方向向量的长度变为1。这确保了斜向移动（例如同时按W和D）的速度和直线移动的速度一致，避免了“斜走更快”的问题
                const Vec3 translation = movementDirection.unitVector() * MOVE_SPEED;
                newCameraCenter += translation;
                newCameraTarget += translation;
            }

            //更新相机
            if (isReceiveInput) {
                updateCameraProperties(pin_camera, newCameraCenter, newCameraTarget);
            }

            //a. 映射资源，让CUDA接管纹理
            cudaCheckError(cudaGraphicsMapResources(1, &cudaResource, nullptr));
            //b. 获取指向纹理的CUDA数组
            cudaArray_t cudaTextureArray;
            cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaResource, 0, 0));
            //c. 为CUDA数组创建一个 Surface Object，以便核函数写入
            cudaResourceDesc resDesc {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cudaTextureArray;
            cudaSurfaceObject_t surfaceObject;
            cudaCheckError(cudaCreateSurfaceObject(&surfaceObject, &resDesc));

            //d. 启动核函数
            render<<<blocks, threads>>>(dev_geometryData, dev_materialData, surfaceObject, dev_asTraverseData);
            cudaCheckError(cudaDeviceSynchronize());

            //e. 销毁 Surface Object
            cudaCheckError(cudaDestroySurfaceObject(surfaceObject));
            //f. 解除映射，将纹理控制权还给OpenGL
            cudaCheckError(cudaGraphicsUnmapResources(1, &cudaResource, nullptr));
            //g. OpenGL渲染并显示到SDL窗口
            glUseProgram(shaderProgram);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
            SDL_GL_SwapWindow(window);

            //计算本帧耗时
            const auto frameEndTime = std::chrono::steady_clock::now();
            const auto frameTime = std::chrono::duration_cast<std::chrono::microseconds>(frameEndTime - frameStartTime);
            const auto workTime = std::chrono::steady_clock::now() - frameStartTime;

            //如果本帧工作时间小于目标帧时长，则需要等待
            if (workTime < TARGET_FRAME_DURATION) {
                const auto timeToWait = TARGET_FRAME_DURATION - workTime;

                //1. 粗略休眠
                //如果需要等待的时间大于设定的安全边界，就先 sleep
                if (timeToWait > SLEEP_MARGIN) {
                    std::this_thread::sleep_for(timeToWait - SLEEP_MARGIN);
                }

                //2. 精确自旋
                //在最后一点时间里，不断查询时间，直到达到目标
                while (std::chrono::steady_clock::now() - frameStartTime < TARGET_FRAME_DURATION) {}
            }
        }

        //释放参数结构体
        cudaCheckError(cudaFree(dev_geometryData));
        cudaCheckError(cudaFree(dev_materialData));
        cudaCheckError(cudaFree(dev_asTraverseData));

        // ====== 清理资源 ======
        //~CUDA
        cudaCheckError(cudaGraphicsUnregisterResource(cudaResource));

        //~OGL
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
        glDeleteTextures(1, &textureID);

        //~SDL
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Log("Render completed.");
    }

    void Renderer::constructCamera(Camera * pin_camera) {
        Camera & cam = *pin_camera;

        cam.focusDistance = cam.cameraCenter.distance(cam.cameraTarget);
        const double thetaFOV = MathHelper::degreeToRadian(cam.fov);
        const double vWidth = 2.0 * tan(thetaFOV / 2.0) * cam.focusDistance;
        const double vHeight = vWidth / (cam.windowWidth * 1.0 / cam.windowHeight);

        cam.viewPortWidth = vWidth;
        cam.viewPortHeight = vHeight;
        cam.cameraW = Point3::constructVector(cam.cameraCenter, cam.cameraTarget).unitVector();
        cam.cameraU = Vec3::cross(cam.cameraW, cam.upDirection).unitVector();
        cam.cameraV = Vec3::cross(cam.cameraU, cam.cameraW).unitVector();

        cam.viewPortX = vWidth * cam.cameraU;
        cam.viewPortY = vHeight * cam.cameraV;
        cam.viewPortPixelDx = cam.viewPortX / cam.windowWidth;
        cam.viewPortPixelDy = cam.viewPortY / cam.windowHeight;
        cam.viewPortOrigin = cam.cameraCenter + cam.focusDistance * cam.cameraW - cam.viewPortX * 0.5 - cam.viewPortY * 0.5;
        cam.pixelOrigin = cam.viewPortOrigin + cam.viewPortPixelDx * 0.5 + cam.viewPortPixelDy * 0.5;
        cam.sqrtSampleCount = static_cast<size_t>(std::sqrt(cam.sampleCount));
        cam.reciprocalSqrtSampleCount = 1.0 / static_cast<double>(cam.sqrtSampleCount);
    }

    void Renderer::updateCameraProperties(Camera * pin_camera, const Point3 & newCenter, const Point3 & newTarget) {
        //重新计算相机数据
        pin_camera->cameraCenter = newCenter;
        pin_camera->cameraTarget = newTarget;
        constructCamera(pin_camera);

        //拷贝到常量内存
        cudaCheckError(cudaMemcpyToSymbol(dev_camera, pin_camera, sizeof(Camera)));
    }
}