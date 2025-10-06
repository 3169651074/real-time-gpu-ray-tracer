#include <Global/Render.cuh>

namespace renderer {
    extern __global__ void render(const TraverseData * dev_traverseData, cudaSurfaceObject_t surfaceObject);

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
    //geometryData和materialData变量本身本质上是指针，不存放在页面锁定内存中
    void Renderer::allocSceneDataPinnedMemory(
            SceneGeometryData & geometryData, SceneMaterialData & materialData, Camera * & pin_camera, Instance * & pin_instances, size_t instanceCount)
    {
        SDL_Log("Allocating pinned memory for scene data ...");
        SDL_Log("Sphere count: %zd; Parallelogram count: %zd", geometryData.sphereCount, geometryData.parallelogramCount);
        SDL_Log("Rough material count: %zd; Metal material count: %zd", materialData.roughCount, materialData.metalCount);
        SDL_Log("Instance count: %zd", instanceCount);

        //分配几何体和材质内存
        _mallocHost(geometryData, Sphere, spheres, sphereCount);
        _mallocHost(geometryData, Parallelogram, parallelograms, parallelogramCount);

        _mallocHost(materialData, Rough, roughs, roughCount);
        _mallocHost(materialData, Metal, metals, metalCount);

        //分配相机内存
        SDL_Log("Allocating pinned memory for camera ...");
        cudaCheckError(cudaHostAlloc(&pin_camera, sizeof(Camera), cudaHostAllocDefault));

        //分配实例内存
        SDL_Log("Allocating pinned memory for instance ...");
        cudaCheckError(cudaHostAlloc(&pin_instances, instanceCount * sizeof(Instance), cudaHostAllocDefault));

        SDL_Log("Scene data pinned memory allocated.");
    }

    //释放页面锁定内存
#define _freeHost(structName, arrayName) cudaCheckError(cudaFreeHost(structName.arrayName))
    void Renderer::freeSceneDataPinnedMemory(
            SceneGeometryData & geometryData, SceneMaterialData & materialData,
            Camera * & pin_camera, Instance * & pin_instances)
    {
        SDL_Log("Freeing pinned memory for scene data ...");

        _freeHost(geometryData, spheres);
        _freeHost(geometryData, parallelograms);

        _freeHost(materialData, roughs);
        _freeHost(materialData, metals);

        //释放相机空间
        SDL_Log("Freeing pinned memory for camera ...");
        cudaCheckError(cudaFreeHost(pin_camera));

        //释放实例空间
        SDL_Log("Freeing pinned memory for instances...");
        cudaCheckError(cudaFreeHost(pin_instances));

        SDL_Log("Scene data pinned memory freed.");
    }

    //分配全局内存并拷贝数据
#define _mallocGlobalAndCopy(srcStructName, dstStructName, arrayName, countName, className)\
        do {                                        \
            if (srcStructName.countName != 0) {     \
                cudaCheckError(cudaMalloc(&dstStructName.arrayName, srcStructName.countName * sizeof(className)));\
                cudaCheckError(cudaMemcpy(dstStructName.arrayName, srcStructName.arrayName, srcStructName.countName * sizeof(className), cudaMemcpyHostToDevice));\
            }                                       \
        } while (false)
    Pair<SceneGeometryData, SceneMaterialData> Renderer::copySceneDataToGlobalMemory(
            const SceneGeometryData & geometryData, const SceneMaterialData & materialData,
            const Camera * pin_camera)
    {
        SDL_Log("Allocating and copying scene data to global memory...");

        //拷贝数组长度信息
        SceneGeometryData geometryDataWithDevPtr = geometryData;
        SceneMaterialData materialDataWithDevPtr = materialData;

        _mallocGlobalAndCopy(geometryData, geometryDataWithDevPtr, spheres, sphereCount, Sphere);
        _mallocGlobalAndCopy(geometryData, geometryDataWithDevPtr, parallelograms, parallelogramCount, Parallelogram);

        _mallocGlobalAndCopy(materialData, materialDataWithDevPtr, roughs, roughCount, Rough);
        _mallocGlobalAndCopy(materialData, materialDataWithDevPtr, metals, metalCount, Metal);

        //拷贝相机到常量内存
        SDL_Log("Copying camera to constant memory...");
        cudaCheckError(cudaMemcpyToSymbol(dev_camera, pin_camera, sizeof(Camera)));

        SDL_Log("Global memory for scene data allocated and data copied.");
        return {geometryDataWithDevPtr, materialDataWithDevPtr};
    }

    //释放全局内存，无需释放常量内存
#define _freeGlobal(structName, arrayName) cudaCheckError(cudaFree(structName.arrayName))
    void Renderer::freeSceneDataGlobalMemory(
            SceneGeometryData & geometryDataWithDevPtr, SceneMaterialData & materialDataWithDevPtr)
    {
        SDL_Log("Freeing global memory for scene data...");

        _freeGlobal(geometryDataWithDevPtr, spheres);
        _freeGlobal(geometryDataWithDevPtr, parallelograms);

        _freeGlobal(materialDataWithDevPtr, roughs);
        _freeGlobal(materialDataWithDevPtr, metals);

        SDL_Log("Global memory for scene data freed");
    }

    ASBuildResult Renderer::buildAccelerationStructure(
            const SceneGeometryData & geometryDataWithPinPtr, Instance * pin_instances, size_t instanceCount)
    {
        SDL_Log("Building acceleration structure...");
        const Sphere * spheres = geometryDataWithPinPtr.spheres;
        const Parallelogram * parallelograms = geometryDataWithPinPtr.parallelograms;

        //构建BLAS：为每个物体创建独立的BLAS，存储到数组中
        //当前一个实例对应一个BLAS
        std::vector<BLASBuildResult> blasBuildResultVector;
        blasBuildResultVector.reserve(instanceCount);

        for (size_t i = 0; i < instanceCount; i++) {
            SDL_Log("Building BLAS for instance [%zd]...", i);

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

            SDL_Log("BLAS for instance [%zd]: Node array length: %zd, index array length: %zd", i, blasBuildResultVector[i].first.size(), blasBuildResultVector[i].second.size());
        }

        //构建TLAS
        SDL_Log("Building TLAS...");
        const TLASBuildResult tlasBuildResult = TLAS::constructTLAS(pin_instances, instanceCount);
        SDL_Log("TLAS node array length: %zd, index array length: %zd", tlasBuildResult.first.size(), tlasBuildResult.second.size());
        SDL_Log("Acceleration structure constructed.");

        //将加速结构拷贝到页面锁定内存
        SDL_Log("Allocating pinned memory for acceleration structure...");
        ASBuildResult result;

        SDL_Log("Copying TLAS data...");
        const auto & tlasNodeArray = tlasBuildResult.first.data();
        const size_t tlasNodeArrayLength = tlasBuildResult.first.size();
        const auto & tlasIndexArray = tlasBuildResult.second.data();
        const size_t tlasIndexArrayLength = tlasBuildResult.second.size();

        //Node
        cudaCheckError(cudaHostAlloc(&result.pin_tlasArray.first.first, tlasNodeArrayLength * sizeof(TLASNode), cudaHostAllocDefault));
        memcpy(result.pin_tlasArray.first.first, tlasNodeArray, tlasNodeArrayLength * sizeof(TLASNode));
        result.pin_tlasArray.first.second = tlasNodeArrayLength;

        //Index
        cudaCheckError(cudaHostAlloc(&result.pin_tlasArray.second.first, tlasIndexArrayLength * sizeof(TLASIndex), cudaHostAllocDefault));
        memcpy(result.pin_tlasArray.second.first, tlasIndexArray, tlasIndexArrayLength * sizeof(TLASIndex));
        result.pin_tlasArray.second.second = tlasIndexArrayLength;

        SDL_Log("Copying BLAS data...");
        //BLAS数量为实例数量
        //分配指针数组内存
        cudaCheckError(cudaHostAlloc(&result.pin_blasArray, instanceCount * sizeof(BLASArray), cudaHostAllocDefault));

        //逐个BLAS拷贝数据
        for (size_t i = 0; i < instanceCount; i++) {
            const auto & blasNodeArray = blasBuildResultVector[i].first.data();
            const size_t blasNodeArrayLength = blasBuildResultVector[i].first.size();
            const auto & blasIndexArray = blasBuildResultVector[i].second.data();
            const size_t blasIndexArrayLength = blasBuildResultVector[i].second.size();

            //Node
            cudaCheckError(cudaHostAlloc(&result.pin_blasArray[i].first.first, blasNodeArrayLength * sizeof(BLASNode), cudaHostAllocDefault));
            memcpy(result.pin_blasArray[i].first.first, blasNodeArray, blasNodeArrayLength * sizeof(BLASNode));
            result.pin_blasArray[i].first.second = blasNodeArrayLength;

            //Index
            cudaCheckError(cudaHostAlloc(&result.pin_blasArray[i].second.first, blasIndexArrayLength * sizeof(BLASIndex), cudaHostAllocDefault));
            memcpy(result.pin_blasArray[i].second.first, blasIndexArray, blasIndexArrayLength * sizeof(BLASIndex));
            result.pin_blasArray[i].second.second = blasIndexArrayLength;

            SDL_Log("Copying BLAS [%zd] to pinned memory...", i);
        }
        result.blasArrayCount = instanceCount;
        SDL_Log("Acceleration structure copied to pinned memory.");

        return result;
    }

    void Renderer::freeAccelerationStructurePinnedMemory(ASBuildResult & asBuildResultWithPinPtr) {
        SDL_Log("Freeing pinned memory for acceleration structure...");

        //TLAS
        cudaCheckError(cudaFreeHost(asBuildResultWithPinPtr.pin_tlasArray.first.first));
        cudaCheckError(cudaFreeHost(asBuildResultWithPinPtr.pin_tlasArray.second.first));

        //BLAS
        for (size_t i = 0; i < asBuildResultWithPinPtr.blasArrayCount; i++) {
            cudaCheckError(cudaFreeHost(asBuildResultWithPinPtr.pin_blasArray[i].first.first));
            cudaCheckError(cudaFreeHost(asBuildResultWithPinPtr.pin_blasArray[i].second.first));
        }
        SDL_Log("Pinned memory for acceleration structure freed.");
    }

    ASTraverseData Renderer::copyAccelerationStructureToGlobalMemory(const ASBuildResult & asBuildResult, const Instance * pin_instances, size_t instanceCount) {
        SDL_Log("Allocating and copying instance data and acceleration structure data to global memory...");
        ASTraverseData ret{};

        //拷贝实例数组
        SDL_Log("Copying instance data...");
        cudaCheckError(cudaMalloc(&ret.instances, instanceCount * sizeof(Instance)));
        cudaCheckError(cudaMemcpy(ret.instances, pin_instances, instanceCount * sizeof(Instance), cudaMemcpyHostToDevice));

        //拷贝TLAS
        SDL_Log("Copying TLAS data...");
        const auto & tlasNodeArray = asBuildResult.pin_tlasArray.first.first;
        const size_t tlasNodeArrayLength = asBuildResult.pin_tlasArray.first.second;
        const auto & tlasIndexArray = asBuildResult.pin_tlasArray.second.first;
        const size_t tlasIndexArrayLength = asBuildResult.pin_tlasArray.second.second;
        //Node
        cudaCheckError(cudaMalloc(&ret.tlasArray.first.first, tlasNodeArrayLength * sizeof(TLASNode)));
        cudaCheckError(cudaMemcpy(ret.tlasArray.first.first, tlasNodeArray, tlasNodeArrayLength * sizeof(TLASNode), cudaMemcpyHostToDevice));
        ret.tlasArray.first.second = tlasNodeArrayLength;

        //Index
        cudaCheckError(cudaMalloc(&ret.tlasArray.second.first, tlasIndexArrayLength * sizeof(TLASIndex)));
        cudaCheckError(cudaMemcpy(ret.tlasArray.second.first, tlasIndexArray, tlasIndexArrayLength * sizeof(TLASIndex), cudaMemcpyHostToDevice));
        ret.tlasArray.second.second = tlasIndexArrayLength;

        //拷贝BLAS
        SDL_Log("Copying BLAS data...");

        const auto & blasVector = asBuildResult.pin_blasArray;
        const size_t blasCount = asBuildResult.blasArrayCount;
        //分配临时指针数组，BLASArray本身作为指针
        auto tempBlasArray = new BLASArray [blasCount];
        //逐个拷贝BLAS
        for (size_t i = 0; i < blasCount; i++) {
            const auto & blas = blasVector[i];
            const auto & blasNodeArray = blas.first.first;
            const size_t blasNodeArrayLength = blas.first.second;
            const auto & blasIndexArray = blas.second.first;
            const size_t blasIndexArrayLength = blas.second.second;

            //Node
            cudaCheckError(cudaMalloc(&tempBlasArray[i].first.first, blasNodeArrayLength * sizeof(BLASNode)));
            cudaCheckError(cudaMemcpy(tempBlasArray[i].first.first, blasNodeArray, blasNodeArrayLength * sizeof(BLASNode), cudaMemcpyHostToDevice));
            tempBlasArray[i].first.second = blasNodeArrayLength;

            //Index
            cudaCheckError(cudaMalloc(&tempBlasArray[i].second.first, blasIndexArrayLength * sizeof(BLASIndex)));
            cudaCheckError(cudaMemcpy(tempBlasArray[i].second.first, blasIndexArray, blasIndexArrayLength * sizeof(BLASIndex), cudaMemcpyHostToDevice));
            tempBlasArray[i].second.second = blasIndexArrayLength;

            SDL_Log("BLAS [%zd] copied.", i);
        }
        //将临时指针数组的指针和长度拷贝到设备
        cudaCheckError(cudaMalloc(&ret.blasArray, blasCount * sizeof(BLASArray)));
        cudaCheckError(cudaMemcpy(ret.blasArray, tempBlasArray, blasCount * sizeof(BLASArray), cudaMemcpyHostToDevice));
        ret.blasArrayCount = blasCount;
        delete[] tempBlasArray;

        SDL_Log("Instances and acceleration structure copied to global memory.");
        return ret;
    }

    void Renderer::freeAccelerationStructureGlobalMemory(ASTraverseData &asTraverseData) {
        //释放实例数组
        SDL_Log("Freeing global memory for instance...");
        cudaCheckError(cudaFree(asTraverseData.instances));

        //释放TLAS
        SDL_Log("Freeing global memory for TLAS...");
        cudaCheckError(cudaFree(asTraverseData.tlasArray.first.first));
        cudaCheckError(cudaFree(asTraverseData.tlasArray.second.first));

        SDL_Log("Freeing global memory for BLAS ...");
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

            SDL_Log("BLAS [%zd] freed.", i);
        }
        //4. 释放设备端的BLAS指针数组本身
        cudaCheckError(cudaFree(asTraverseData.blasArray));
        //5. 释放主机端的临时数组
        delete[] tempBlasArray;

        SDL_Log("Global memory for Acceleration structure freed");
    }

    void Renderer::renderLoop(const TraverseData & traverseData, Camera * pin_camera) {
        constexpr float MOUSE_SENSITIVITY = 0.001f;
        constexpr float PITCH_LIMIT_RADIAN = PI / 2.2f;
        constexpr float MOVE_SPEED = 0.1f;

        constexpr bool isRestrictFrameCount = true;
        //仅当isRestrictFrameCount为true时以下参数有效
        constexpr float TARGET_FPS = 60.0f;
        constexpr auto TARGET_FRAME_DURATION = std::chrono::microseconds(static_cast<Sint64>(1000000.0 / TARGET_FPS));
        constexpr auto SLEEP_MARGIN = std::chrono::milliseconds(2);

        SDL_Log("Creating window...");
        const int w = pin_camera->windowWidth;
        const int h = pin_camera->windowHeight;

        //SDL
        SDL_Window * window;
        SDL_CheckErrorPtr(window = SDL_CreateWindow(
                                   "Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                   w, h, SDL_WINDOW_OPENGL));
        SDL_GLContext context;
        SDL_CheckErrorPtr(context = SDL_GL_CreateContext(window));

        //禁用垂直同步
        SDL_CheckErrorInt(SDL_GL_SetSwapInterval(0));

        //OGL
        SDL_Log("Initializing OGL...");
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
        SDL_Log("Preparing temporary device data...");

        //将结构体本身拷贝到全局内存
        TraverseData * dev_traverseData;
        cudaCheckError(cudaMalloc(&dev_traverseData, sizeof(TraverseData)));
        cudaCheckError(cudaMemcpy(dev_traverseData, &traverseData, sizeof(TraverseData), cudaMemcpyHostToDevice));

        SDL_Log("Confirming kernel parameters...");
        const dim3 blocks(w % 16 == 0 ? w / 16 : w / 16 + 1,
                          h % 16 == 0 ? h / 16 : h / 16 + 1, 1);
        const dim3 threads(16, 16, 1);

        // ====== 渲染循环 ======
        bool isQuit = false;
        bool isReceiveInput = true;
        SDL_Event event;

        SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_TRUE));
        bool keyW = false;
        bool keyA = false;
        bool keyS = false;
        bool keyD = false;
        bool keySpace = false;
        bool keyLShift = false;
        
        std::chrono::time_point<std::chrono::steady_clock> frameStartTime;
        SDL_Log("Starting render loop...");
        
        while (!isQuit) {
            //帧开始时间点
            frameStartTime = std::chrono::steady_clock::now();

            //接收输入并更新输入变量
            int dx = 0;
            int dy = 0;
            Point3 newCameraCenter = pin_camera->cameraCenter;
            Point3 newCameraTarget = pin_camera->cameraTarget;

            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    isQuit = true; break;
                }
                if (event.type == SDL_KEYDOWN) {
                    switch (event.key.keysym.sym) {
                        case SDLK_w: keyW = true; break;
                        case SDLK_a: keyA = true; break;
                        case SDLK_s: keyS = true; break;
                        case SDLK_d: keyD = true; break;
                        case SDLK_SPACE: keySpace = true; break;
                        case SDLK_LSHIFT: keyLShift = true; break;
                        case SDLK_ESCAPE: isQuit = true; break;
                        default:;
                    }
                }
                if (event.type == SDL_KEYUP) {
                    switch (event.key.keysym.sym) {
                        case SDLK_w: keyW = false; break;
                        case SDLK_a: keyA = false; break;
                        case SDLK_s: keyS = false; break;
                        case SDLK_d: keyD = false; break;
                        case SDLK_SPACE: keySpace = false; break;
                        case SDLK_LSHIFT: keyLShift = false; break;
                        default:;
                    }
                }
                if (event.type == SDL_MOUSEBUTTONDOWN) {
                    SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() == SDL_TRUE ? SDL_FALSE : SDL_TRUE));
                    isReceiveInput = !isReceiveInput;
                }
                if (event.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode() == SDL_TRUE) {
                    dx += event.motion.xrel;
                    dy += event.motion.yrel;
                }
            }
            if (isQuit) break;
            
            //根据输入变量计算更新数据
            //鼠标移动
            if (dx != 0 || dy != 0) {
                const Vec3 viewDirection = Point3::constructVector(pin_camera->cameraCenter, pin_camera->cameraTarget);

                //获取当前相机的方向向量
                Vec3 W = pin_camera->cameraW.unitVector();
                Vec3 U = pin_camera->cameraU.unitVector();
                Vec3 V = pin_camera->cameraV.unitVector();

                //左右旋转 (Yaw)
                //将视线向量(W) 绕着上方向量(V) 进行旋转，实现视角左右旋转
                const float yawAngle = -static_cast<float>(dx) * MOUSE_SENSITIVITY;
                W = W.rotate(V, yawAngle);

                //上下旋转 (Pitch)
                //将已经左右旋转过的视线向量(W) 绕着右方向量(U) 进行旋转
                float pitchAngle = -static_cast<float>(dy) * MOUSE_SENSITIVITY;
                W = W.rotate(U, pitchAngle);

                //限制俯仰角超过限制
                //从已经左右旋转过的W向量中获取当前俯仰角
                //W向量的Y分量是俯仰角(pitch)的正弦值，所以可以用asin来获取
                float newPitch = std::asin(W[1]);

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
                    const Vec3 horizontalDir = Vec3{W[0], 0.0f, W[2]}.unitVector();
                    //用被钳制后的俯仰角 newPitch 重新计算W
                    const float horizontalMagnitude = std::cos(newPitch);
                    W = horizontalDir * horizontalMagnitude + Vec3{0.0f, std::sin(newPitch), 0.0f};
                }
                //更新目标点。鼠标移动只改变看向的位置
                newCameraTarget = newCameraCenter + W * viewDirection.length();
            }

            //键盘按键
            Vec3 movementDirection{};
            //使得键盘按键总是在水平平面上移动，移除方向向量的竖直分量（取平面投影）
            const Vec3 forwardHorizontal = Vec3{pin_camera->cameraW[0], 0.0f, pin_camera->cameraW[2]}.unitVector();
            if (keyW) movementDirection += forwardHorizontal; //Forward
            if (keyS) movementDirection -= forwardHorizontal; //Backward
            if (keyD) movementDirection += pin_camera->cameraU; //Right
            if (keyA) movementDirection -= pin_camera->cameraU; //Left
            //上下移动
            if (keySpace) movementDirection += pin_camera->upDirection;  //Up
            if (keyLShift) movementDirection -= pin_camera->upDirection; //Down

            if (movementDirection.lengthSquared() > 0.0f) {
                //将移动方向向量的长度变为1，确保斜向移动（例如同时按W和D）的速度和直线移动的速度一致，避免了“斜走更快”的问题
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
            render<<<blocks, threads>>>(dev_traverseData, surfaceObject);
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

            if (isRestrictFrameCount) {
                //计算本帧耗时
                const auto frameEndTime = std::chrono::steady_clock::now();
                const auto frameTime = std::chrono::duration_cast<std::chrono::microseconds>(frameEndTime - frameStartTime);
                const auto workTime = std::chrono::steady_clock::now() - frameStartTime;

                //如果本帧工作时间小于目标帧时长，则需要等待
                if (workTime < TARGET_FRAME_DURATION) {
                    const auto timeToWait = TARGET_FRAME_DURATION - workTime;
                    //1. 粗略休眠
                    if (timeToWait > SLEEP_MARGIN) {
                        std::this_thread::sleep_for(timeToWait - SLEEP_MARGIN);
                    }
                    //2. 精确自旋
                    while (std::chrono::steady_clock::now() - frameStartTime < TARGET_FRAME_DURATION) {}
                }
            }
        }
        SDL_Log("Render stopped.");

        //释放参数结构体
        SDL_Log("Freeing temporary device memory...");
        cudaCheckError(cudaFree(dev_traverseData));

        // ====== 清理资源 ======
        //~CUDA
        cudaCheckError(cudaGraphicsUnregisterResource(cudaResource));

        //~OGL
        SDL_Log("Cleaning up OGL...");
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
        glDeleteTextures(1, &textureID);

        //~SDL
        SDL_Log("Cleaning up SDL resources...");
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
    }

    void Renderer::constructCamera(Camera * pin_camera) {
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

    void Renderer::updateCameraProperties(Camera * pin_camera, const Point3 & newCenter, const Point3 & newTarget) {
        //重新计算相机数据
        pin_camera->cameraCenter = newCenter;
        pin_camera->cameraTarget = newTarget;
        constructCamera(pin_camera);

        //拷贝到常量内存
        cudaCheckError(cudaMemcpyToSymbol(dev_camera, pin_camera, sizeof(Camera)));
    }

    TraverseData Renderer::castToRestrictDevPtr(
            const SceneGeometryData & geometryDataWithDevPtr, const SceneMaterialData & materialDataWithDevPtr,
            const ASTraverseData & asTraverseDataWithDevPtr)
    {
        return {
            .dev_spheres = geometryDataWithDevPtr.spheres,
            .dev_parallelograms = geometryDataWithDevPtr.parallelograms,

            .dev_roughs = materialDataWithDevPtr.roughs,
            .dev_metals = materialDataWithDevPtr.metals,

            .dev_instances = asTraverseDataWithDevPtr.instances,

            .tlasNodeArray = asTraverseDataWithDevPtr.tlasArray.first.first,
            .tlasIndexArray = asTraverseDataWithDevPtr.tlasArray.second.first,

            .blasArray = asTraverseDataWithDevPtr.blasArray
        };
    }
}