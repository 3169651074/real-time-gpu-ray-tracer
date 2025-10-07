#include <Global/SDL_OpenGLWindow.cuh>

namespace renderer {

    Pair<SDL_Window *, SDL_GLContext> SDL_OpenGLWindow::createSDLGLWindow(const char * title, int w, int h, bool isVsync) {
        SDL_Log("Creating SDL window...");
        SDL_Window * window;
        SDL_CheckErrorPtr(window = SDL_CreateWindow(
                "Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                w, h, SDL_WINDOW_OPENGL));
        SDL_GLContext context;
        SDL_CheckErrorPtr(context = SDL_GL_CreateContext(window));
        SDL_CheckErrorInt(SDL_GL_SetSwapInterval(isVsync ? 1 : 0));

        return {window, context};
    }
    void SDL_OpenGLWindow::destroySDLGLWindow(SDL_Window * window, SDL_GLContext context) {
        SDL_Log("Cleaning up SDL resources...");
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
    }

    SDL_OpenGLWindow::OGLArgs SDL_OpenGLWindow::initializeOGL(SDL_Window * window) {
        SDL_Log("Initializing OGL...");
        int w, h;
        SDL_GetWindowSize(window, &w, &h);

        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
            SDL_Log("Failed to init GLAD!"); return {};
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
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));
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

        return {
                .textureID = textureID,
                .VAO = VAO,
                .VBO = VBO,
                .EBO = EBO,
                .shaderProgram = shaderProgram
        };
    }
    void SDL_OpenGLWindow::releaseOGL(SDL_OpenGLWindow::OGLArgs & oglArgs) {
        SDL_Log("Cleaning up OGL...");
        glDeleteVertexArrays(1, &oglArgs.VAO);
        glDeleteBuffers(1, &oglArgs.VBO);
        glDeleteBuffers(1, &oglArgs.EBO);
        glDeleteProgram(oglArgs.shaderProgram);
        glDeleteTextures(1, &oglArgs.textureID);

        oglArgs = {};
    }

    SDL_OpenGLWindow::CUDAArgs SDL_OpenGLWindow::getCudaResource(SDL_Window * window, const OGLArgs & oglArgs, Uint32 threadsPerBlock) {
        SDL_Log("Getting Cuda resources...");
        int w, h;
        SDL_GetWindowSize(window, &w, &h);

        CUDAArgs cudaArgs;
        cudaArgs.blocks = {
                static_cast<Uint32>(w % threadsPerBlock == 0 ? w / threadsPerBlock : w / threadsPerBlock + 1),
                static_cast<Uint32>(h % threadsPerBlock == 0 ? h / threadsPerBlock : h / threadsPerBlock + 1), 1};
        cudaArgs.threads = {threadsPerBlock, threadsPerBlock, 1};

        cudaCheckError(cudaGraphicsGLRegisterImage(
                &cudaArgs.cudaResource, oglArgs.textureID, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard));
        return cudaArgs;
    }
    void SDL_OpenGLWindow::releaseCudaResource(CUDAArgs & cudaArgs) {
        SDL_Log("Releasing Cuda resources...");
        cudaCheckError(cudaGraphicsUnregisterResource(cudaArgs.cudaResource));
        cudaArgs = {};
    }

    void SDL_OpenGLWindow::getKeyMouseInput(KeyMouseInputArgs & inputArgs) {
        //鼠标移动变化量需要每帧重置
        inputArgs.dx = 0;
        inputArgs.dy = 0;
        inputArgs.mouseClick = false;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                inputArgs.keyQuit = true; break;
            }
            if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_w: inputArgs.keyW = true; break;
                    case SDLK_a: inputArgs.keyA = true; break;
                    case SDLK_s: inputArgs.keyS = true; break;
                    case SDLK_d: inputArgs.keyD = true; break;
                    case SDLK_SPACE: inputArgs.keySpace = true; break;
                    case SDLK_LSHIFT: inputArgs.keyLShift = true; break;
                    case SDLK_ESCAPE: inputArgs.keyQuit = true; break;
                    default:;
                }
            }
            if (event.type == SDL_KEYUP) {
                switch (event.key.keysym.sym) {
                    case SDLK_w: inputArgs.keyW = false; break;
                    case SDLK_a: inputArgs.keyA = false; break;
                    case SDLK_s: inputArgs.keyS = false; break;
                    case SDLK_d: inputArgs.keyD = false; break;
                    case SDLK_SPACE: inputArgs.keySpace = false; break;
                    case SDLK_LSHIFT: inputArgs.keyLShift = false; break;
                    default:;
                }
            }
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                inputArgs.mouseClick = true;
            }
            if (event.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode() == SDL_TRUE) {
                inputArgs.dx += event.motion.xrel;
                inputArgs.dy += event.motion.yrel;
            }
        }
    }
    Pair<bool, Pair<Point3, Point3>> SDL_OpenGLWindow::calculateNewPosition(
            const SDL_OpenGLWindow::OperateArgs & operateArgs,
            const SDL_OpenGLWindow::KeyMouseInputArgs & inputArgs,
            const Point3 & currentCenter, const Point3 & currentTarget, const Vec3 & upDirection,
            const Vec3 & cameraU, const Vec3 & cameraV, const Vec3 & cameraW)
    {
        Point3 retCenter = currentCenter, retTarget = currentTarget;
        bool isCameraMoved = false;

        //鼠标移动
        if (inputArgs.dx != 0 || inputArgs.dy != 0) {
            isCameraMoved = true;
            const Vec3 viewDirection = Point3::constructVector(currentCenter, currentTarget);

            //获取当前相机的方向向量
            Vec3 W = cameraW.unitVector();
            Vec3 U = cameraU.unitVector();
            Vec3 V = cameraV.unitVector();

            //左右旋转 (Yaw)
            //将视线向量(W) 绕着上方向量(V) 进行旋转，实现视角左右旋转
            const float yawAngle = -static_cast<float>(inputArgs.dx) * operateArgs.mouseSensitivity;
            W = W.rotate(V, yawAngle);

            //上下旋转 (Pitch)
            //将已经左右旋转过的视线向量(W) 绕着右方向量(U) 进行旋转
            float pitchAngle = -static_cast<float>(inputArgs.dy) * operateArgs.mouseSensitivity;
            W = W.rotate(U, pitchAngle);

            //限制俯仰角超过限制
            //从已经左右旋转过的W向量中获取当前俯仰角
            //W向量的Y分量是俯仰角(pitch)的正弦值，所以可以用asin来获取
            float newPitch = std::asin(W[1]);

            bool needsCorrection = false;
            if (newPitch > operateArgs.pitchLimitDegree) {
                newPitch = operateArgs.pitchLimitDegree;
                needsCorrection = true;
            } else if (newPitch < -operateArgs.pitchLimitDegree) {
                newPitch = -operateArgs.pitchLimitDegree;
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
            retTarget = currentCenter + W * viewDirection.length();
        }

        //键盘按键
        Vec3 movementDirection{};
        //使得键盘按键总是在水平平面上移动，移除方向向量的竖直分量（取平面投影）
        const Vec3 forwardHorizontal = Vec3{cameraW[0], 0.0f, cameraW[2]}.unitVector();
        if (inputArgs.keyW) movementDirection += forwardHorizontal; //Forward
        if (inputArgs.keyS) movementDirection -= forwardHorizontal; //Backward
        if (inputArgs.keyD) movementDirection += cameraU; //Right
        if (inputArgs.keyA) movementDirection -= cameraU; //Left
        //上下移动
        if (inputArgs.keySpace) movementDirection += upDirection;  //Up
        if (inputArgs.keyLShift) movementDirection -= upDirection; //Down

        if (movementDirection.lengthSquared() > 0.0f) {
            isCameraMoved = true;
            //将移动方向向量的长度变为1，确保斜向移动的速度和直线移动的速度一致，避免了“斜走更快”的问题
            const Vec3 translation = movementDirection.unitVector() * operateArgs.moveSpeed;
            retCenter += translation;
            retTarget += translation;
        }
        return {isCameraMoved, {retCenter, retTarget}};
    }

    void SDL_OpenGLWindow::mapCudaResource(cudaStream_t stream, SDL_OpenGLWindow::CUDAArgs & cudaArgs) {
        cudaCheckError(cudaGraphicsMapResources(1, &cudaArgs.cudaResource, stream));
        cudaArray_t cudaTextureArray;
        cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaArgs.cudaResource, 0, 0));
        cudaResourceDesc resDesc {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaTextureArray;
        cudaCheckError(cudaCreateSurfaceObject(&cudaArgs.surfaceObject, &resDesc));
    }
    void SDL_OpenGLWindow::unmapCudaResource(cudaStream_t stream, SDL_OpenGLWindow::CUDAArgs & cudaArgs) {
        cudaCheckError(cudaDestroySurfaceObject(cudaArgs.surfaceObject));
        cudaCheckError(cudaGraphicsUnmapResources(1, &cudaArgs.cudaResource, stream));
    }

    void SDL_OpenGLWindow::presentFrame(SDL_Window * window, const SDL_OpenGLWindow::OGLArgs & oglArgs) {
        glUseProgram(oglArgs.shaderProgram);
        glBindTexture(GL_TEXTURE_2D, oglArgs.textureID);
        glBindVertexArray(oglArgs.VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        SDL_GL_SwapWindow(window);
    }
}