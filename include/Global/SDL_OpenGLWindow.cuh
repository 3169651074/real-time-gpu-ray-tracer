#ifndef RENDERERINTERACTIVE_SDL_OPENGLWINDOW_CUH
#define RENDERERINTERACTIVE_SDL_OPENGLWINDOW_CUH

#include <Basic/Point3.cuh>
#include <Util/Pair.cuh>

namespace renderer {
    class SDL_OpenGLWindow {
    public:
        //创建窗口
        static Pair<SDL_Window *, SDL_GLContext> createSDLGLWindow(const char * title, int w, int h, bool isVsync = false);
        static void destroySDLGLWindow(SDL_Window * window, SDL_GLContext context);

        //初始化OGL
        typedef struct OGLArgs {
            GLuint textureID;
            GLuint VAO, VBO, EBO;
            GLuint shaderProgram;
        } OGLArgs;
        static OGLArgs initializeOGL(SDL_Window * window);
        static void releaseOGL(OGLArgs & args);

        //获取CUDA资源
        typedef struct CUDAArgs {
            dim3 blocks;
            dim3 threads;
            cudaGraphicsResource_t cudaResource;
            cudaSurfaceObject_t surfaceObject;
        } CUDAArgs;
        static CUDAArgs getCudaResource(SDL_Window * window, const OGLArgs & oglArgs, Uint32 threadsPerBlock = 16);
        static void releaseCudaResource(CUDAArgs & cudaArgs);

        //更新相机位置
        typedef struct OperateArgs {
            float mouseSensitivity;
            float pitchLimitDegree;
            float moveSpeed;

            float fpsLimit;
            bool isRestrictFrameCount;

            std::chrono::microseconds targetFrameDuration;
            std::chrono::milliseconds sleepMargin;
        } OperateArgs;
        //键鼠输入变量（跨帧变量，在渲染循环外定义）
        typedef struct KeyMouseInputArgs {
            bool keyW;
            bool keyA;
            bool keyS;
            bool keyD;
            bool keySpace;
            bool keyLShift;

            int dx;
            int dy;

            bool mouseClick;
            bool keyQuit;
        } KeyMouseInputArgs;
        //初始化移动速度和帧率参数
        static OperateArgs getOperateArgs(float mouseSensitivity, float pitchLimitDegree, float moveSpeed, float fpsLimit) {
            return {
                .mouseSensitivity = mouseSensitivity,
                .pitchLimitDegree = PI / MathHelper::degreeToRadian(pitchLimitDegree),
                .moveSpeed = moveSpeed,
                .fpsLimit = fpsLimit,
                .isRestrictFrameCount = fpsLimit != INFINITY,
                .targetFrameDuration = std::chrono::microseconds(static_cast<Sint64>(1000000.0f / fpsLimit)),
                .sleepMargin = std::chrono::milliseconds(2)
            };
        }
        //根据一次事件查询结果设置键鼠输入变量
        static void getKeyMouseInput(KeyMouseInputArgs & inputArgs);
        //根据键鼠输入变量，操作信息和当前相机位置计算新的相机位置
        static Pair<bool, Pair<Point3, Point3>> calculateNewPosition(
                const OperateArgs & operateArgs, const KeyMouseInputArgs & inputArgs,
                const Point3 & currentCenter, const Point3 & currentTarget, const Vec3 & upDirection,
                const Vec3 & cameraU, const Vec3 & cameraV, const Vec3 & cameraW);

        //映射资源，让CUDA接管纹理，并获取可写入的Surface Object
        static void mapCudaResource(cudaStream_t stream, CUDAArgs & cudaArgs);
        static void unmapCudaResource(cudaStream_t stream, CUDAArgs & cudaArgs);

        //使用OGL渲染当前画面
        static void presentFrame(SDL_Window * window, const OGLArgs & oglArgs);
    };
}

#endif //RENDERERINTERACTIVE_SDL_OPENGLWINDOW_CUH
