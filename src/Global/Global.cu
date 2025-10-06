#include <Global/Global.cuh>

namespace renderer {
#ifdef SDL
#define SDL_ERROR_EXIT_CODE (-100)
    void _SDL_CheckError(bool isError, const char * file, const char * function, int line) {
        if (!isError) return;
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "SDL error: %s. At file %s in function %s. (Line %d)",
                     SDL_GetError(), file, function, line);
        SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Aborting due to SDL error!");
        exit(SDL_ERROR_EXIT_CODE);
    }
    void _SDL_CheckErrorIntImpl(int val, const char * file, const char * function, int line) {
        _SDL_CheckError(val < 0, file, function, line);
    }
    void _SDL_CheckErrorPtrImpl(const void * val, const char * file, const char * function, int line) {
        _SDL_CheckError(val == nullptr, file, function, line);
    }
#endif

#ifdef CUDA
#define CUDA_ERROR_EXIT_CODE (-200)
    void _cudaCheckError(cudaError_t val, const char * file, const char * function, int line) {
        if (val != cudaSuccess) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "Cuda error: %s. At file %s in function %s. (Line %d)",
                         cudaGetErrorString(val), function, file, line);
            exit(CUDA_ERROR_EXIT_CODE);
        }
    }
#endif
}