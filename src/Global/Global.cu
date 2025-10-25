#include <Global/Global.cuh>
#ifndef NULL
#define NULL 0 //For filesystem header
#endif
#include <filesystem>
#include <fstream>

#if defined(CUDA) && defined(OPTIX)
#include <optix_function_table_definition.h>
#endif

namespace project {
#ifdef SDL
    void _SDL_CheckError(bool isError, const char * file, const char * function, int line) {
        if (!isError) return;
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[SDL] Error: %s. At file %s in function %s. (Line %d)",
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
    void _SDL_CheckErrorPtrBool(SDL_bool val, const char * file, const char * function, int line) {
        _SDL_CheckError(val == SDL_FALSE, file, function, line);
    }
#endif

#ifdef CUDA
    void _cudaCheckError(cudaError_t val, const char * file, const char * function, int line) {
        if (val != cudaSuccess) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[CUDA] Error: %s. At file %s in function %s. (Line %d)",
                         cudaGetErrorString(val), function, file, line);
            exit(CUDA_ERROR_EXIT_CODE);
        }
    }
#endif

#if defined(CUDA) && defined(OPTIX)
    void _optixCheckError(OptixResult result, const char * file, const char * function, int line) {
        if (result != OPTIX_SUCCESS) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[OptiX] Error: %s. At file %s in function %s. (Line %d)",
                         optixGetErrorString(result), function, file, line);
            exit(OPTIX_ERROR_EXIT_CODE);
        }
    }

    char optixPerCallLogBuffer[2048] = { 0 };
    size_t optixPerCallLogSize = sizeof(optixPerCallLogBuffer);
    void optixLogCallBackFunction(Uint32 level, const char * tag, const char * message, void * callbackData) {
        SDL_Log("[OptiX] [%u][%s] --> %s", level, tag, message);
    }
#endif //OPTIX

#if defined(_WIN32) && (defined(D3D11) || defined(D3D12))
    void _D3DPrintErrorMessage(HRESULT result, const char * file, const char * function, int line) {
        //获取错误信息
        WCHAR * errorText = nullptr;
        FormatMessageW(
                FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                nullptr,
                result,
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPWSTR)&errorText,
                0,
                nullptr
        );

        //转换为宽字符串，允许包含中文字符
        std::wstring resultStr;
        if (errorText != nullptr) {
            resultStr = errorText;
            LocalFree(errorText);
        }
        SDL_Log("[D3D] Error: %ls. At file %s in function %s. (Line %d)", resultStr.c_str(), file, function, line);

/*#ifdef D3D12
        const Uint64 numMessages = infoQueue->GetNumStoredMessages();
        if (numMessages == 0) {
            SDL_Log("No message in debug layer!");
            return;
        }
        for (Uint64 i = 0; i < numMessages; i++) {
            //获取消息的长度,以便分配正确的缓冲区大小
            size_t messageLength = 0;
            infoQueue->GetMessageA(i, nullptr, &messageLength);
            //分配缓冲区
            std::unique_ptr<D3D12_MESSAGE> message(
                    static_cast<D3D12_MESSAGE *>(malloc(messageLength))
            );
            //获取消息内容
            if (FAILED(infoQueue->GetMessageA(i, message.get(), &messageLength))) {
                SDL_Log("Failed to get debug message");
                exit(-100);
            }
            const char * severity;
            switch (message->Severity) {
                case D3D12_MESSAGE_SEVERITY_CORRUPTION:  severity = "CORRUPTION"; break;
                case D3D12_MESSAGE_SEVERITY_ERROR:       severity = "ERROR";      break;
                case D3D12_MESSAGE_SEVERITY_WARNING:     severity = "WARNING";    break;
                case D3D12_MESSAGE_SEVERITY_INFO:        severity = "INFO";       break;
                case D3D12_MESSAGE_SEVERITY_MESSAGE:     severity = "MESSAGE";    break;
                default:                                 severity = "UNKNOWN";    break;
            }
            SDL_Log("D3D12 debug message: [%s]: %s\n", severity, message->pDescription);
        }
        infoQueue->ClearStoredMessages();
#endif //D3D12*/
        exit(D3D_ERROR_EXIT_CODE);
    }
#endif

#ifdef VK
    VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugCallBackFunction(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void * pUserData)
    {
        SDL_LogPriority priority;
        switch (messageSeverity) {
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
                priority = SDL_LOG_PRIORITY_VERBOSE;
                break;
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
                priority = SDL_LOG_PRIORITY_INFO;
                break;
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
                priority = SDL_LOG_PRIORITY_WARN;
                break;
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                priority = SDL_LOG_PRIORITY_ERROR;
                break;
            default:
                priority = SDL_LOG_PRIORITY_INFO;
        }
        SDL_LogMessage(SDL_LOG_CATEGORY_APPLICATION, priority,
                       "[VK] [Validation]: %s", pCallbackData->pMessage);
        return VK_FALSE;
    }

    void _vkCheckError(VkResult result, const char * file, const char * function, int line) {
        if (result != VK_SUCCESS) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[Vulkan] Error: [%d]. At file %s in function %s. (Line %d)",
                         result, function, file, line);
            exit(VULKAN_ERROR_EXIT_CODE);
        }
    }
#endif

    std::string FileHelper::readTextFile(const std::string & filePath) {
        //打开文件
        const std::ifstream file(filePath);
        if (!std::filesystem::exists(filePath)) {
            throw std::runtime_error("File " + filePath + " does not exist!");
        }
        if(!file.is_open() || file.fail()) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }
        //读取文件
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    void FileHelper::writeTextFile(const std::string & filePath, const std::string & content) {
        //获取文件所在的目录路径
        std::filesystem::path pathObj(filePath);
        std::filesystem::path parentPath = pathObj.parent_path();
        //检查并创建目录（如果需要）
        if (!parentPath.empty()) {
            try {
                //创建路径中所有缺失的目录
                if (!std::filesystem::exists(parentPath)) {
                    if (!std::filesystem::create_directories(parentPath)) {
                        throw std::runtime_error("Failed to create directories to put new file!");
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                throw std::runtime_error("Filesystem error when creating directories!");
            }
        }

        //打开文件并写入内容
        //std::ofstream 默认使用 std::ios::out，如果文件不存在则创建，如果存在则截断（覆盖）
        std::ofstream outfile(filePath);
        if (outfile.is_open()) {
            outfile << content;
            outfile.close();
        } else {
            throw std::runtime_error("Failed to write text to file: " + content);
        }
    }
}