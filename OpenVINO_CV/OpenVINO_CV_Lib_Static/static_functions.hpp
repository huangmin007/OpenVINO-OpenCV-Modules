#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <Windows.h>
#include <winioctl.h>
#include <inference_engine.hpp>


#include "cmdline.h"

#define LOG(type)       (std::cout << "[" << std::setw(5) << std::right << type << "] ")
#define LOG_INFO            LOG("INFO")
#define LOG_WARN            LOG("WARN")
#define LOG_ERROR           LOG("ERROR")

void InferenceEngineInfomation();

BOOL CloseHandlerRoutine(DWORD CtrlType);
DWORD GetLastErrorFormatMessage(LPVOID& message);
bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name);
bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name);

bool ParseArgForSize(const std::string& size, int& width, int& height);
const std::map<std::string, std::string> ParseArgsForModel(const std::string& args);
const std::vector<std::string> SplitString(const std::string& src, const char delimiter);

/// <summary>
/// ms
/// </summary>
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

/// <summary>
/// 参数分界符
/// </summary>
static const char Delimiter = ':';

/// <summary>
/// OpenCV cv::waitKey 默认参考延时
/// </summary>
static const int WaitKeyDelay = 33;


/// <summary>
/// 控制台线程运行状态，会监听 SetConsoleCtrlHandler 改变
/// </summary>
static bool IsRunning = true;

/// <summary>
/// 输出 InferenceEngine 引擎版本，及可计算设备列表
/// </summary>
static void InferenceEngineInfomation()
{
    static size_t width = 16;

    //-------------------------------- 版本信息 --------------------------------
    const InferenceEngine::Version* ie_version = InferenceEngine::GetInferenceEngineVersion();

    std::cout.setf(std::ios::left);
    std::cout << "[InferenceEngine]" << std::endl;
    std::cout << std::setw(width) << "Version:" << ie_version << std::endl;
    std::cout << std::setw(width) << "Major:" << ie_version->apiVersion.major << std::endl;
    std::cout << std::setw(width) << "Minor:" << ie_version->apiVersion.minor << std::endl;
    std::cout << std::setw(width) << "BuildNumber:" << ie_version->buildNumber << std::endl;
    std::cout << std::setw(width) << "Description:" << ie_version->description << std::endl;

    //-------------------------------- 支持的硬件设备 --------------------------------
    InferenceEngine::Core core;
    std::vector<std::string> devices = core.GetAvailableDevices();

    std::cout << "[SupportDevices]" << std::endl;
    for (const auto& device : devices)
    {
        std::string dn = core.GetMetric(device, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();

        std::cout << std::setw(width) << " (" << device << ") " << dn << std::endl;
    }
    std::cout << std::endl;

    system("pause");
}

/// <summary>
/// 控制台程序操作关闭或退出处理，WinAPI 函数 SetConsoleCtrlHandler 调用
/// </summary>
/// <param name="CtrlType"></param>
/// <returns></returns>
static BOOL CloseHandlerRoutine(DWORD CtrlType)
{
    std::cout << "SetConsoleCtrlHandler:::" << CtrlType << std::endl;
    switch (CtrlType)
    {
    //当用户按下了CTRL+C,或者由GenerateConsoleCtrlEvent API发出
    case CTRL_C_EVENT:          IsRunning = false;  break;
    //当试图关闭控制台程序，系统发送关闭消息
    case CTRL_CLOSE_EVENT:      IsRunning = false;  break;
    //用户按下CTRL+BREAK, 或者由 GenerateConsoleCtrlEvent API 发出
    case CTRL_BREAK_EVENT:      IsRunning = false;  break;
    //用户退出时，但是不能决定是哪个用户
    case CTRL_LOGOFF_EVENT:     IsRunning = false;  break;
    //当系统被关闭时
    case CTRL_SHUTDOWN_EVENT:   IsRunning = false;  break;
    }

    return TRUE;
}


/// <summary>
/// 获取 Window API 函数执行状态
/// </summary>
/// <param name="message">输出信息文本</param>
/// <returns></returns>
static DWORD GetLastErrorFormatMessage(LPVOID& message)
{
    DWORD id = GetLastError();
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, id, 0, (LPTSTR)&message, 0, NULL);

    return id;
};


/// <summary>
/// 创建只写的 内存共享文件句柄 及 内存空间映射视图
/// </summary>
/// <param name="handle">内存文件句柄</param>
/// <param name="buffer">内存映射到当前程序的数据指针</param>
/// <param name="size">分配的内存空间大小，字节为单</param>
/// <param name="name">内存共享文件的名称，其它进程读写使用该名称</param>
/// <returns>创建成功 返回 true</returns>
static bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name)
{
    ///创建共享内存
    handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCSTR)name);
    LPVOID message = NULL;
    DWORD ErrorID = GetLastErrorFormatMessage(message);     //GetLastError 检查 CreateFileMapping 状态
    std::cout << "[ INFO] CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

    if (ErrorID != 0 || handle == NULL) return false;

    ///映射到当前进程的地址空间视图
    buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);  //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
    ErrorID = GetLastErrorFormatMessage(message);                   //GetLastError 检查 MapViewOfFile 状态
    std::cout << "[ INFO] MapViewOfFile     [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

    if (ErrorID != 0 || buffer == NULL)
    {
        CloseHandle(handle);
        return false;
    }

    return true;
}


/// <summary>
/// 获取只读的 内存共享文件句柄 及 内存空间映射视图
/// </summary>
/// <param name="handle">内存文件句柄</param>
/// <param name="buffer">内存映射到当前程序的数据指针</param>
/// <param name="name">需要读取的内存共享文件的名称</param>
/// <returns>获取成功 返回 true</returns>
static bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name)
{
    ///创建共享内存
    handle = OpenFileMapping(FILE_MAP_READ, FALSE, (LPCSTR)name);   //PAGE_READONLY
    //GetLastError 检查 OpenFileMapping 状态
    LPVOID message = NULL;
    DWORD ErrorID = GetLastErrorFormatMessage(message);
    std::cout << "[ INFO] OpenFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

    if (ErrorID != 0 || handle == NULL) return false;

    ///映射到当前进程的地址空间视图
    buffer = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
    //GetLastError 检查 MapViewOfFile 状态
    ErrorID = GetLastErrorFormatMessage(message);
    std::cout << "[ INFO] MapViewOfFile   [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

    if (ErrorID != 0 || buffer == NULL)
    {
        CloseHandle(handle);
        return false;
    }

    return true;
}

/// <summary>
/// [参考用的]创建通用参数解析；已经添加了 help/info/input/output/model/async/show 参数
/// </summary>
/// <param name="args">参数解析对象</param>
/// <param name="program_name">程序名称</param>
/// <param name="default_model">默认的模型配置参数</param>
/// <returns></returns>
static void CreateGeneralCmdLine(cmdline::parser & args, const std::string &program_name, const std::string &default_model)
{
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
    args.add<std::string>("output", 'o', "输出源参数，格式：(shared|console|socket)[:value[:value[:...]]]", false, "shared:o_source.bin");// "shared:o_source.bin");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：face-detection-adas-0001:FP16:CPU 或 face-detection-adas-0001:FP16:HETERO:CPU,GPU", default_model.empty(), default_model);
    
    args.add<bool>("async", 0, "是否异步分析识别", false, true);

#ifdef _DEBUG
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif

    args.set_program_name(program_name);
}

/// <summary>
/// 解析 Size 参数，格式：width(x|,)height
/// </summary>
/// <param name="size"></param>
/// <param name="width"></param>
/// <param name="height"></param>
/// <returns></returns>
static bool ParseArgForSize(const std::string& size, int& width, int& height)
{
    if (size.empty())return false;

    int index = size.find('x') != std::string::npos ? size.find('x') : size.find(',');
    if (index == std::string::npos) return false;

    width = std::stoi(size.substr(0, index));
    height = std::stoi(size.substr(index + 1));

    return true;
}

/// <summary>
/// 解析输入模型参数
/// <para>输入格式为：(AI模型名称)[:精度[:硬件]]，示例：face-detection-adas-0001:FP32:CPU 或 face-detection-adas-0001:FP16:GPU </para>
/// </summary>
/// <param name="args"></param>
/// <returns>返回具有 ["model","fp","device","path"] 属性的 std::map 数据</returns>
static const std::map<std::string, std::string> ParseArgsForModel(const std::string &args)
{
    std::map<std::string, std::string> argMap
    {
        {"path", ""},
        {"model", ""},
        {"fp", "FP16"},
        {"device", "CPU"}
    };

    std::vector<std::string> model = SplitString(args, ':');
    size_t length = model.size();
    if (length <= 0) return argMap;

    //model
    if (length >= 1)    argMap["model"] = model[0];
    //model[:FP]
    if (length >= 2)    argMap["fp"] = model[1];
    //model[:FP[:device]]
    if (length >= 3)    argMap["device"] = model[2];
    //model[:fp[:device(HETERO:CPU,GPU)]]
    if(length >= 4)     argMap["device"] = model[2] + ":" + model[3];

    argMap["path"] = "models\\" + argMap["model"] + "\\" + argMap["fp"] + "\\" + argMap["model"] + ".xml";

    return argMap;
}

/// <summary>
/// 将字符分割成数组
/// </summary>
/// <param name="src">源始字符串</param>
/// <param name="delimiter">定界符</param>
/// <returns></returns>
static const std::vector<std::string> SplitString(const std::string& src, const char delimiter = ':')
{
    std::vector<std::string> vec;
    size_t start = 0, end = src.find(delimiter);

    while (end != std::string::npos)
    {
        vec.push_back(src.substr(start, end - start));

        start = end + 1;
        end = src.find(delimiter, start);
    }

    if (start != src.length())
        vec.push_back(src.substr(start));

    return vec;
}
