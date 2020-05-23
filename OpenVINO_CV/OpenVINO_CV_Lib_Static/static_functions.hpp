#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <Windows.h>
#include <winioctl.h>

#include "cmdline.h"

#define LOG(type)       (std::cout << "[" << std::setw(5) << std::right << type << "] ")
#define LOG_INFO            LOG("INFO")
#define LOG_WARN            LOG("WARN")
#define LOG_ERROR           LOG("ERROR")

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
/// 控制台程序操作关闭或退出处理，WinAPI 函数 SetConsoleCtrlHandler 调用
/// </summary>
/// <param name="CtrlType"></param>
/// <returns></returns>
static BOOL CloseHandlerRoutine(DWORD CtrlType)
{
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
/// [参考用的]创建通用参数解析器；已经添加了 help/input/output/model/pc/async/show 参数
/// </summary>
/// <param name="program_name">程序名称</param>
/// <returns></returns>
static void CreateGeneralCmdLine(cmdline::parser & args, const std::string &program_name, const std::string &default_model)
{
    args.add<std::string>("help", 'h', "参数说明", false, "");
    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
    args.add<std::string>("output", 'o', "输出源参数，格式：(shared|console|socket)[:value[:value[:...]]]", false, "console:TEST");// "shared:o_source.bin");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：face-detection-adas-0001:FP16:CPU 或 face-detection-adas-0001:FP16:HETERO:&lt;CPU,GPU&gt;", default_model.empty(), default_model);
    
    args.add<bool>("pc", 0, "启用每层性能报告", false, false);
    args.add<bool>("async", 0, "是否异步分析识别", false, true);
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
    args.set_program_name(program_name);
}

/// <summary>
/// 解析输入模型参数
/// <para>输入格式为：(AI模型名称)[:精度[:硬件]]，示例：face-detection-adas-0001:FP16:CPU 或 face-detection-adas-0001:FP16:HETERO:&lt;CPU,GPU&gt; </para>
/// </summary>
/// <param name="args"></param>
/// <returns>返回具有 ["model"/"name","fp","device","path"/"file"] 属性的 std::map 数据</returns>
static const std::map<std::string, std::string> ParseArgsForModel(std::string args)
{
    std::stringstream path;
    std::map<std::string, std::string> argMap;
    argMap["fp"] = "";
    argMap["path"] = "";
    argMap["model"] = "";
    argMap["device"] = "CPU";

    if (args.empty() || args.length() == 0)   return argMap;

    //model
    size_t start = 0, end = args.find(Delimiter);
    if (end == std::string::npos)
    {
        argMap["fp"] = "FP16";
        argMap["device"] = "CPU";
        argMap["model"] = args.substr(start);

        path << "models\\" << argMap["model"] << "\\" << argMap["fp"] << "\\" << argMap["model"] << ".xml";
        argMap["path"] = path.str();
        return argMap;
    }

    //model:fp
    argMap["model"] = args.substr(start, end - start);
    start = end + 1;
    end = args.find(Delimiter, start);
    if (end == std::string::npos)
    {
        argMap["fp"] = args.substr(start);
        if (argMap["fp"].length() < 4) argMap["fp"] = "FP16";

        argMap["device"] = "CPU";

        path << "models\\" << argMap["model"] << "\\" << argMap["fp"] << "\\" << argMap["model"] << ".xml";
        argMap["path"] = path.str();
        return argMap;
    }
    
    //model:fp:device
    argMap["fp"] = args.substr(start, end - start);
    if (argMap["fp"].length() < 4) argMap["fp"] = "FP16";

    start = end + 1;
    end = args.find(Delimiter, start);

    argMap["device"] = args.substr(start);
    if (argMap["device"].length() < 3) argMap["device"] = "CPU";

    path << "models\\" << argMap["model"] << "\\" << argMap["fp"] << "\\" << argMap["model"] << ".xml";
    argMap["path"] = path.str();

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

        start = end;
        end = src.find(delimiter, start);
    }

    if (start != src.length())
        vec.push_back(src.substr(start));

    return vec;
}
