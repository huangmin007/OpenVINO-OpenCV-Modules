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
/// �����ֽ��
/// </summary>
static const char Delimiter = ':';

/// <summary>
/// OpenCV cv::waitKey Ĭ�ϲο���ʱ
/// </summary>
static const int WaitKeyDelay = 33;


/// <summary>
/// ����̨�߳�����״̬������� SetConsoleCtrlHandler �ı�
/// </summary>
static bool IsRunning = true;

/// <summary>
/// ��� InferenceEngine ����汾�����ɼ����豸�б�
/// </summary>
static void InferenceEngineInfomation()
{
    static size_t width = 16;

    //-------------------------------- �汾��Ϣ --------------------------------
    const InferenceEngine::Version* ie_version = InferenceEngine::GetInferenceEngineVersion();

    std::cout.setf(std::ios::left);
    std::cout << "[InferenceEngine]" << std::endl;
    std::cout << std::setw(width) << "Version:" << ie_version << std::endl;
    std::cout << std::setw(width) << "Major:" << ie_version->apiVersion.major << std::endl;
    std::cout << std::setw(width) << "Minor:" << ie_version->apiVersion.minor << std::endl;
    std::cout << std::setw(width) << "BuildNumber:" << ie_version->buildNumber << std::endl;
    std::cout << std::setw(width) << "Description:" << ie_version->description << std::endl;

    //-------------------------------- ֧�ֵ�Ӳ���豸 --------------------------------
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
/// ����̨��������رջ��˳�����WinAPI ���� SetConsoleCtrlHandler ����
/// </summary>
/// <param name="CtrlType"></param>
/// <returns></returns>
static BOOL CloseHandlerRoutine(DWORD CtrlType)
{
    std::cout << "SetConsoleCtrlHandler:::" << CtrlType << std::endl;
    switch (CtrlType)
    {
    //���û�������CTRL+C,������GenerateConsoleCtrlEvent API����
    case CTRL_C_EVENT:          IsRunning = false;  break;
    //����ͼ�رտ���̨����ϵͳ���͹ر���Ϣ
    case CTRL_CLOSE_EVENT:      IsRunning = false;  break;
    //�û�����CTRL+BREAK, ������ GenerateConsoleCtrlEvent API ����
    case CTRL_BREAK_EVENT:      IsRunning = false;  break;
    //�û��˳�ʱ�����ǲ��ܾ������ĸ��û�
    case CTRL_LOGOFF_EVENT:     IsRunning = false;  break;
    //��ϵͳ���ر�ʱ
    case CTRL_SHUTDOWN_EVENT:   IsRunning = false;  break;
    }

    return TRUE;
}


/// <summary>
/// ��ȡ Window API ����ִ��״̬
/// </summary>
/// <param name="message">�����Ϣ�ı�</param>
/// <returns></returns>
static DWORD GetLastErrorFormatMessage(LPVOID& message)
{
    DWORD id = GetLastError();
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, id, 0, (LPTSTR)&message, 0, NULL);

    return id;
};


/// <summary>
/// ����ֻд�� �ڴ湲���ļ���� �� �ڴ�ռ�ӳ����ͼ
/// </summary>
/// <param name="handle">�ڴ��ļ����</param>
/// <param name="buffer">�ڴ�ӳ�䵽��ǰ���������ָ��</param>
/// <param name="size">������ڴ�ռ��С���ֽ�Ϊ��</param>
/// <param name="name">�ڴ湲���ļ������ƣ��������̶�дʹ�ø�����</param>
/// <returns>�����ɹ� ���� true</returns>
static bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name)
{
    ///���������ڴ�
    handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCSTR)name);
    LPVOID message = NULL;
    DWORD ErrorID = GetLastErrorFormatMessage(message);     //GetLastError ��� CreateFileMapping ״̬
    std::cout << "[ INFO] CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

    if (ErrorID != 0 || handle == NULL) return false;

    ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
    buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);  //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
    ErrorID = GetLastErrorFormatMessage(message);                   //GetLastError ��� MapViewOfFile ״̬
    std::cout << "[ INFO] MapViewOfFile     [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

    if (ErrorID != 0 || buffer == NULL)
    {
        CloseHandle(handle);
        return false;
    }

    return true;
}


/// <summary>
/// ��ȡֻ���� �ڴ湲���ļ���� �� �ڴ�ռ�ӳ����ͼ
/// </summary>
/// <param name="handle">�ڴ��ļ����</param>
/// <param name="buffer">�ڴ�ӳ�䵽��ǰ���������ָ��</param>
/// <param name="name">��Ҫ��ȡ���ڴ湲���ļ�������</param>
/// <returns>��ȡ�ɹ� ���� true</returns>
static bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name)
{
    ///���������ڴ�
    handle = OpenFileMapping(FILE_MAP_READ, FALSE, (LPCSTR)name);   //PAGE_READONLY
    //GetLastError ��� OpenFileMapping ״̬
    LPVOID message = NULL;
    DWORD ErrorID = GetLastErrorFormatMessage(message);
    std::cout << "[ INFO] OpenFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

    if (ErrorID != 0 || handle == NULL) return false;

    ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
    buffer = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
    //GetLastError ��� MapViewOfFile ״̬
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
/// [�ο��õ�]����ͨ�ò����������Ѿ������ help/info/input/output/model/async/show ����
/// </summary>
/// <param name="args">������������</param>
/// <param name="program_name">��������</param>
/// <param name="default_model">Ĭ�ϵ�ģ�����ò���</param>
/// <returns></returns>
static void CreateGeneralCmdLine(cmdline::parser & args, const std::string &program_name, const std::string &default_model)
{
    args.add("help", 'h', "����˵��");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "����Դ��������ʽ��(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
    args.add<std::string>("output", 'o', "���Դ��������ʽ��(shared|console|socket)[:value[:value[:...]]]", false, "shared:o_source.bin");// "shared:o_source.bin");
    args.add<std::string>("model", 'm', "���� AIʶ���� �� ����ģ������/�ļ�(.xml)��Ŀ���豸����ʽ��(AIģ������)[:����[:Ӳ��]]��"
        "ʾ����face-detection-adas-0001:FP16:CPU �� face-detection-adas-0001:FP16:HETERO:CPU,GPU", default_model.empty(), default_model);
    
    args.add<bool>("async", 0, "�Ƿ��첽����ʶ��", false, true);

#ifdef _DEBUG
    args.add<bool>("show", 0, "�Ƿ���ʾ��Ƶ���ڣ����ڵ���", false, true);
#else
    args.add<bool>("show", 0, "�Ƿ���ʾ��Ƶ���ڣ����ڵ���", false, false);
#endif

    args.set_program_name(program_name);
}

/// <summary>
/// ���� Size ��������ʽ��width(x|,)height
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
/// ��������ģ�Ͳ���
/// <para>�����ʽΪ��(AIģ������)[:����[:Ӳ��]]��ʾ����face-detection-adas-0001:FP32:CPU �� face-detection-adas-0001:FP16:GPU </para>
/// </summary>
/// <param name="args"></param>
/// <returns>���ؾ��� ["model","fp","device","path"] ���Ե� std::map ����</returns>
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
/// ���ַ��ָ������
/// </summary>
/// <param name="src">Դʼ�ַ���</param>
/// <param name="delimiter">�����</param>
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
