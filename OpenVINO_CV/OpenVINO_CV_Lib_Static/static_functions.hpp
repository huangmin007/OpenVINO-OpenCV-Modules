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
/// ����̨��������رջ��˳�����WinAPI ���� SetConsoleCtrlHandler ����
/// </summary>
/// <param name="CtrlType"></param>
/// <returns></returns>
static BOOL CloseHandlerRoutine(DWORD CtrlType)
{
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
/// [�ο��õ�]����ͨ�ò������������Ѿ������ help/input/output/model/pc/async/show ����
/// </summary>
/// <param name="program_name">��������</param>
/// <returns></returns>
static void CreateGeneralCmdLine(cmdline::parser & args, const std::string &program_name, const std::string &default_model)
{
    args.add<std::string>("help", 'h', "����˵��", false, "");
    args.add<std::string>("input", 'i', "����Դ��������ʽ��(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
    args.add<std::string>("output", 'o', "���Դ��������ʽ��(shared|console|socket)[:value[:value[:...]]]", false, "console:TEST");// "shared:o_source.bin");
    args.add<std::string>("model", 'm', "���� AIʶ���� �� ����ģ������/�ļ�(.xml)��Ŀ���豸����ʽ��(AIģ������)[:����[:Ӳ��]]��"
        "ʾ����face-detection-adas-0001:FP16:CPU �� face-detection-adas-0001:FP16:HETERO:&lt;CPU,GPU&gt;", default_model.empty(), default_model);
    
    args.add<bool>("pc", 0, "����ÿ�����ܱ���", false, false);
    args.add<bool>("async", 0, "�Ƿ��첽����ʶ��", false, true);
    args.add<bool>("show", 0, "�Ƿ���ʾ��Ƶ���ڣ����ڵ���", false, true);
    args.set_program_name(program_name);
}

/// <summary>
/// ��������ģ�Ͳ���
/// <para>�����ʽΪ��(AIģ������)[:����[:Ӳ��]]��ʾ����face-detection-adas-0001:FP16:CPU �� face-detection-adas-0001:FP16:HETERO:&lt;CPU,GPU&gt; </para>
/// </summary>
/// <param name="args"></param>
/// <returns>���ؾ��� ["model"/"name","fp","device","path"/"file"] ���Ե� std::map ����</returns>
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

        start = end;
        end = src.find(delimiter, start);
    }

    if (start != src.length())
        vec.push_back(src.substr(start));

    return vec;
}
