#pragma once

#include <iostream>
#include <chrono>
#include <Windows.h>
#include <winioctl.h>
#include <opencv2/opencv.hpp>

#define LOG(type)       (std::cout << "[" << std::setw(5) << std::right << type << "] ")
#define LOGI_NFO            LOG("INFO")
#define LOG_WARN            LOG("WARN")
#define LOG_ERROR           LOG("ERROR")


//�� 1 �ֽڶ���
#pragma pack(push, 1)

//����̨�߳�����״̬������� SetConsoleCtrlHandler �ı�
static bool IsRunning = true;
static const int WaitKeyDelay = 33;     //OpenCV cv::waitKey Ĭ�ϲο���ʱ

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

// [�ڴ湲������] ��Ƶ�����Դ���ݸ�ʽ
// pack(1) ���Ϊ 32 �ֽڴ�С������ͷ��Ϣ���鲻Ҫ���� 32 �ֽ�
struct VideoSourceData
{
    uint32_t    fid = 0;        //Frame ID�����ݿ�ÿ����һ�Σ���ֵ�����һ�Σ������ڼ��㲻���ظ�����֡

    /// Camera ����
    uint16_t    width;          //��Ƶ�����Դ��
    uint16_t    height;         //��Ƶ�����Դ��
    uint8_t     channels;       //��Ƶ�����Դͨ������RGB��3ͨ����ARGB��4ͨ�����Ҷ�ͼ��2ͨ���͵�ͨ��(1ͨ��)
    uint32_t    length;         //�ļ����ݴ�С
    uint8_t     type;           //Mat type
    uint8_t     format;         //��Ƶ�����Դ��ʽ��0 ��ʾδԭʼ���ݸ�ʽ��>0 ��ʾѹ�����ͼ�����ݣ�����ͼ���ѹ����δ����

    /// Video ����
    uint8_t     reserve0 = 0;   //����
    uint32_t    reserve1 = 0;   //����
    uint32_t    reserve2 = 0;   //����
    uint32_t    reserve3 = 0;   //����
    uint32_t    reserve4 = 0;   //����

    //uint8_t     *data;          //raw data

    //copy Mat info to VideoSourceData
    void copy(const cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();
        
        this->length = this->width * this->height * this->channels * src.elemSize1();
    };

    //copy Mat info to VideoSourceData
    VideoSourceData& operator << (const cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();
        this->length = this->width * this->height * this->channels * src.elemSize1();

        return *this;
    };
};

// ���ݸ�ʽ
enum InferenceDataFormat:uint8_t
{
    RAW,        //ԭʼ�ֽ����ݣ��ɽ�ԭʼ�ֽ�תΪָ���Ľṹ����
    STRING,     //�ַ����ͣ�δ�����ַ���ʽ������һ������
    JSON,       //�ַ����ͣ�����Ϊ JSON ��ʽ
    XML,        //�ַ����ͣ�����Ϊ XML ��ʽ
};

// [�ڴ湲������] �����������ݽṹ
// pack(1) ���Ϊ 32 �ֽڴ�С������ͷ��Ϣ���鲻Ҫ���� 32 �ֽ�
struct InferenceData
{
    uint32_t    fid = 0;        //֡ id
    uint8_t     mid;            //����Ӧ����ָģ��ID�����ĸ�ģ�����������
    uint32_t    length;         //���ݵ���Ч����
    InferenceDataFormat     format;         //���ݸ�ʽ��see InferenceDataFormat

    uint16_t    reserve0 = 0;
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;
    uint32_t    reserve5 = 0;

    //uint8_t     data[];         //raw data����ͬ�Ľ�����ͣ����ݵĽ��Ͳ�һ����Ӧ�����ݵ���Ч���ȶ�̬
};

struct PoseData2D
{
    
};

//SetConsoleCtrlHandler
//����̨��������رջ��˳�����
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

//��ȡ Window API ����ִ��״̬
static DWORD GetLastErrorFormatMessage(LPVOID& message)
{
    DWORD id = GetLastError();
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, id, 0, (LPTSTR)&message, 0, NULL);

    return id;
};

//����ֻд���ڴ湲���ļ�������ڴ�ռ�ӳ����ͼ
//�����ɹ� ���� true
static bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint size, const char* name)
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

//��ȡֻ�����ڴ湲���ļ�������ڴ�ռ�ӳ����ͼ
//��ȡ�ɹ� ���� true
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

#pragma pack(pop)