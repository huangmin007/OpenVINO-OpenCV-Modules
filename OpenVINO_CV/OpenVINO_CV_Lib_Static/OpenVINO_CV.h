#pragma once

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Windows.h>

#include "LoggerConfig.h"
#include <winioctl.h>

//����̨�߳�����״̬������� SetConsoleCtrlHandler �ı�
static bool IsRunning = true;
//OpenCV cv::waitKey Ĭ�ϲο���ʱ
static const int WaitKeyDelay = 33;

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

//�ڴ湲���ļ�ͷ����Ϣ
struct MapFileHeadInfo
{
    uint32_t    fid = 0;       //4�ֽڣ����ݿ�ÿ����һ�Σ���ֵ�����һ�Σ������ڼ��㲻���ظ�����֡

    /// Camera ���� 11
    uint16_t    width;         //��ƵԴ��
    uint16_t    height;        //��ƵԴ��
    uint8_t     channels;       //��ƵԴͨ������RGB��3ͨ����ARGB��4ͨ�����Ҷ�ͼ��2ͨ���͵�ͨ��(1ͨ��)
    uint32_t    length;        //�ļ����ݴ�С
    uint8_t     type;           //Mat type
    uint8_t     format;         //��ƵԴ��ʽ��0 ��ʾδԭʼ���ݸ�ʽ��>0 ��ʾѹ�����ͼ�����ݣ�����ͼ���ѹ����δ����

    ///  Video ����
    //double duration;

    /// ����Ϊ�����ֶ� 4 x 4 = 16
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;

    //uint8_t     data[];             //����

    void copy(const cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();
        
        this->length = this->width * this->height * this->channels * src.elemSize1();
    };

    MapFileHeadInfo& operator << (const cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();
        this->length = this->width * this->height * this->channels * src.elemSize1();

        return *this;
    };
};

//���������ݸ�ʽ
enum ResultDataFormat:uint8_t
{
    RAW,        //ԭʼ�ֽ�����
    STRING,     //�ַ����ͣ�δ�����ַ���ʽ
    JSON,       //�ַ����ͣ�����Ϊ JSON ��ʽ
    XML,        //�ַ����ͣ�����Ϊ XML ��ʽ
};

// ������ݽṹ
struct OutputData
{
    uint32_t    fid = 0;        //֡ id
    uint8_t     mid;            //����Ӧ����ָģ��ID�����ĸ�ģ�����������
    uint32_t    length;         //���ݵ���Ч����
    uint8_t     format;         //���ݸ�ʽ

    /// ����Ϊ�����ֶ� 4 x 4 = 16
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;

    uint8_t     data[];         //ԭʼ���ݣ���ͬ�Ľ�����ͣ����ݵĽ��Ͳ�һ����Ӧ�����ݵ���Ч���ȶ�̬
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

//����ֻд���ڴ湲���ļ�
//�����ɹ� ���� true
static bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint size, const char* name)
{
    ///���������ڴ�
    handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCSTR)name);
    //GetLastError ��� CreateFileMapping ״̬
    LPVOID message = NULL;
    DWORD ErrorID = GetLastErrorFormatMessage(message);
    std::stringstream info; info << "CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(info.str()));

    if (handle == NULL) return false;

    ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
    buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size); //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
    //GetLastError ��� MapViewOfFile ״̬
    ErrorID = GetLastErrorFormatMessage(message);
    info.str("");  info << "MapViewOfFile [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(info.str()));

    if (buffer == NULL)
    {
        CloseHandle(handle);
        return false;
    }

    return true;
}

//д�빲���ڴ�
//����д���ʱʱ��ms
static bool WriteMapFile(HANDLE& handle, LPVOID& buffer, MapFileHeadInfo* info, const cv::Mat& frame)
{
    if (handle == NULL || buffer == NULL) return false;

    static int offset = sizeof(MapFileHeadInfo);

    info->fid += 1;
    info->copy(frame);

    std::copy((uint8_t*)info, (uint8_t*)info + offset, (uint8_t*)buffer);
    std::copy((uint8_t*)frame.data, (uint8_t*)frame.data + info->length, (uint8_t*)buffer + offset);

    return true;
}
