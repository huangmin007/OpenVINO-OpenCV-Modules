#pragma once

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Windows.h>

#include "LoggerConfig.h"
#include <winioctl.h>

//控制台线程运行状态，会监听 SetConsoleCtrlHandler 改变
static bool IsRunning = true;
//OpenCV cv::waitKey 默认参考延时
static const int WaitKeyDelay = 33;

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

//内存共享文件头部信息
struct MapFileHeadInfo
{
    uint32_t    fid = 0;       //4字节，数据块每更新一次，该值会递增一次，可用于计算不做重复分析帧

    /// Camera 参数 11
    uint16_t    width;         //视频源宽
    uint16_t    height;        //视频源高
    uint8_t     channels;       //视频源通道数，RGB是3通道，ARGB是4通道，灰度图分2通道和单通道(1通道)
    uint32_t    length;        //文件数据大小
    uint8_t     type;           //Mat type
    uint8_t     format;         //视频源格式，0 表示未原始数据格式，>0 表示压缩后的图像数据，各种图像的压缩暂未定义

    ///  Video 参数
    //double duration;

    /// 以下为保留字段 4 x 4 = 16
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;

    //uint8_t     data[];             //数据

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

//输出结果数据格式
enum ResultDataFormat:uint8_t
{
    RAW,        //原始字节数据
    STRING,     //字符类型，未定义字符格式
    JSON,       //字符类型，定义为 JSON 格式
    XML,        //字符类型，定义为 XML 格式
};

// 输出数据结构
struct OutputData
{
    uint32_t    fid = 0;        //帧 id
    uint8_t     mid;            //这里应该是指模块ID，由哪个模块输出的数据
    uint32_t    length;         //数据的有效长度
    uint8_t     format;         //数据格式

    /// 以下为保留字段 4 x 4 = 16
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;

    uint8_t     data[];         //原始数据，不同的结果类型，数据的解释不一样，应该数据的有效长度动态
};

//SetConsoleCtrlHandler
//控制台程序操作关闭或退出处理
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

//获取 Window API 函数执行状态
static DWORD GetLastErrorFormatMessage(LPVOID& message)
{
    DWORD id = GetLastError();
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, id, 0, (LPTSTR)&message, 0, NULL);

    return id;
};

//创建只写的内存共享文件
//创建成功 返回 true
static bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint size, const char* name)
{
    ///创建共享内存
    handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCSTR)name);
    //GetLastError 检查 CreateFileMapping 状态
    LPVOID message = NULL;
    DWORD ErrorID = GetLastErrorFormatMessage(message);
    std::stringstream info; info << "CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(info.str()));

    if (handle == NULL) return false;

    ///映射到当前进程的地址空间视图
    buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size); //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
    //GetLastError 检查 MapViewOfFile 状态
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

//写入共享内存
//返回写入耗时时间ms
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
