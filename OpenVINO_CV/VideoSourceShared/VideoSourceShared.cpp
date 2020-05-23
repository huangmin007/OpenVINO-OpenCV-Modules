﻿// VideoSourceShared.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include "cmdline.h"
#include "io_data_format.hpp"
#include "static_functions.hpp"

// 内存还没有加读写锁，先测试后在看结果

int main(int argc, char **argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cerr << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

    cmdline::parser args;
    args.add<std::string>("help", 'h', "参数说明", false, "");
    args.add<std::string>("input", 'i', "输入视频或相机源，格式：-i (video|camera)[:value]", false, "cam:0");
    args.add<std::string>("size", 's', "设置视频源或是相机的输出尺寸", false, "640,480");

    args.add<std::string>("fn", '\0', "内存共享文件名称", false, "source.bin");
    args.add<uint32_t>("fs", '\0', "分配给共享内存的空间大小，默认为：(1920 x 1080 x 4)Bytes", false, (1920 * 1080 * 4));
    args.add<bool>("show", '\0', "是否显示视频窗口，用于调试", false, false);
    args.set_program_name("VideoSourceShared");

    bool isParser = args.parse(argc, argv);    
    if (args.exist("help"))
    {
        std::cerr << args.usage();
        return EXIT_SUCCESS;
    }
    if (!isParser) 
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    cv::VideoCapture capture;

#pragma region 解析input参数
    std::string input = args.get<std::string>("input");
    size_t start = 0;
    size_t end = input.find(Delimiter);
    std::string type = input.substr(start, end);

    if (type == "cam" || type == "camera")
    {
        //相机 默认索引 为 0
        int index = (end == std::string::npos) ? 0 : std::stoi(input.substr(end + 1).c_str());

        capture.open(index);        
        LOG_INFO << "Source Camera Index:" << index <<  std::endl;
    }
    else if (type == "url" || type == "void")
    {
        if (end == std::string::npos)
        {
            throw std::invalid_argument("未指定 VIDEO 源路径：" + input);
            return EXIT_FAILURE;
        }

        //视频 URL 地址
        std::string url = input.substr(end + 1);
        capture.open(url);
        LOG_INFO << "Source Video URL:" << url << std::endl;
    }
    else
    {
        LOG_ERROR << "input 参数输入错误：" + input + "， 示例：(video|camera)[:value]" << std::endl;
        return EXIT_FAILURE;
    }
#pragma endregion

#pragma region 解析size参数
    int w = 640, h = 480;
    try
    {
        //获取配置宽高信息
        std::string size = args.get<std::string>("size");
        int c = size.find(',') > 0 ? size.find(',') : size.find('x') > 0 ? size.find('x') : std::string::npos;
        if (c > 0)
        {
            w = std::stoi(size.substr(0, c));
            h = std::stoi(size.substr(c + 1));
        }
    }
    catch (std::exception & e)
    {
        LOG_ERROR << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, w);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    LOG_INFO << "Setting Source Size:" << w << "," << h << std::endl;
#pragma endregion

    if (!capture.isOpened())
    {
        LOG_ERROR << "无法打开视频或相机源：" << input << std::endl;
        return EXIT_FAILURE;
    }

    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int format = capture.get(cv::CAP_PROP_FORMAT);
    int fps = capture.get(cv::CAP_PROP_FPS);

    LOG_INFO << "Capture Size:" << width << "," << height << "  FPS:" << fps << "  Count:" << count << "  Format:" << format << std::endl;

    cv::Mat srcFrame;
    VideoSourceData vsData;
    static int vsdSize = sizeof(VideoSourceData);

    LPVOID srcBuffer = NULL;
    HANDLE srcMapFile = NULL;    
    if (!CreateOnlyWriteMapFile(srcMapFile, srcBuffer, args.get<uint32_t>("fs") + vsdSize,
        args.get<std::string>("fn").c_str()))
    {
        std::cerr << "[ERROR] 创建共享内存失败" << std::endl;
        return EXIT_FAILURE;
    }

    double use_time = 0.0;
    int waitDelay = WaitKeyDelay;
    bool isShow = args.get<bool>("show");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (!IsRunning) break;

        t0 = std::chrono::high_resolution_clock::now();
        if (!capture.read(srcFrame))
        {
            if (srcFrame.empty())
            {
                LOG_ERROR << "读取到空帧，或视频播放结束" << std::endl;
                break;
            }
        }

        //写入到共享内存
        vsData.fid += 1;
        vsData.copy(srcFrame);
        std::copy((uint8_t*)&vsData, (uint8_t*)&vsData + vsdSize, (uint8_t*)srcBuffer);
        std::copy((uint8_t*)srcFrame.data, (uint8_t*)srcFrame.data + vsData.length, (uint8_t*)srcBuffer + vsdSize);

        t1 = std::chrono::high_resolution_clock::now();
        use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

        waitDelay = WaitKeyDelay - use_time;
        waitDelay = waitDelay <= 0 ? 1 : waitDelay;

        std::cout << "\33[K\r[ INFO] " << "Current FrameID:" << vsData.fid << "  Read/Write Use:" << use_time << "ms";

        cv::waitKey(waitDelay);
        if (isShow) cv::imshow("Video Source", srcFrame);
    }
    std::cout << std::endl;

    capture.release();
    cv::destroyAllWindows();

    //撤消地址空间内的视图
    if (srcBuffer != NULL)   UnmapViewOfFile(srcBuffer);
    //关闭共享文件句柄
    if (srcMapFile != NULL)  CloseHandle(srcMapFile);

    std::cout << "[ INFO] Exiting ..." << std::endl;
    Sleep(500);

    return EXIT_SUCCESS;
}


