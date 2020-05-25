// VideoSourceShared.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
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

#pragma region 设置解析参数
    cmdline::parser args;
    args.add<std::string>("help", 'h', "参数说明", false, "");
    args.add<std::string>("input", 'i', "输入视频或相机源，格式：-i (video|camera)[:value[:size]]", false, "cam:0:640x480");
    args.add<std::string>("output", 'o', "输出内存共享源，格式：-o (shared)[:name[:size]]", false, "shared:source.bin:8294400");
#ifdef _DEBUG
    args.add<bool>("show", '\0', "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", '\0', "是否显示视频窗口，用于调试", false, false);
#endif
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

    bool isShow = args.get<bool>("show");
    std::string input = args.get<std::string>("input");
    std::string output = args.get<std::string>("output");
#pragma endregion

    cv::VideoCapture capture;

#pragma region 解析input参数
    std::vector<std::string> inputArr = SplitString(input, ':');
    size_t length = inputArr.size();
    if (length <= 0) throw std::logic_error("输入参数错误：" + input);
    
    if (inputArr[0] == "cam" || inputArr[0] == "camera")
    {
        //相机的默认参数
        int index = 0, width = 640, height = 480;
        if (length >= 2)    index = std::stoi(inputArr[1]);
        if (length >= 3)    ParseArgForSize(inputArr[2], width, height);
        
        capture.open(index);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        LOG_INFO << "Input Source Camera [Index:" << index << ", Size:" 
            << capture.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << capture.get(cv::CAP_PROP_FRAME_HEIGHT) 
            << "]" << std::endl;
    }
    else if (inputArr[0] == "url" || inputArr[0] == "void")
    {
        if (length == 1)
        {
            throw std::invalid_argument("未指定 VIDEO 源路径：" + input);
            return EXIT_FAILURE;
        }

        capture.open(inputArr[1]);
        if (length >= 3)
        {
            int width = 640, height = 480;
            if (ParseArgForSize(inputArr[2], width, height))
            {
                capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            }
        }

        LOG_INFO << "Input Source Video [Index:" << inputArr[1] << ", Size:"
            << capture.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << capture.get(cv::CAP_PROP_FRAME_HEIGHT)
            << "]" << std::endl;
    }
    else
    {
        LOG_ERROR << "input 参数不支持，输入错误：" + input + "， 格式：-i (video|camera)[:value[:size]]" << std::endl;
        return EXIT_FAILURE;
    }
#pragma endregion

    if (!capture.isOpened())
    {
        LOG_ERROR << "无法打开视频或相机源：" << input << std::endl;
        return EXIT_FAILURE;
    }

#pragma region 解析output参数
    std::vector<std::string> outputArr = SplitString(output, ':');
    length = outputArr.size();
    if (length <= 0) throw std::logic_error("输出参数错误：" + output);

    size_t shared_size = 1920 * 1080 * 4;
    std::string shared_name = "source.bin";

    if (outputArr[0] == "share" || outputArr[0] == "shared")
    {
        if (length >= 2)
            shared_name = outputArr[1];
        if (length >= 3)
            shared_size = std::stoi(outputArr[2]);
    }
    else
    {
        LOG_ERROR << "output 参数不支持，输入错误：" + output + "， 格式：-o (shared)[:name[:size]]" << std::endl;
        return EXIT_FAILURE;
    }
    LOG_INFO << "Output Source Shared [Name:" << shared_name << ", Size:" << shared_size << "]" << std::endl;
#pragma endregion      

    LPVOID srcBuffer = NULL;
    HANDLE srcMapFile = NULL;    
    if (!CreateOnlyWriteMapFile(srcMapFile, srcBuffer, shared_size + VSD_SIZE, shared_name.c_str()))
    {
        LOG_ERROR << "创建共享内存失败" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat srcFrame;
    VideoSourceData vsData;

    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int format = capture.get(cv::CAP_PROP_FORMAT);
    int fps = capture.get(cv::CAP_PROP_FPS);
    std::string title = "Video Source [" + input + "]";

    LOG_INFO << "Capture Size:" << width << "," << height << "  FPS:" << fps << "  Count:" << count << "  Format:" << format << std::endl;

    double use_time = 0.0;
    int waitDelay = WaitKeyDelay;
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

        fps = capture.get(cv::CAP_PROP_FPS);

        //写入到共享内存
        vsData.fid += 1;
        vsData.copy(srcFrame);
        std::copy((uint8_t*)&vsData, (uint8_t*)&vsData + VSD_SIZE, (uint8_t*)srcBuffer);
        std::copy((uint8_t*)srcFrame.data, (uint8_t*)srcFrame.data + vsData.length, (uint8_t*)srcBuffer + VSD_SIZE);

        t1 = std::chrono::high_resolution_clock::now();
        use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

        waitDelay = WaitKeyDelay - use_time;
        waitDelay = waitDelay <= 0 ? 1 : waitDelay;

        std::cout << "\33[K\r[ INFO] " << "Current FPS:" << fps << "  FrameID:" << vsData.fid << "  Read/Write Use:" << use_time << "ms";

        cv::waitKey(waitDelay);
        if (isShow) cv::imshow(title, srcFrame);
    }
    std::cout << std::endl;

    capture.release();
    cv::destroyAllWindows();

    //撤消地址空间内的视图
    if (srcBuffer != NULL)   UnmapViewOfFile(srcBuffer);
    //关闭共享文件句柄
    if (srcMapFile != NULL)  CloseHandle(srcMapFile);

    LOG_INFO << "Exiting ..." << std::endl;
    Sleep(500);

    return EXIT_SUCCESS;
}


