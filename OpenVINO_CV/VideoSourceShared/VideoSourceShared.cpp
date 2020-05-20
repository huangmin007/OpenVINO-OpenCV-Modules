// VideoSourceShared.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include "cmdline.h"
#include "OpenVINO_CV.h"


int main(int argc, char **argv)
{
    cmdline::parser args;

    args.add<std::string>("help", 'h', "参数说明", false, "");
    args.add<std::string>("input", 'i', "输入视频或相机源，示例：-i url:<path> 或 -i cam:<index>", false, "cam:0");
    args.add<std::string>("size", 's', "设置视频源或是相机的输出尺寸", false, "640,480");
    args.add<std::string>("fn", '\0', "内存共享文件名称", false, "source.bin");
    args.add<uint32_t>("fs", '\0', "分配给共享内存的空间大小，默认为：(1920 x 1080 x 4)Bytes", false, (1920 * 1080 * 4));
    args.add<bool>("show", '\0', "是否显示视频窗口，用于调试", false, false);
    args.set_program_name("VideoSourceShared");

    bool isParser = args.parse(argc, argv);
    
    if (args.exist("help"))// || argc == 1)
    {
        std::cerr << args.usage();
        return EXIT_SUCCESS;
    }
    if (!isParser) 
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cerr << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

    cv::VideoCapture capture;
    std::string input = args.get<std::string>("input");
    if (input.find("cam:") == 0)
    {
        //相机索引
        int index = 0, w = 640, h = 480;
        try
        {
            index = std::stoi(input.substr(4).c_str());
            std::string size = args.get<std::string>("size");

            //获取配置宽高信息
            int x = size.find(',') > 0 ? size.find(',') : size.find('x') > 0 ? size.find('x') : -1;
            if (x > 0)
            {
                w = std::stoi(size.substr(0, x));
                h = std::stoi(size.substr(x + 1));
            }
        }
        catch (std::exception & e)
        {
            std::cerr << "[ERROR] " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        //打开相机，设置相机宽高属性
        capture.open(index);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, w);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        std::cout << "[ INFO] Setting Camera Index:" << index << "  Size:" << w << "," << h << std::endl;
    }
    else if (input.find("url:") == 0)
    {
        //视频 URL 地址
        std::string url = input.substr(4);
        capture.open(url);

        std::cout << input << std::endl;
    }
    else
    {
        std::cerr << "[ERROR] 参数输入错误：" + input + "， 示例：-i url:<path> OR -i cam:<index>" << std::endl;
        return EXIT_FAILURE;
    }

    if (!capture.isOpened())
    {
        std::cerr << "[ERROR] 无法打开视频源：" << input << std::endl;
        return EXIT_FAILURE;
    }

    //if (args.get<bool>("show")) cv::namedWindow("Video Source");

    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int format = capture.get(cv::CAP_PROP_FORMAT);
    int fps = capture.get(cv::CAP_PROP_FPS);

    std::cout << "[ INFO] Capture Size:" << width << "," << height << "  FPS:" << fps << "  Count:" << count << "  Format:" << format << std::endl;

    bool isLastFrame = false;
    cv::Mat srcFrame;

    LPVOID srcBuffer = NULL;
    HANDLE srcMapFile = NULL;
    MapFileData srcHeadInfo;
    if (!CreateOnlyWriteMapFile(srcMapFile, srcBuffer, 
        args.get<uint32_t>("fs") + sizeof(MapFileData),
        args.get<std::string>("fn").c_str()))
    {
        std::cerr << "[ERROR] 创建共享内存失败" << std::endl;
        return EXIT_FAILURE;
    }

    double use_time = 0.0;
    int waitDelay = WaitKeyDelay;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    bool isShow = args.get<bool>("show");

    while (true)
    {
        if (!IsRunning) break;

        t0 = std::chrono::high_resolution_clock::now();
        if (!capture.read(srcFrame))
            if (srcFrame.empty()) break;

        //写入到共享内存
        if (!WriteMapFile(srcMapFile, srcBuffer, &srcHeadInfo, srcFrame))
            std::cerr << "[ERROR] 写入到共享内存失败" << std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

        waitDelay = WaitKeyDelay - use_time;
        waitDelay = waitDelay <= 0 ? 1 : waitDelay;

        std::cout << "\33[K\r[ INFO] " << "Current Frame Read/Write Use :" << use_time << "ms";

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


