// VSourceShared.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//


//  input args -i url:aa  -i cam:0
//  run args
//  output args


#include <string>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <errno.h>
//#include <opencv2/opencv.hpp>
#include <Windows.h>

#include "args.hpp"
#include "OpenVINO_CV.h"
#include "LoggerConfig.h"


#define     WINDOW_TITLE            "Video Source"    //窗体标题

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "Chinese-simplified");
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cout << "Unable to install handler!\n" << std::endl;
        return EXIT_FAILURE;
    }
    
    //google::SetVersionString(VERSION);
    //google::SetUsageMessage(GetUsageMessage());
    if (!ParseAndCheckCommandLine(argc, argv))
    {
        std::cout << "bbb" << std::endl;
        //log4cplus::Logger::shutdown();
        gflags::ShutDownCommandLineFlags();
        std::cout << "ccc" << std::endl;

        return EXIT_SUCCESS;
        exit(EXIT_SUCCESS);
    }
    
    //google::ParseCommandLineFlags(&argc, &argv, true);
#if 0

    Log4CplusConfigure();
    LOG4CPLUS_INFO(logger, "Video Source Shared");    
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING("Video Source:" + FLAGS_i));
    //throw std::logic_error("相机参数配置错误: ");

    //配置 VideoCapture
    cv::VideoCapture capture;
    if (FLAGS_i.find("cam:") == 0)
    {
        //相机索引
        int index = 0, w = 640, h = 480;
        try
        {
            index = std::stoi(FLAGS_i.substr(4).c_str());
            //获取配置宽高信息
            int x = FLAGS_s.find(',') > 0 ? FLAGS_s.find(',') : FLAGS_s.find('x') > 0 ? FLAGS_s.find('x') : -1;
            if (x > 0)
            {
                w = std::stoi(FLAGS_s.substr(0, x));
                h = std::stoi(FLAGS_s.substr(x + 1));
            }
        }
        catch(std::exception &e)
        {
            LOG4CPLUS_ERROR(logger, LOG4CPLUS_STRING_TO_TSTRING(e.what()));
            system("pause");
            return EXIT_FAILURE;
        }

        //打开相机，设置相机宽高属性
        capture.open(index);        
        capture.set(cv::CAP_PROP_FRAME_WIDTH, w);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, h);

        std::stringstream info;
        info << "Setting Camera Index: " << index << "  Size: " << w << "," << h;
        LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(info.str()));
    }
    else if (FLAGS_i.find("url:") == 0)
    {
        //视频 URL 地址
        std::string url = FLAGS_i.substr(4);
        capture.open(url);

        LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(FLAGS_i));
    }
    else
    {
        LOG4CPLUS_ERROR(logger, LOG4CPLUS_STRING_TO_TSTRING("输入参数错误：" + FLAGS_i + "， 示例：-i url:path OR -i cam:0"));
        return EXIT_FAILURE;
    }
    
    if (!capture.isOpened())
    {
        LOG4CPLUS_ERROR(logger, "无法打开视频源");
        return EXIT_FAILURE;
    }

    if (!FLAGS_noshow)   cv::namedWindow(WINDOW_TITLE);

    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int format = capture.get(cv::CAP_PROP_FORMAT);
    int fps = capture.get(cv::CAP_PROP_FPS);

    std::stringstream info;
    info << "Output Info Size:" << width << "," << height;
    info << "  FPS:" << fps;
    info << "  Count:" << count;
    info << "  Format:" << format;
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(info.str()));
    
    bool isLastFrame = false;
    cv::Mat srcFrame;

    LPVOID srcBuffer = NULL;
    HANDLE srcMapFile = NULL;
    MapFileHeadInfo srcHeadInfo;

    if (!CreateOnlyWriteMapFile(srcMapFile, srcBuffer, FLAGS_fs, FLAGS_fn.c_str()))
    {
        LOG4CPLUS_ERROR(logger, LOG4CPLUS_STRING_TO_TSTRING("创建共享内存失败"));
        return EXIT_FAILURE;
    }
    
    while (true)
    {
        if (!IsRunning) break;
        auto t0 = std::chrono::high_resolution_clock::now();

        if (!capture.read(srcFrame))
        {
            if (srcFrame.empty())
            {
                isLastFrame = true;     //视频播放结束
            }
            else
            {
                LOG4CPLUS_ERROR(logger, LOG4CPLUS_STRING_TO_TSTRING("读取视频帧错误：" + FLAGS_i));
                return EXIT_FAILURE;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        std::cout << "\33[2K\r" << "Read/Decode Frame:" << decode_time << "ms";

        //写入到共享内存
        auto t2 = std::chrono::high_resolution_clock::now();
        if (!WriteMapFile(srcMapFile, srcBuffer, &srcHeadInfo, srcFrame))
        {
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        double write_time = std::chrono::duration_cast<ms>(t3 - t2).count();
        std::cout << "\tWrite To Memory:" << write_time << "ms";

        double use_time = std::chrono::duration_cast<ms>(t3 - t0).count();
        int delay = WaitKeyDelay - use_time;
        delay = delay < 0 ? 1 : delay;

        cv::waitKey(delay);
        if (!FLAGS_noshow) cv::imshow(WINDOW_TITLE, srcFrame);

        if (isLastFrame) break;
    }

    capture.release();
    cv::destroyAllWindows();

	//撤消地址空间内的视图
	if(srcBuffer != NULL) UnmapViewOfFile(srcBuffer);
    //关闭共享文件句柄
    if (srcMapFile != NULL)  CloseHandle(srcMapFile);

    LOG4CPLUS_INFO(logger, "Exiting ... ");
    log4cplus::Logger::shutdown();
#endif
    Sleep(500);

    return EXIT_SUCCESS;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
