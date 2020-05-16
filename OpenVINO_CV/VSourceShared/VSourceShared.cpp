// VSourceShared.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

// 启动参数设计
//  input args -i url:aa  -i cam:0
//  run args
//  output args


//#include <string>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <errno.h>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <signal.h>
#include "OpenVINO_CV.h"
//#include <cstring>

//窗体标题
#define     WINDOW_TITLE    "source"
//delay
#define     DELAY           30
//分配源共享内存大小
#define     SOURCE_BUFFER_SIZE      1024 * 1024 * 16

bool isRunning = true;

DEFINE_string(i, "cam:0", "输入视频或图片源路径");
DEFINE_string(mn, "source.bin", "内存共享名称");
DEFINE_string(size, "640,480", "输出尺寸大小");
DEFINE_bool(imshow, false, "是否显示视频源");
DEFINE_double(r, 0.24, "用于AI分析的源比例");
DEFINE_string(rmn, "source.ai.bin", "用于AI分析的内存共享名称");

BOOL WINAPI HandlerRoutine(DWORD key)
{
    if (key == CTRL_C_EVENT)
        isRunning = false;

    return true;
}

void onMouseEventHandler(int event, int x, int y, int flags, void* ustc)
{
    std::cout << "mouse event::" << event << std::endl;
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        
    }
}

//创建只写的内存共享文件
//创建成功 返回 true
static bool CreateOnlyWriteMapFile(HANDLE &handle, LPVOID &buffer, uint size, const char *name)
{
    buffer = NULL;
    handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCWSTR)name);
    if (handle != NULL)
    {
        buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);
        if (buffer == NULL)
        {
            std::cout << "[Error]共享内存块 " << name << ", MapViewOfFile 失败" << std::endl;
            CloseHandle(handle);
            return false;
        }        
    }
    else
    {
        std::cout << "[Error]创建共享内存块 " << name <<  " 失败" << std::endl;
        return false;
    }

    std::cout << "[Info]创建共享内存块 " << name << " 完成，大小：" << size << "Bytes" << std::endl;
    return true;
}

//写入共享内存
//返回写入耗时时间ms
static double WriteMapFile(LPVOID &buffer, const SourceMapFileInfo *info, const cv::Mat frame)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    static int offset = sizeof(SourceMapFileInfo);

    std::copy((uint8_t*)info, (uint8_t*)info + offset, (uint8_t *)buffer);
    std::copy((uint8_t*)frame.data, (uint8_t*)frame.dataend, (uint8_t*)buffer + offset);

    //std::cout << "FFw:" << info->width << " h:" << info->height << " c:" << info->channels << " t:" << info->type << " size:" << offset << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    double decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

    return decode_time;
}


int main(int argc, char** argv)
{
    //SetConsoleCtrlHandler(HandlerRoutine, TRUE);
    
    google::ParseCommandLineFlags(&argc, &argv, false);
    std::cout << "Video Source:" << FLAGS_i << std::endl;
    
    cv::VideoCapture capture;
    if (FLAGS_i.find("cam:") == 0)
    {
        int index = 0, w = 640, h = 480;
        try
        {
            index = std::stoi(FLAGS_i.substr(4).c_str());
            int x = FLAGS_size.find(',') < 0 ? FLAGS_size.find('x') : -1;
            if (x > 0)
            {
                w = std::stoi(FLAGS_size.substr(0, x));
                h = std::stoi(FLAGS_size.substr(x + 1));
            }
        }
        catch(std::exception &e)
        {
            std::cerr << "[Error]" << e.what() << std::endl;
            system("pause");
            return EXIT_FAILURE;
        }
        capture.open(index);

        capture.set(cv::CAP_PROP_FRAME_WIDTH, w);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    }
    else if (FLAGS_i.find("url:") == 0)
    {
        std::string url = FLAGS_i.substr(4);
        std::cout << "URL:" << url << std::endl;

        capture.open(url);
    }
    else
    {
        std::cerr << "[Error]" << "输入参数错误 - i url : path OR - i cam : 0" << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }
    
    if (!capture.isOpened())
    {
        std::cout << "[Error]无法打开视频源 " << FLAGS_i << std::endl;
        system("pasue");
        return EXIT_FAILURE;
    }

    if (FLAGS_imshow)
    {
        cv::namedWindow(WINDOW_TITLE);
        //cv::setMouseCallback("source", onMouseEventHandler, 0);
    }

    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int format = capture.get(cv::CAP_PROP_FORMAT);
    int fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "Video Size:" << width << "," << height << std::endl;
    std::cout << "Video Count:" << count << std::endl;
    std::cout << "Video Format:" << format << std::endl;
    std::cout << "Video FPS:" << fps << std::endl;
    cv::Size dsize(width * FLAGS_r, height * FLAGS_r);
    
    uint32_t fid = 0;
    bool isLastFrame = false;
    cv::Mat srcFrame, aiFrame;

    SourceMapFileInfo srcHeadInfo, aiHeadInfo;
    LPVOID srcBuffer = NULL, aiBuffer = NULL;
    HANDLE srcMapFile = NULL, aiMapFile = NULL;

    CreateOnlyWriteMapFile(srcMapFile, srcBuffer, SOURCE_BUFFER_SIZE, FLAGS_mn.c_str());
    CreateOnlyWriteMapFile(aiMapFile, aiBuffer, SOURCE_BUFFER_SIZE * FLAGS_r, FLAGS_rmn.c_str());
    
    while (true)
    {
        if (!isRunning) break;
        auto t0 = std::chrono::high_resolution_clock::now();

        if (!capture.read(srcFrame))
        {
            if (srcFrame.empty())
            {
                isLastFrame = true;     //视频播放结束
            }
            else
            {
                std::cout << "[Error]读取视频帧错误" << FLAGS_i << std::endl;
                system("pasue");
                return EXIT_FAILURE;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        std::cout << "\33[2K\r" << "Read/Decode Frame:" << decode_time << "ms";

        auto t2 = std::chrono::high_resolution_clock::now();
        //Write To Memory
        if (srcMapFile != NULL && srcBuffer != NULL)
        {
            srcHeadInfo.fid = fid;
            srcHeadInfo << srcFrame;

            double decode_time = WriteMapFile(srcBuffer, &srcHeadInfo, srcFrame);
            //std::cout << "\t" << "Write To Memory:" << decode_time << "ms " << srcFrame.rows * srcFrame.cols * srcFrame.channels() << "Bytes";
        }
        if (aiMapFile != NULL && aiBuffer != NULL)
        {
            cv::resize(srcFrame, aiFrame, dsize);
            aiHeadInfo.fid = fid;
            aiHeadInfo << aiFrame;

            double decode_time = WriteMapFile(aiBuffer, &aiHeadInfo, aiFrame);
            //std::cout << "\t" << "Write To Memory:" << decode_time << "ms " << aiFrame.rows * aiFrame.cols * aiFrame.channels() << "Bytes";
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double write_time = std::chrono::duration_cast<ms>(t3 - t2).count();
        std::cout << "\tWrite To Memory:" << write_time << "ms";

        const int key = cv::waitKey(DELAY) & 0xFF;
        if (key == 'c') FLAGS_imshow ^= true;

        if (FLAGS_imshow)
            cv::imshow(WINDOW_TITLE, srcFrame);

        fid++;
        if (isLastFrame) break;
    }

    capture.release();
    cv::destroyAllWindows();

    if (aiMapFile != NULL)  CloseHandle(aiMapFile);
    if (srcMapFile != NULL)  CloseHandle(srcMapFile);

    std::cout << "Completed!" << std::endl;
    system("pause");

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
