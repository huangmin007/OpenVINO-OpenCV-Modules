// OpenPoseModel.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include <Windows.h>
#include "OpenVINO_CV.h"

DEFINE_string(rmn, "source.ai.bin", "用于AI分析的内存共享名称");
uint8_t buffer[1024 * 1024 * 8];

int main(int argc, char** argv)
{
    google::ParseCommandLineFlags(&argc, &argv, false);
    //system("pause");

    HANDLE hMapFile = OpenFileMapping(FILE_MAP_READ, NULL, (LPCWSTR)FLAGS_rmn.c_str());
    if (hMapFile == NULL)
    {
        std::cout << "共享内存块未创建：" << FLAGS_rmn << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }

    LPVOID pBuffer = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
    if (pBuffer == NULL)
    {
        std::cout << "共享内存块 MapViewOfFile 失败" << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }
    
    cv::Mat frame;
    SourceMapFileInfo info;
    uint8_t head[32];
    std::cout << "size::" << sizeof(info) << std::endl;
    char str[16];
    static int offset = sizeof(SourceMapFileInfo);
    
    while (true)
    {
        std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + offset, reinterpret_cast<uint8_t*>(&info));
        std::cout << "w:" << info.width << " h:" << info.height << " c:" << info.channels << " t:" << info.type << std::endl;

        //std::copy((uint8_t*)pBuffer + offset, (uint8_t*)pBuffer + info.width * info.height * info.channels, frame.data);
        //info = reinterpret_cast<SourceMapFileInfo*>(head);
        //cv::Mat t(info.width, info.height, info.channls);

        //std::copy((char*)pBuffer, (char*)pBuffer + 8, head);
        //uint8_t szBuffer[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        //std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + 8, head);

        //for (int i = 0; i < 8; i++)
        //    std::cout << (void *)head[i] << " <";
        //std::cout << std::endl;
        
        cv::waitKey(30);
    }

    system("pause");
    std::cout << "Hello World!\n";
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
