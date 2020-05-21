// HumanPoseEstimation3D.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "cmdline.h"
#include "OpenVINO_CV.h"

int main(int argc, char **argv)
{
    //解析参数
    cmdline::parser args;

    args.add<std::string>("help", 'h', "参数说明", false, "");
    args.add<std::string>("model", 'm', "人体姿势估计模型（.xml）文件", false, "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml");
    args.add<std::string>("device", 'd', "指定用于 人体姿势评估 的目标设备", false, "CPU", cmdline::oneof<std::string>("CPU", "GNA" "GPU"));

    args.add<std::string>("rfn", 0, "用于读取AI分析的数据内存共享名称", false, "source.bin");
    args.add<std::string>("wfn", 0, "用于存储AI分析结果数据的内存共享名称", false, "pose3d.bin");
    args.add<uint32_t>("wfs", 0, "用于存储AI分析结果数据的内存大小，默认：1024*1024(Bytes)", false, 1024 * 1024);

    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
    args.add<bool>("output", 0, "在控制台上输出推断结果信息", false, false);

    args.add<bool>("pc", 0, "启用每层性能报告", false, false);
    args.add<std::string>("um", 0, "最初显示的监视器列表", false, "");
    args.set_program_name("HumanPoseEstimation3D");

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

    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cerr << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

    bool show = args.get<bool>("show");
    bool output = args.get<bool>("output");
    std::string model = args.get<std::string>("model");
    std::string device = args.get<std::string>("device");

    //引擎信息
    std::cout << "[ INFO] InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;
    std::cout << "[ INFO] AI模型文件：" << model << std::endl;
    std::cout << "[ INFO] 获取/创建内存共享数据文件 ......" << std::endl;

    std::cout << "[ INFO] 获取/创建内存共享数据文件 ......" << std::endl;
    LPVOID sBuffer = NULL, dBuffer = NULL;
    HANDLE sMapFile = NULL, dMapFile = NULL;
    if (!GetOnlyReadMapFile(sMapFile, sBuffer, args.get<std::string>("rfn").c_str())) return EXIT_FAILURE;
    if (!CreateOnlyWriteMapFile(dMapFile, dBuffer, args.get<uint32_t>("wfs"), args.get<std::string>("wfn").c_str())) return EXIT_FAILURE;

    while (true)
    {
        if (!IsRunning) break;
    }

    std::cout << std::endl;
    cv::destroyAllWindows();

    //撤消地址空间内的视图
    if (sBuffer != NULL) UnmapViewOfFile(sBuffer);
    if (dBuffer != NULL) UnmapViewOfFile(dBuffer);
    //关闭共享文件句柄
    if (sMapFile != NULL) CloseHandle(sMapFile);
    if (dMapFile != NULL) CloseHandle(dMapFile);

    std::cout << "[ INFO] Exiting ..." << std::endl;
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
