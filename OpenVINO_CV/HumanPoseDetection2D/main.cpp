// HumanPoseEstimation3D.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <chrono>
#include "human_pose_detection.hpp"

#include <input_source.hpp>
#include <static_functions.hpp>

#include "cmdline.h"

#define USE_SHARED_BLOB false

using namespace space;

int main(int argc, char **argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cerr << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

#pragma region cmdline参数设置解析
    //---------------- 第一步：设计输入参数 --------------
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "cam:0:1920x1080");
    args.add<std::string>("output_layer_names", 'o', "多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，"
        "区分大小写，格式：layerName:layerName:...", false, "Mconv7_stage2_L1:Mconv7_stage2_L2");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：human-pose-estimation-0001:FP32:CPU 或 human-pose-estimation-0001:FP16:GPU", false, "human-pose-estimation-0001:FP32:CPU");

    args.add<bool>("reshape", 'r', "重塑输入层，使输入源内存映射到网络输入层实现共享内存数据，不进行数据缩放源", false, true);

#ifdef _DEBUG
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif
    args.set_program_name("HumanPoseDetection2D");

    LOG("INFO") << "参数解析 ... " << std::endl;
    //----------------- 第二步：解析输入参数 ----------------
    bool isParser = args.parse(argc, argv);
    if (args.exist("help"))
    {
        std::cerr << args.usage();
        return EXIT_SUCCESS;
    }
    if (args.exist("info"))
    {
        InferenceEngineInfomation(args.get<std::string>("model"));
        return EXIT_SUCCESS;
    }
    if (!isParser)
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    bool show = args.get<bool>("show");
    bool reshape = args.get<bool>("reshape");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::vector<std::string> output_layer_names = SplitString(args.get<std::string>("output_layer_names"), ':');

#pragma endregion
    cv::Mat frame;
    InputSource inputSource("pose_source.bin");
    if (!inputSource.open(args.get<std::string>("input")))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        InferenceEngine::Core ie;
        HumanPoseDetection detector(output_layer_names, show);
        detector.Configuration(ie, args.get<std::string>("model"), true, 1);

        std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));

        inputSource.read(frame);
        detector.RequestInfer(frame);

        std::stringstream title;
        title << "[Source] Human Pose Detection [" << model["full"] << "]";

        bool isResult = false;
        std::stringstream txt;
        std::stringstream use_time;

        int delay = WaitKeyDelay;
        double frame_use_time = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();

        while (true)
        {
            if (!IsRunning) break;
            t0 = std::chrono::high_resolution_clock::now();

            use_time.str("");
            use_time << "Frame Use Time:" << frame_use_time << "ms  FPS:" << detector.GetFPS();
            //std::cout << "\33[2K\r[ INFO] " << use_time.str();

            if (show)
            {
                cv::imshow(title.str(), frame);
            }

            if (!inputSource.read(frame))
            {
                LOG("WARN") << "读取数据帧失败 ... " << std::endl;
                Sleep(15);
                continue;
            }

            detector.RequestInfer(frame);

            t1 = std::chrono::high_resolution_clock::now();
            frame_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            delay = WaitKeyDelay - frame_use_time;
            if (delay <= 0) delay = 1;

            cv::waitKey(delay);
        }
    }
    catch (const std::exception& error)
    {
        LOG("ERROR") << error.what() << std::endl;
    }
    catch (...)
    {
        LOG("ERROR") << "发生未知/内部异常 ... " << std::endl;
    }

    std::cout << std::endl;

    inputSource.release();
    if (show) cv::destroyAllWindows();
    LOG("INFO") << "Exiting ..." << std::endl;
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
