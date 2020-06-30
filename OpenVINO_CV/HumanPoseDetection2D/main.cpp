// HumanPoseEstimation3D.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <chrono>
#include "human_pose_detection.hpp"

#include <input_source.hpp>
#include <static_functions.hpp>

#include <ie_compound_blob.h>

#include "cmdline.h"

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

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "camera:0:1280x720");// "shared:sharedMemory");
    //args.add<std::string>("output_layer_names", 'o', "多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，"
    //    "区分大小写，格式：layerName:layerName:...", false, "Mconv7_stage2_L1:Mconv7_stage2_L2");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：human-pose-estimation-0001:FP32:CPU 或 human-pose-estimation-0001:FP16:GPU", false, "human-pose-estimation-0001:FP32:CPU");

    //args.add<bool>("reshape", 'r', "重塑输入层，使输入源内存映射到网络输入层实现共享内存数据，不进行数据源缩放和拷贝", false, true);

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
        InferenceEngineInformation(args.get<std::string>("model"));
        return EXIT_SUCCESS;
    }
    if (!isParser)
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    bool show = args.get<bool>("show");
    //bool reshape = args.get<bool>("reshape");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    //std::vector<std::string> output_layer_names = SplitString(args.get<std::string>("output_layer_names"), ':');

#pragma endregion
    cv::Mat frame;
    InputSource inputSource("pose_source.bin");
    if (!inputSource.open(args.get<std::string>("input")))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    inputSource.read(frame);
    if (frame.channels() == 1)
    {
        LOG("ERROR") << "Frame Channels  ... " << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        InferenceEngine::Core ie;
        HumanPoseDetection detector(show, frame.channels() == 4 ? InferenceEngine::ColorFormat::BGRX : InferenceEngine::ColorFormat::BGR);
        detector.ConfigNetwork(ie, args.get<std::string>("model"));

        detector.RequestInfer(frame);

        std::stringstream title;
        title << "[Source] Human Pose Detection [" << model["full"] << "]";

        bool isResult = false;
        std::stringstream txt;
        std::stringstream use_time;

        int delay = WaitKeyDelay;
        double frame_use_time = 0.0f;
        auto t0 = hc::now();
        auto t1 = hc::now();

        while (true)
        {
            if (!IsRunning) break;
            t0 = hc::now();

            use_time.str("");
            use_time << "Infer/Frame Use Time:" << detector.GetInferUseTime() << "/" << frame_use_time << "ms  FPS:" << detector.GetFPS();
            std::cout << "\33[2K\r[ INFO] " << use_time.str();

#ifdef _DEBUG
            cv::imshow(title.str(), frame);
#endif

            if (!inputSource.read(frame))
            {
                //LOG("WARN") << "读取数据帧失败 ... " << std::endl;
                Sleep(15);
                continue;
            }

            //detector.RequestInfer(frame);

            t1 = hc::now();
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
