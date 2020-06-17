// ObjectRecognition.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include "cmdline.h"
#include "object_recognition.hpp"
#include "static_functions.hpp"
#include "input_source.hpp"

using namespace space;

int main(int argc, char** argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)space::CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }


#pragma region cmdline参数设置解析
    //---------------- 第一步：设计输入参数 --------------confidence
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "shared:od_source.bin");
    args.add<std::string>("input_shared", 0, "输入共享内存名称", false, "527");

    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，", false,
        "facial-landmarks-35-adas-0002:FP32:CPU"); //facial-landmarks-35-adas-0002 , landmarks-regression-retail-0009
    args.add<std::string>("output_layers", 'o', "(output layer names)多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，区分大小写，格式：layerName:layerName:...", false, "");
    

#ifdef _DEBUG
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif
    args.set_program_name("Object Recognition");

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
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::vector<std::string> input_shared = SplitString(args.get<std::string>("input_shared"), ':');
    std::vector<std::string> output_layers = SplitString(args.get<std::string>("output_layers"), ':');
#pragma endregion

    cv::Mat frame;
    InputSource inputSource;
    if (!inputSource.open(args.get<std::string>("input")))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        InferenceEngine::Core ie;
        ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES} }, "CPU");

        ObjectRecognition detector(input_shared, output_layers, show);
        detector.SetParameters({4, 1, true});
        detector.ConfigNetwork(ie, args.get<std::string>("model"), false);

        inputSource.read(frame);
        detector.RequestInfer(frame);

        std::stringstream use_time;
        int delay = WaitKeyDelay;
        double infer_use_time = 0.0f;
        double frame_use_time = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();

        while (true)
        {
            if (!IsRunning) break;
            t0 = std::chrono::high_resolution_clock::now();

            use_time.str("");
            use_time << "Infer/Frame Use Time:" << infer_use_time << "/" << frame_use_time << "ms  fps:" << detector.GetFPS();
            std::cout << "\33[2K\r[ INFO] " << use_time.str();

#if _DEBUG
                cv::imshow("[Source] Object Recognition", frame);
#endif

            if (!inputSource.read(frame))
            {
                Sleep(10);
                continue;
            }

            detector.RequestInfer(frame);

            t1 = std::chrono::high_resolution_clock::now();
            infer_use_time = detector.GetInferUseTime();
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
    LOG("INFO") << "Exiting ... " << std::endl;

    Sleep(500);
    return EXIT_SUCCESS;
}

