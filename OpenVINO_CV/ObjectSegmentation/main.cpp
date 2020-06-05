// InstanceSegmentation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// Object/Screen/Instance Segmentation 对象/场景/实例分割 (Open Model Zoo)

#include <iostream>
#include <fstream>
#include "cmdline.h"
#include "object_segmentation.hpp"
#include "static_functions.hpp"
#include "input_source.hpp"

using namespace space;
using namespace InferenceEngine;


int main(int argc, char **argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)space::CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "通用对象(场景/实例)分割模块 说明：" << std::endl;
    std::cout << "1.图像分割主要有两种类型：语义分割和实例分割；在语义分割中，同一类型的所有物体都使用一个类标签进行标记，而在实例分割中，相似的物体使用自己的独立标签。" << std::endl;
    std::cout << "2.支持一个网络层的输入，输入参数为：BGR(U8) [BxCxHxW] = [1x3xHxW]" << std::endl;
    std::cout << "3.支持一个或多个网络层的输出，输出为共享内存数据，与源网络层输出数据一致，共享名称为网络输出层名称" << std::endl;
    std::cout << "4.在 show 模式下，只解析了默认一例作为示例：a).单层输出 [B,C,H,W] 或 [B,H,W]" << std::endl;
    std::cout << "\r\n" << std::endl;

#pragma region cmdline参数设置解析
    //---------------- 第一步：设计输入参数 --------------confidence
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "video:video/video_03.mp4");// "camera:0:1280x720", video:video/video_03.mp4
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：road-segmentation-adas-0001:FP32:CPU 或 road-segmentation-adas-0001:FP16:GPU", false, "road-segmentation-adas-0001:FP32:CPU");
    //road-segmentation-adas-0001
    args.add<std::string>("output_layer_names", 'o', "(output layer name)多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，区分大小写，格式：layerName:layerName:...", false, "");

    //args.add<bool>("async", 0, "是否异步分析识别", false, true);
#ifdef _DEBUG
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif
    args.set_program_name("Object Segmentation");

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

    bool reshape = true;
    bool show = args.get<bool>("show");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::vector<std::string> output_layer_names = SplitString(args.get<std::string>("output_layer_names"), ':');
#pragma endregion

    cv::Mat frame;
    InputSource inputSource("os_source.bin");
    if (!inputSource.open(args.get<std::string>("input")))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        InferenceEngine::Core ie;
        ObjectSegmentation detector(output_layer_names, show);
        detector.ConfigNetwork(ie, args.get<std::string>("model"), reshape);

        inputSource.read(frame);
        detector.RequestInfer(frame);

        std::stringstream title;
        title << "[Source] Object Segmentation [" << model["full"] << "]";

        std::stringstream use_time;

        int delay = 40;
        double frame_use_time = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();

        while (true)
        {
            if (!IsRunning) break;
            t0 = std::chrono::high_resolution_clock::now();

            use_time.str("");
            use_time << "Infer/Frame Use Time:" << detector.GetInferUseTime() << "/" <<  frame_use_time << "ms  FPS:" << detector.GetFPS();
            std::cout << "\33[2K\r[ INFO] " << use_time.str();

            if (show)
            {
                //不能在源图上绘图，会影响识别
                //cv::putText(frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
                cv::imshow(title.str(), frame);
            }

            if (!inputSource.read(frame))
            {
                LOG("WARN") << "读取数据帧失败，等待读取下一帧 ... " << std::endl;
                Sleep(15);
                continue;
            }

            detector.RequestInfer(frame);

            t1 = std::chrono::high_resolution_clock::now();
            frame_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
            delay = 40.0f - frame_use_time;
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
