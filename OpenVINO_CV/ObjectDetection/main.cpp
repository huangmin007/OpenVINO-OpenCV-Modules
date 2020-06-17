// 对象检测
//

#include <iostream>
#include <fstream>
#include "cmdline.h"
#include "object_detection.hpp"
#include "static_functions.hpp"
#include "input_source.hpp"

#define G_OBJECT false

using namespace space;

int main(int argc, char** argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)space::CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "通用对象检测模块 说明：" << std::endl;
    std::cout << "1.支持一个网络层的输入，输入参数为：BGR(U8) [BxCxHxW] = [1x3xHxW]" << std::endl;
    std::cout << "2.支持一个或多个网络层的输出，输出为共享内存数据，与源网络层输出数据一致，共享名称为网络输出层名称" << std::endl;
    std::cout << "3.对象检测一般输出的数据为标定的对象边界线，也就是 Box " << std::endl;
    std::cout << "4.在 show 模式下，只解析了默认两例作为示例：a).单层输出[1,1,N,7]  b).两层输出[N,5][N]" << std::endl;
    std::cout << "\r\n" << std::endl;

#pragma region cmdline参数设置解析
//---------------- 第一步：设计输入参数 --------------confidence
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "camera:0:1280x720");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：face-detection-adas-0001:FP32:CPU 或 face-detection-adas-0001:FP16:GPU", false, "face-detection-retail-0005:FP32:CPU");
    //face-detection-adas-0001,person-detection-retail-0013,face-detection-0105,boxes/Split.0:labels
    args.add<std::string>("output_layers", 'o', "(output layer names)多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，区分大小写，格式：layerName:layerName:...", false, "");
    args.add<float>("conf", 'c', "检测结果的置信度阈值(confidence threshold)", false, 0.5);
    args.add<bool>("reshape", 'r', "重塑输入层，使输入源内存映射到网络输入层实现共享内存数据，不进行数据源缩放和拷贝", false, true);

#if G_OBJECT
    args.add<std::string>("m0", 0, "用于 AI识别检测 的 联级网络模型名称/文件(.xml)和目标设备；示例："
        "landmarks-regression-retail-0009:FP32:CPU,head-pose-estimation-adas-0001:FP32:CPU", false, "");// "facial-landmarks-35-adas-0002:FP32:CPU");
    //facial-landmarks-35-adas-0002
#endif
    //args.add<bool>("async", 0, "是否异步分析识别", false, true);
#ifdef _DEBUG
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif
    args.set_program_name("Object Detection");

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

    bool reshape = args.get<bool>("reshape");
    bool show = args.get<bool>("show"); 
    float conf = args.get<float>("conf");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::vector<std::string> output_layers = SplitString(args.get<std::string>("output_layers"), ':');

#pragma region ReadLebels
    std::vector<std::string> labels;    
    std::string labels_file = "models\\" + model["model"] + "\\labels.txt";
    std::fstream file(labels_file, std::fstream::in);
    if (!file.is_open())
    {
        LOG("WARN") << "标签文件不存在 " << labels_file << " 文件 ...." << std::endl;
    }
    else
    {
        std::string line;
        while (std::getline(file, line))
        {
            labels.push_back(line);
        }
        file.close();
    }
    
#pragma endregion

#pragma endregion

    cv::Mat frame;
    InputSource inputSource("od_source.bin");
    if (!inputSource.open(args.get<std::string>("input")))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        InferenceEngine::Core ie;

        //Parent
        ObjectDetection detector(output_layers, show);
        detector.SetParameters({1,1,true}, conf, labels);
        detector.ConfigNetwork(ie, args.get<std::string>("model"), reshape);
        //detector.ConfigNetwork(ie, "models\\A_Test\\mobilenet_v1_0.25_128_frozen.xml", "CPU", !reshape);

#if G_OBJECT
        //Sub
        //std::vector<std::string> model_infos = SplitString(args.get<std::string>("m0"), ',');
        std::vector<std::string> model_infos = {
            "facial-landmarks-35-adas-0002:FP32:CPU",
            //"landmarks-regression-retail-0009:FP32:CPU",
            //"head-pose-estimation-adas-0001:FP32:CPU",
            //"face-reidentification-retail-0095:FP32:CPU"
        };
        for (int i = 0; i < model_infos.size(); i++)
        {
            ObjectRecognition* recognition = new ObjectRecognition({}, show);
            recognition->SetParameters({ 8, 1, true });
            recognition->ConfigNetwork(ie, model_infos[i], false);

            detector.AddSubNetwork(recognition);
        }
#endif

        inputSource.read(frame);
        detector.RequestInfer(frame);

        std::vector<ObjectDetection::Result> results;
        std::stringstream title;
        title << "[Source] Object Detection [" << model["full"] << "]";

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
            use_time << "Infer/Frame Use Time:" << infer_use_time <<  "/" << frame_use_time << "ms  fps:" << detector.GetFPS();
            std::cout << "\33[2K\r[ INFO] " << use_time.str();

#if _DEBUG
            cv::imshow(title.str(), frame);
#endif

            if (!inputSource.read(frame))
            {
                LOG("WARN") << "读取数据帧失败 ... " << std::endl;
                Sleep(15);
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

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
