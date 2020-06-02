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

#if 1
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

    bool show = args.get<bool>("show");
    //bool async = args.get<bool>("async");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::vector<std::string> output_layer_names = SplitString(args.get<std::string>("output_layer_names"), ':');
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
        ObjectSegmentation detector(output_layer_names, show);
        detector.config(ie, args.get<std::string>("model"), true);

        inputSource.read(frame);
        detector.start(frame);

        std::stringstream title;
        title << "Object Segmentation [" << model["full"] << "]";

        std::stringstream use_time;

        int delay = 40;
        double infer_use_time = 0.0f;
        double total_use_time = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();

        while (true)
        {
            if (!IsRunning) break;
            t0 = std::chrono::high_resolution_clock::now();

            if (!inputSource.read(frame))
            {
                LOG("WARN") << "读取数据帧失败，等待读取下一帧 ... " << std::endl;
                Sleep(15);
                continue;
            }

            infer_use_time = detector.getUseTime();

            use_time.str("");
            use_time << "Total/Inference Use Time:" << total_use_time << "/" << infer_use_time << "ms";

            if (show)
            {
                cv::putText(frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
                cv::imshow(title.str(), frame);
            }
            std::cout << "\33[2K\r[ INFO] " << use_time.str();

            t1 = std::chrono::high_resolution_clock::now();
            total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
            delay = 40.0f - total_use_time;
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
#else

#include "samples/common.hpp"
#include <samples/ocv_common.hpp>
#include <monitors\presenter.h>

int main(int argc, char* argv[])
{

    try
    {
        Core ie;

        CNNNetwork network = ie.ReadNetwork("models\\road-segmentation-adas-0001\\FP32\\road-segmentation-adas-0001.xml");
        ICNNNetwork::InputShapes inputShapes = network.getInputShapes();

        if (inputShapes.size() != 1)
            throw std::runtime_error("Demo supports topologies only with 1 input");

        const std::string& inName = inputShapes.begin()->first;
        SizeVector& inSizeVector = inputShapes.begin()->second;
        if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
            throw std::runtime_error("3-channel 4-dimensional model's input is expected");

        inSizeVector[0] = 1;  // set batch size to 1
        network.reshape(inputShapes);

        InputInfo& inputInfo = *network.getInputsInfo().begin()->second;
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.setLayout(Layout::NHWC);
        inputInfo.setPrecision(Precision::U8);

        const OutputsDataMap& outputsDataMap = network.getOutputsInfo();
        if (outputsDataMap.size() != 1) 
            throw std::runtime_error("Demo supports topologies only with 1 output");

        const std::string& outName = outputsDataMap.begin()->first;
        Data& data = *outputsDataMap.begin()->second;

        // 如果模型执行Arg Max，则其输出类型可以为I32，但是对于返回每个类的热图的模型，输出通常为FP32。
        // 重置精度以避免使用开关后处理处理不同类型
        data.setPrecision(Precision::FP32);
        const SizeVector& outSizeVector = data.getTensorDesc().getDims();
        int outChannels, outHeight, outWidth;
        switch (outSizeVector.size()) {
        case 3:
            outChannels = 0;
            outHeight = outSizeVector[1];
            outWidth = outSizeVector[2];
            break;
        case 4:
            outChannels = outSizeVector[1];
            outHeight = outSizeVector[2];
            outWidth = outSizeVector[3];
            break;
        default:
            throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are"
                "supported.");
        }

        ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");
        InferRequest inferRequest = executableNetwork.CreateInferRequest();

        InputSource cap;
        cap.open("video:video/video_03.mp4");
        

        float blending = 0.3f;
        constexpr char WIN_NAME[] = "segmentation";

        bool show = true;
        if (show) 
        {
            cv::namedWindow(WIN_NAME);
            int initValue = static_cast<int>(blending * 100);
            cv::createTrackbar("blending", WIN_NAME, &initValue, 100,
                [](int position, void* blendingPtr) {*static_cast<float*>(blendingPtr) = position * 0.01f; },
                &blending);
        }

        cv::Mat inImg, resImg, maskImg(outHeight, outWidth, CV_8UC3);
        std::vector<cv::Vec3b> colors(arraySize(CITYSCAPES_COLORS));
        for (std::size_t i = 0; i < colors.size(); ++i)
            colors[i] = { CITYSCAPES_COLORS[i].blue(), CITYSCAPES_COLORS[i].green(), CITYSCAPES_COLORS[i].red() };

        std::mt19937 rng;
        std::uniform_int_distribution<int> distr(0, 255);

        int FLAGS_delay = 1;
        int delay = FLAGS_delay;

        cv::Size graphSize{ static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60 };
        Presenter presenter("", 10, graphSize);

        std::chrono::steady_clock::duration latencySum{ 0 };
        unsigned latencySamplesNum = 0;
        std::ostringstream latencyStream;

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        while (cap.read(inImg) && delay >= 0) 
        {
            if (CV_8UC3 != inImg.type())
                throw std::runtime_error("BGR (or RGB) image expected to come from input");

            inferRequest.SetBlob(inName, wrapMat2Blob(inImg));
            inferRequest.Infer();

            const float* const predictions = inferRequest.GetBlob(outName)->cbuffer().as<float*>();
            for (int rowId = 0; rowId < outHeight; ++rowId)
            {
                for (int colId = 0; colId < outWidth; ++colId) 
                {
                    std::size_t classId = 0;
                    if (outChannels == 0) 
                    {  
                        // assume the output is already ArgMax'ed
                        classId = static_cast<std::size_t>(predictions[rowId * outWidth + colId]);
                    }
                    else 
                    {
                        float maxProb = -1.0f;
                        for (int chId = 0; chId < outChannels; ++chId) 
                        {
                            float prob = predictions[chId * outHeight * outWidth + rowId * outWidth + colId];
                            if (prob > maxProb) {
                                classId = chId;
                                maxProb = prob;
                            }
                        }
                    }
                    while (classId >= colors.size())
                    {
                        cv::Vec3b color(distr(rng), distr(rng), distr(rng));
                        colors.push_back(color);
                    }
                    maskImg.at<cv::Vec3b>(rowId, colId) = colors[classId];
                }
            }
            cv::resize(maskImg, resImg, inImg.size());
            //resImg = inImg * blending + resImg * (1 - blending);
            //presenter.drawGraphs(resImg);
            imshow("BG", resImg);

            latencySum += std::chrono::steady_clock::now() - t0;
            ++latencySamplesNum;
            latencyStream.str("");
            latencyStream << std::fixed << std::setprecision(1)
                << (std::chrono::duration_cast<ms>(latencySum) / latencySamplesNum).count() << " ms";

            constexpr int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
            constexpr double FONT_SCALE = 2;
            constexpr int THICKNESS = 2;
            int baseLine;

            cv::getTextSize(latencyStream.str(), FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
            cv::putText(resImg, latencyStream.str(), cv::Size{ 0, resImg.rows - baseLine }, FONT_FACE, FONT_SCALE,
                cv::Scalar{ 255, 0, 0 }, THICKNESS);

            if (show) {
                cv::imshow(WIN_NAME, resImg);
                int key = cv::waitKey(delay);
                switch (key) {
                case 'q':
                case 'Q':
                case 27: // Esc
                    delay = -1;
                    break;
                case 'p':
                case 'P':
                case 32: // Space
                    delay = !delay * (FLAGS_delay + !FLAGS_delay);
                    break;
                default:
                    presenter.handleKey(key);
                }
            }
            t0 = std::chrono::steady_clock::now();
        }
        std::cout << "Mean pipeline latency: " << latencyStream.str() << '\n';
        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        std::cout << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cout << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "Execution successful" << std::endl;
    return 0;
}
#endif

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
