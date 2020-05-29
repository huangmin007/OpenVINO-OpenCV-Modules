// 对象检测
//

#include <iostream>
#include <fstream>
#include "cmdline.h"
#include "object_detection.hpp"
#include "static_functions.hpp"
#include "input_source.hpp"


using namespace space;

/// <summary>
/// 绘制检测对象的边界
/// </summary>
/// <param name="frame"></param>
/// <param name="results"></param>
/// <param name="labels"></param>
void DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string> &labels)
{
    size_t length = results.size();
    if (length <= 0) return;
    
    std::stringstream txt;
    cv::Scalar border_color(0, 255, 0);

    float scale = 0.2f;
    int shift = 0;
    int thickness = 2;
    int lineType = cv::LINE_AA;

    std::string label;
    size_t label_length = labels.size();

    for (const auto &result : results)
    {
        cv::Rect rect = ScaleRectangle(result.location);
        cv::rectangle(frame, rect, border_color, 1);

        cv::Point h_width = cv::Point(rect.width * scale, 0);   //水平宽度
        cv::Point v_height = cv::Point(0, rect.height * scale); //垂直高度

        cv::Point left_top(rect.x, rect.y);
        cv::line(frame, left_top, left_top + h_width, border_color, thickness, lineType, shift);     //-
        cv::line(frame, left_top, left_top + v_height, border_color, thickness, lineType, shift);    //|

        cv::Point left_bottom(rect.x, rect.y + rect.width - 1);
        cv::line(frame, left_bottom, left_bottom + h_width, border_color, thickness, lineType, shift);       //-
        cv::line(frame, left_bottom, left_bottom - v_height, border_color, thickness, lineType, shift);      //|

        cv::Point right_top(rect.x + rect.width - 1, rect.y);
        cv::line(frame, right_top, right_top - h_width, border_color, thickness, lineType, shift);   //-
        cv::line(frame, right_top, right_top + v_height, border_color, thickness, lineType, shift);  //|

        cv::Point right_bottom(rect.x + rect.width - 1, rect.y + rect.height - 1);
        cv::line(frame, right_bottom, right_bottom - h_width, border_color, thickness, lineType, shift);   //-
        cv::line(frame, right_bottom, right_bottom - v_height, border_color, thickness, lineType, shift);  //|

        txt.str("");
        label = (label_length <= 0 || result.label < 0 || result.label >= label_length) ? 
            std::to_string(result.label) : labels[result.label];
        txt << std::setprecision(3) << "Label:" << label << "  Conf:" << result.confidence;
        cv::putText(frame, txt.str(), cv::Point(left_top.x, left_top.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }
}

int main(int argc, char** argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)space::CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

    //未完成 labels 加载
    std::cout << "通用对象检测模块 说明：" << std::endl;
    std::cout << "1.对象检测一般输出的数据为标定的对象边界线，也就是 Box " << std::endl;
    std::cout << "2.支持一个网络层的输入，输入参数为：[BxCxHxW] = [1x3xHxW]" << std::endl;
    std::cout << "3.支持一个或多个网络层的输出，输出为共享内存数据，与源网络层输出数据一致，共享名称为网络输出层名称" << std::endl;
    std::cout << "4.在 show 模式下，做为示例暂时只处理了：a).单层输出[1,1,N,7]  b).两层输出[N,5][N]" << std::endl;
    std::cout << "\r\n" << std::endl;


#pragma region cmdline参数设置解析
//---------------- 第一步：设计输入参数 --------------confidence
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "camera:0:1280x720");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：face-detection-adas-0001:FP32:CPU 或 face-detection-adas-0001:FP16:GPU", false, "face-detection-0105:FP32:CPU");
    //face-detection-adas-0001,person-detection-retail-0013,face-detection-0105
    args.add<std::string>("output_layer_names", 'o', "(output layer name)多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，区分大小写，格式：(layerName):(layername):(...)", false, "boxes/Split.0:labels");
    args.add<float>("conf", 'c', "检测结果的置信度阈值(confidence threshold)", false, 0.5);

    args.add<bool>("async", 0, "是否异步分析识别", false, true);
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
        InferenceEngineInfomation();
        return EXIT_SUCCESS;
    }
    if (!isParser)
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    bool show = args.get<bool>("show"); 
    bool async = args.get<bool>("async");
    float conf = args.get<float>("conf");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::vector<std::string> output_layer_names = SplitString(args.get<std::string>("output_layer_names"), ':');

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

    cv::Mat frame, next_frame;    
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
        ObjectDetection detector(output_layer_names, show);
        detector.config(ie, args.get<std::string>("model"), async, conf);

        inputSource.read(frame);
        detector.request(frame);
        std::vector<ObjectDetection::Result> results;
        std::stringstream title;
        title << "Object Detection [" << model["full"] << "]";

        bool isResult = false;
        std::stringstream txt;
        std::stringstream use_time;

        int delay = WaitKeyDelay;
        double total_use_time = 0.0f;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = std::chrono::high_resolution_clock::now();

        while (true)
        {
            t0 = std::chrono::high_resolution_clock::now();

            use_time.str(""); 
            isResult = detector.getResults(results);

            if (show)
            {
                use_time << "Detection Count:" << results.size() << "  Total Use Time:" << total_use_time << "ms";

                if (!next_frame.empty())
                    frame = next_frame;

                DrawObjectBound(frame, results, labels);
               
                cv::putText(frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
                cv::imshow(title.str(), frame);
            }
            else
            {
                use_time << std::setprecision(3) << std::boolalpha << "Result:" << isResult << "    Total Use Time:" << total_use_time << "ms";
            }
            std::cout << "\33[2K\r[ INFO] " << use_time.str();

            if (!inputSource.read(next_frame))
            {
                LOG("WARN") << "读取数据帧失败 ... " << std::endl;
                Sleep(15);
                continue;
            }
            detector.request(next_frame);

            t1 = std::chrono::high_resolution_clock::now();
            total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
            delay = WaitKeyDelay - total_use_time;
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
