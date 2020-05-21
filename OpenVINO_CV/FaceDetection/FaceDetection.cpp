// FaceDetection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <functional>
#include <fstream>
#include <random>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <set>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "monitors/presenter.h"
#include "cmdline.h"
#include "OpenVINO_CV.h"
#include "detectors.hpp"
#include "visualizer.hpp"
#include "face.hpp"

int main()
{
    std::cout << "Hello World!\n";
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }
    // ---------------------------- 输入参数的解析和验证 ----------------------------------------

    // --------------------------- 0. VideoCapture ------------------------------------------
    cv::VideoCapture capture;
    capture.open(0);

    cv::Mat frame, prev_frame, next_frame;
    if (!capture.read(frame))
    {
        LOG("ERROR") << "读取视频帧失败 ..." << std::endl;
        return EXIT_FAILURE;
    }
    const int width = frame.cols;
    const int height = frame.rows;

    // -------- input args
    std::string deviceName = "CPU";
    bool pc = false;    //启用每层性能报告
    bool async = false;
    bool raw_output_message = false;
    double dx_coef = 1.0;
    double dy_coef = 1.0;
    double bb_enlarge_coef_output_message = 1.2;

    // --------------------------- 1. 加载推理引擎实例 ------------------------------------------
    InferenceEngine::Core ie;
    FaceDetection faceDetector("models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml",
        deviceName, 1, false, async, 0.5, raw_output_message, bb_enlarge_coef_output_message,
        dx_coef, dy_coef);

    LOG("INFO") << "Loading device [" << deviceName << "]" <<std::endl;
    //LOG("INFO") << ie.GetVersions(deviceName) << std::endl;
    //std::cout << ie.GetVersions(deviceName) << std::endl;
    //ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, ""} }, "GPU");
    if (pc)
        ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES} });
    
    // ------------ 2. 读取由 Model Optimizer 生成的 IR(Intermediate Representation)（.xml和.bin文件）
    faceDetector.net = ie.LoadNetwork(faceDetector.read(ie), deviceName, {});


    // --------------------------- 3. 配置输入和输出 --------------------------------------------
    // --------------------------- 3.1 准备输入Blob --------------------------------------------
    // --------------------------- 3.2 准备输出Blob --------------------------------------------
    // --------------------------- 4. 将模型加载到设备 ------------------------------------------
    // --------------------------- 5. 创建推断请求 ----------------------------------------------
    // --------------------------- 6. 准备输入 -------------------------------------------------
    // --------------------------- 7. 推断 -----------------------------------------------------
    LOG("INFO") << "开始推断 ...... " << std::endl;
    std::list<Face::Ptr> faces;
    bool isFaceAnalyticsEnabled = false;

    faceDetector.enqueue(frame);
    faceDetector.submitRequest();

    uint32_t id = 0;
    uint32_t framesCounter = 0;
    prev_frame = frame.clone();
    //读取下一帧
    bool frameReadStatus = capture.read(frame);

    const cv::Point THROUGHPUT_METRIC_POSITION{ 10, 45 };
    cv::Size graphSize{ static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60 };
    Presenter presenter("", THROUGHPUT_METRIC_POSITION.y + 15, graphSize);
    
    Visualizer::Ptr visualizer;
    visualizer = std::make_shared<Visualizer>(cv::Size(width, height));

    while (true)
    {
        if (!IsRunning) break;

        framesCounter++;
        bool isLastFrame = !frameReadStatus;

        // 检索前一帧的面部检测结果
        faceDetector.wait();
        faceDetector.fetchResults();
        auto prev_detection_results = faceDetector.results;

        //如果前一帧是最后一帧，则无法推断出有效帧
        if (!isLastFrame)
        {
            faceDetector.enqueue(frame);
            faceDetector.submitRequest();
        }

        // 填充面部分析网络的输入
        for (auto&& face : prev_detection_results)
        {
            if (isFaceAnalyticsEnabled)
            {
                //ageGenderDetector.enqueue(face);
            }
        }
        //运行 年龄/性别识别，头姿势估计，情绪识别和面部地标估计网络
        if (isFaceAnalyticsEnabled)
        {
            //ageGenderDetector.submitRequest();
        }

        //如果当前帧不是最后一帧，则读取下一帧
        if (!isLastFrame)
        {
            frameReadStatus = capture.read(next_frame);
        }

        if (isFaceAnalyticsEnabled) 
        {
            //ageGenderDetector.wait();
            //headPoseDetector.wait();
            //emotionsDetector.wait();
            //facialLandmarksDetector.wait();
        }

        //  后期处理
        std::list<Face::Ptr> prev_faces;
        faces.clear();

        //处理检测到的每张脸
        for (size_t i = 0; i < prev_detection_results.size(); i++) 
        {
            auto& result = prev_detection_results[i];
            cv::Rect rect = result.location & cv::Rect(0, 0, width, height);

            Face::Ptr face;
            face = std::make_shared<Face>(id++, rect);

            faces.push_back(face);
        }

        presenter.drawGraphs(prev_frame);
        // drawing faces
        visualizer->draw(prev_frame, faces);
        cv::imshow("Detection Results", prev_frame);

        prev_frame = frame;
        frame = next_frame;
        next_frame = cv::Mat();

        if (isLastFrame)
        {

        }
        else
        {
            int key = cv::waitKey(10);
            if (27 == key || 'Q' == key || 'q' == key) {
                break;
            }
            presenter.handleKey(key);
        }
    }

    // --------------------------- 8. 处理输出 --------------------------------------------------

    capture.release();
    cv::destroyAllWindows();
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
