// OpenOpseDetection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include "pose_detection.hpp"
#include <static_functions.hpp>
#include <input_source.hpp>

using namespace space;

int main()
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

#pragma region 参数设置解析
    //wait ...
    bool show = true;
#pragma endregion

    InputSource inputSource;
    if (!inputSource.open("camera:0:640x480"))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    InferenceEngine::Core ie;
    PoseDetection detector;
    detector.ConfigNetwork(ie, "models\\A_hand_pose_iter_102000\\FP32\\pose_iter_160000.xml", "CPU", false);

    cv::Mat frame;
    inputSource.read(frame);
    detector.RequestInfer(frame);

    std::stringstream use_time;
    std::stringstream title;
    title << "[Source] Object Detection";

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
        infer_use_time = detector.GetInferUseTime();
        frame_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

        delay = WaitKeyDelay - frame_use_time;
        if (delay <= 0) delay = 1;

        cv::waitKey(delay);
    }

    inputSource.release();

    std::cout << "Hello World!\n";
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
