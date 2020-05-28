// 对象检测
//

#include <iostream>

#include "object_detection.hpp"
#include "static_functions.hpp"
#include "input_source.hpp"

using namespace space;

#if false
int main(int argc, char **argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)space::CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

#pragma region 参数设置解析
    //wait ...
#pragma endregion

    cv::Mat prev_frame, frame;
    space::InputSource inputSource;
    //if (!inputSource.open("video:video.mp4"))
    if (!inputSource.open("camera:0:640x480"))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }


    InferenceEngine::Core ie;
    PersonDetection detector("models\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.xml","CPU", true);
    detector.read(ie);

    inputSource.read(frame);

    std::vector<PersonDetection::Result> results;
    detector.enqueue(frame);
    detector.request();

    prev_frame = frame.clone();
    bool frameReadStatus = inputSource.read(frame);

    int delay = WaitKeyDelay;
    double total_use_time = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    while (true)
    {
        //if (!IsRunning) break;
        t0 = std::chrono::high_resolution_clock::now();

        bool isLastFrame = !frameReadStatus;

        detector.wait();
        if (detector.getResults(results))
        {
            cv::imshow("Person Detection", frame);
        }

        //if (isLastFrame)
        //{
        //    detector.enqueue(frame);
         //   detector.request();
        //}

        std::cout << "\33[2K\r[ INFO] Count:" << results.size() << " Total Use Time:" << total_use_time << "ms";

        if (!inputSource.read(frame))
        {
            LOG("WARN") << "读取帧失败 ... " << std::endl;
            //std::this_thread::sleep_for(100);
            Sleep(WaitKeyDelay);
            continue;
        }

        for (int i = 0; i < results.size(); i++)
        {
            std::stringstream txt;
            txt << "Label:" << results[i].label << " Conf:" << results[i].confidence;

            cv::rectangle(frame, results[i].location, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, txt.str(), cv::Point(results[i].location.x, results[i].location.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }

        detector.enqueue(frame);
        detector.request();

        t1 = std::chrono::high_resolution_clock::now();
        total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        delay = WaitKeyDelay - total_use_time;
        if (delay <= 0) delay = 1;

        cv::waitKey(delay);
    }

    inputSource.release();

    std::cout << "Hello World!\n";
    Sleep(500);
    return EXIT_SUCCESS;
}
#else


int main(int argc, char** argv)
{
    cv::Mat frame;
    //frame = cv::imread("E:\\2018\\2019_OpenCV\\C++\\OpenCV410_Projects\\Resources\\DSB.png");
    
    InputSource inputSource;
    if (!inputSource.open("camera:0:1920x1080"))
    {
        inputSource.release();
        LOG("ERROR") << "打开输入源失败 ... " << std::endl;
        return EXIT_FAILURE;
    }

    //cv::imshow("Source", frame);
    //cv::waitKey(30);
    LOG("INFO") << "Hello Wrold" << std::endl;

    InferenceEngine::Core ie;
    ObjectDetection detector(true);    
    detector.config(ie, "person-detection-retail-0013:FP32:CPU", true, 0.5);

    int delay = WaitKeyDelay;
    double total_use_time = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();

    while (true)
    {
        t0 = std::chrono::high_resolution_clock::now();

        if (!inputSource.read(frame))
        {
            LOG("WARN") << "读取数据帧失败 ... " << std::endl;
            Sleep(30);
        }

        detector.request(frame);
        std::vector<ObjectDetection::Result> results;
        //bool hasResults = detector.hasResults();
        //LOG("INFO") << "HasResults:" << std::boolalpha << hasResults << std::endl;
        if (detector.getResults(results))
        {
            //LOG("INFO") << "Result Size:" << results.size() << std::endl;

            for (int i = 0; i < results.size(); i++)
            {
                std::stringstream txt;
                txt << "Label:" << results[i].label << " Conf:" << results[i].confidence;

                cv::rectangle(frame, results[i].location, cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, txt.str(), cv::Point(results[i].location.x, results[i].location.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }

            std::stringstream use_time;
            use_time << "Total Use Time:" << total_use_time << "ms";
            cv::putText(frame, use_time.str(), cv::Point(10,25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 1);
        }
        cv::imshow("Source", frame);

        t1 = std::chrono::high_resolution_clock::now();
        total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        delay = WaitKeyDelay - total_use_time;
        if (delay <= 0) delay = 1;

        cv::waitKey(delay);
    }
    
    system("pause");
    return EXIT_SUCCESS;
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
