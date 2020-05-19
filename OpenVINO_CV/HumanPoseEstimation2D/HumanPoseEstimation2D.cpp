// OpenPoseModel.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <vector>
#include <chrono>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <Windows.h>
#include "OpenVINO_CV.h"
#include "LoggerConfig.h"
#include "human_pose_estimator.hpp"

#include <monitors/presenter.h>
#include "render_human_pose.hpp"
#include "args.hpp"

using namespace human_pose_estimation;


int main(int argc, char** argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cout << "Error: Unable to install handler!\n" << std::endl;
        return EXIT_FAILURE;
    }

    //解析参数
    google::ParseCommandLineFlags(&argc, &argv, false);
    
    //日志格式
    Log4CplusConfigure();
    LOG4CPLUS_INFO(logger, "2D Human Pose Estimation");

    //引擎信息
    std::stringstream msg; msg << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion();
    LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(msg.str()));

#if 会导致共享内存句柄获取失败
    //msg.str(""); msg << "可用目标设备:" << GetAvailableDevices();
    //LOG4CPLUS_INFO(logger, LOG4CPLUS_STRING_TO_TSTRING(msg.str()));
#endif

    LPVOID pBuffer = NULL, dBuffer = NULL;
    HANDLE hMapFile = NULL, dMapFile = NULL;
    if (!GetOnlyReadMapFile(hMapFile, pBuffer, FLAGS_rfn.c_str())) return EXIT_FAILURE;    
    if (!CreateOnlyWriteMapFile(dMapFile, dBuffer, FLAGS_wfs, FLAGS_wfn.c_str())) return EXIT_FAILURE;

    OutputData data;    //输出数据
    static int odSize = sizeof(OutputData);   //OutputData 结构数据大小
    
    cv::Mat frame;
    MapFileHeadInfo head;   //文件头信息
    uint32_t lastFrameId = 0;//上一帧id
    static int hdSize = sizeof(MapFileHeadInfo);    //MapFileHeadInfo 结构数据大小

    std::vector<HumanPose> poses;   //预测的姿态数据
    HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
    
    //读取第一帧数据
    std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + hdSize, reinterpret_cast<uint8_t*>(&head));
    if (frame.empty())  frame.create(head.height, head.width, head.type);
    std::copy((uint8_t*)pBuffer + hdSize, (uint8_t*)pBuffer + hdSize + head.length, frame.data);

    estimator.reshape(frame);   // 如果发生网络重塑，请勿进行测量
    cv::Size graphSize{ static_cast<int>(head.width / 4), 60 };
    Presenter presenter(FLAGS_um, head.height - graphSize.height - 10, graphSize);
    
    int waitDelay = 1;
    double use_time = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (!IsRunning) break;

        t0 = std::chrono::high_resolution_clock::now();
        std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + hdSize, reinterpret_cast<uint8_t*>(&head));
        
        //重复帧，不在读取
        if (lastFrameId != head.fid)
        {
            std::copy((uint8_t*)pBuffer + hdSize, (uint8_t*)pBuffer + hdSize + head.length, frame.data);

            estimator.frameToBlobCurr(frame);
            estimator.startCurr();

            if (estimator.readyCurr())
            {
                std::stringstream rawPose;
                poses = estimator.postprocessCurr();
                
                rawPose << "<data>\r\n";
                rawPose << "\t<fid>" << head.fid << "</fid>\r\n";
                rawPose << "\t<count>" << poses.size() << "</count>\r\n";
                rawPose << "\t<keypoints>ears,eyes,nose,neck,shoulders,elbows,wrists,hips,knees,ankles</keypoints>\r\n";

                for (HumanPose const& pose : poses)
                {
                    rawPose << "\t<pose source=\"" << pose.score << "\">\r\n";
                    for (auto const& keypoint : pose.keypoints)
                        rawPose << "\t\t<position>" << keypoint.x << "," << keypoint.y << "</position>\r\n";
                    
                    rawPose << "\t</pose>\r\n";
                }
                rawPose << "</data>";

                std::string pose = rawPose.str();
                if (FLAGS_output)
                    std::cout << "Source::" << pose << std::endl;

                data.mid = 0;
                data.fid = head.fid;
                data.length = pose.length();
                data.format = ResultDataFormat::XML;

                std::copy((uint8_t*)&data, (uint8_t*)&data + odSize, (uint8_t*)dBuffer);
                std::copy(pose.c_str(), pose.c_str() + pose.length(), (char*)dBuffer + odSize);

                if (FLAGS_show)
                {
                    presenter.drawGraphs(frame);
                    renderHumanPose(poses, frame);
                }
#if 0
                OutputData* t = reinterpret_cast<OutputData*>(dBuffer);
                std::cout << "Size:" << sizeof(OutputData) << " fid:" << t->fid << " len:" << t->length << std::endl;
                
                char *ss = new char[t->length + 1];
                std::copy((char*)dBuffer + odSize, (char*)dBuffer + odSize + t->length, ss);
                ss[t->length + 1] = '\0';
                std::cout << "Memory::" << ss << std::endl;
#endif
            }
        }
    
        t1 = std::chrono::high_resolution_clock::now();
        use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        if(!FLAGS_output) std::cout << "\33[2K\r" << "Total Inference Time:" << use_time << "ms";

        waitDelay = WaitKeyDelay - use_time;
        cv::waitKey(waitDelay <= 0 ? 1 : waitDelay);
        if (FLAGS_show) cv::imshow("2D Human Pose Estimation (" + FLAGS_d + ")", frame);
        
        lastFrameId = head.fid;
    }
    
    cv::destroyAllWindows();

    //撤消地址空间内的视图
    if (pBuffer != NULL) UnmapViewOfFile(pBuffer);
    if (dBuffer != NULL) UnmapViewOfFile(dBuffer);
    //关闭共享文件句柄
    if (hMapFile != NULL) CloseHandle(hMapFile);
    if (hMapFile != NULL) CloseHandle(dMapFile);

    LOG4CPLUS_INFO(logger, "Exiting ... ");
    log4cplus::Logger::shutdown();

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
