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


uint8_t buffer[1024 * 1024 * 8];

int main(int argc, char** argv)
{
    Log4CplusConfigure();
    LOG4CPLUS_INFO(logger, "2D Pose");

    google::ParseCommandLineFlags(&argc, &argv, false);
    std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;
	
	//打开指定的文件映射对象
    HANDLE hMapFile = OpenFileMapping(FILE_MAP_READ, FALSE, FLAGS_rmn.c_str());//PAGE_READONLY
    if (hMapFile == NULL)
    {
        std::cout << "共享内存块未创建：" << FLAGS_rmn << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }
	//映射到当前进程的地址空间
    LPVOID pBuffer = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
    if (pBuffer == NULL)
    {
        std::cout << "共享内存块 MapViewOfFile 失败" << std::endl;
        system("pause");
        return EXIT_FAILURE;
    }
    
    HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);

    cv::Mat frame;
    uint32_t lastFrameId = 0;
    MapFileHeadInfo info;
    static int offset = sizeof(MapFileHeadInfo);
    int t = 0;
    
    std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + offset, reinterpret_cast<uint8_t*>(&info));
    if (frame.empty())
        frame.create(info.height, info.width, info.type);    
    std::copy((uint8_t*)pBuffer + offset, (uint8_t*)pBuffer + offset + info.length, frame.data);

    estimator.reshape(frame);  // 如果发生网络重塑，请勿进行测量

    cv::Size graphSize{ static_cast<int>(info.width / 4), 60 };
    Presenter presenter(FLAGS_u, info.height - graphSize.height - 10, graphSize);
    std::vector<HumanPose> poses;

    while (true)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + offset, reinterpret_cast<uint8_t*>(&info));
        
        //自动读取数据，调整大小
        if (frame.empty() || frame.rows != info.height || frame.cols != info.width || frame.type() != info.type)
        {
            frame.release();
            frame.create(info.height, info.width, info.type);
        }
        
        //重复帧，不在读取
        if (lastFrameId != info.fid)
        {
            //frame.data = (uchar*)pBuffer + offset; //这种是可以的
            std::copy((uint8_t*)pBuffer + offset, (uint8_t*)pBuffer + offset + info.length, frame.data);

            estimator.frameToBlobCurr(frame);
            estimator.startCurr();

            if (estimator.readyCurr())
            {
                poses = estimator.postprocessCurr();
                /*
                for (HumanPose const& pose : poses)
                {
                    std::stringstream rawPose;
                    
                    rawPose << std::fixed << std::setprecision(0);
                    for (auto const& keypoint : pose.keypoints) 
                    {
                        rawPose << keypoint.x << "," << keypoint.y << " ";
                    }
                    rawPose << pose.score;
                    
                    std::cout << rawPose.str() << std::endl;
                }
                */
                presenter.drawGraphs(frame);
                renderHumanPose(poses, frame);
            }
        }
    
        auto t1 = std::chrono::high_resolution_clock::now();
        double read_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        std::cout << "\33[2K\r" << "Memory Frame:" << read_time << "ms";

        cv::waitKey(1);
        cv::imshow("2D Pose ", frame);

        lastFrameId = info.fid;
    }

    system("pause");
    std::cout << "Hello World!\n";
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
