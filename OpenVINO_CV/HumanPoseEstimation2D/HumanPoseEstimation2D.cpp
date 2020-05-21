// HumanPoseEstimation2D.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//rem% INTEL_OPENVINO_DIR% \deployment_tools\inference_engine\bin\intel64\Release; % INTEL_OPENVINO_DIR% \deployment_tools\inference_engine\bin\intel64\Debug; % INTEL_OPENVINO_DIR% \deployment_tools\ngraph\lib; % INTEL_OPENVINO_DIR% \deployment_tools\inference_engine\external\tbb\bin; % HDDL_INSTALL_DIR% \bin; % INTEL_OPENVINO_DIR% \opencv\bin; % INTEL_OPENVINO_DIR% \openvx\bin; % INTEL_DEV_REDIST% redist\intel64_win\compiler; D:\Program Files(x86)\NetSarang\Xshell 6\; C:\ProgramData\Oracle\Java\javapath; D:\Program Files\TortoiseSVN\bin; C:\Program Files\Microsoft SQL Server\120\Tools\Binn\; C:\Program Files\Microsoft SQL Server\130\Tools\Binn\; C:\Program Files\Git\cmd; C:\Program Files\PuTTY\; C:\WINDOWS\System32\OpenSSH\; C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR; D:\Program Files\CMake\bin; C:\Program Files\dotnet\; C:\WINDOWS\system32; C:\WINDOWS; C:\WINDOWS\System32\Wbem; C:\WINDOWS\System32\WindowsPowerShell\v1.0\; C:\WINDOWS\System32\OpenSSH\; C:\Program Files(x86)\NVIDIA Corporation\PhysX\Common; D:\Program Files\nodejs\; C:\Program Files\Intel\WiFi\bin\; C:\Program Files\Common Files\Intel\WirelessCommon\; C:\Program Files(x86)\Windows Kits\10\Windows Performance Toolkit\; D:\Program Files(x86)\GnuWin32\bin; C:\Users\Administrator\AppData\Local\Programs\Python\Python37; C:\Program Files\Microsoft SQL Server\Client SDK\ODBC\170\Tools\Binn\; E:\2018\2018_MultimediaTerminalServices\ConsoleShellApplication\bin\Console.Shell;
//

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <Windows.h>
#include <monitors/presenter.h>

#include "render_human_pose.hpp"
#include "human_pose_estimator.hpp"

#include "cmdline.h"
#include "OpenVINO_CV.h"

//Mat BGR

using namespace human_pose_estimation;

//获取引擎的可计算设备列表
static std::string GetAvailableDevices()
{
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();

    std::stringstream info;
    info << "可用目标设备:";
    for (const auto& device : devices)
        info << "  " << device;

    return info.str();
}

int main(int argc, char** argv)
{
    //解析参数
    cmdline::parser args;

    args.add<std::string>("help", 'h', "参数说明", false, "");
    args.add<std::string>("model", 'm', "人体姿势估计模型（.xml）文件", false, "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml");
    args.add<std::string>("device", 'd', "指定用于 人体姿势评估 的目标设备", false, "CPU");
    
    args.add<std::string>("rfn", 0, "用于读取AI分析的数据内存共享名称", false, "source.bin");
    args.add<std::string>("wfn", 0, "用于存储AI分析结果数据的内存共享名称", false, "pose2d.bin");
    args.add<uint32_t>("wfs", 0, "用于存储AI分析结果数据的内存大小，默认：1024*1024(Bytes)", false, 1024 * 1024);

    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
    args.add<bool>("output", 0, "在控制台上输出推断结果信息", false, false);

    args.add<bool>("pc", 0, "启用每层性能报告", false, false);
    args.add<std::string>("um", 0, "最初显示的监视器列表", false, "");
    args.set_program_name("HumanPoseEstimation2D");

    bool isParser = args.parse(argc, argv);
    if (args.exist("help"))
    {
        std::cerr << args.usage();
        return EXIT_SUCCESS;
    }
    if (!isParser)
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cerr << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

    bool show = args.get<bool>("show");
    bool output = args.get<bool>("output");
    std::string model = args.get<std::string>("model");
    std::string device = args.get<std::string>("device");

    //引擎信息
    std::cout << "[ INFO] InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;
#if 会导致共享内存句柄获取失败？？
    std::cout << "[ INFO] 可用目标设备:" << GetAvailableDevices() << std::endl;
#endif
    std::cout << "[ INFO] AI模型文件：" << model << std::endl;

    std::cout << "[ INFO] 获取/创建内存共享数据文件 ......" << std::endl;
    LPVOID sBuffer = NULL, dBuffer = NULL;
    HANDLE sMapFile = NULL, dMapFile = NULL;
    if (!GetOnlyReadMapFile(sMapFile, sBuffer, args.get<std::string>("rfn").c_str())) return EXIT_FAILURE;    
    if (!CreateOnlyWriteMapFile(dMapFile, dBuffer, args.get<uint32_t>("wfs"), args.get<std::string>("wfn").c_str())) return EXIT_FAILURE;

    InferenceData data;    //输出数据
    static int odSize = sizeof(InferenceData);   //OutputData 结构数据大小
    
    cv::Mat frame;
    VideoSourceData head;       //文件头信息
    uint32_t lastFrameId = 0;   //上一帧id
    static int hdSize = sizeof(VideoSourceData);    //MapFileHeadInfo 结构数据大小

    std::cout << "[ INFO] 模型加载分析中 ......" << std::endl;
    std::vector<HumanPose> poses;   //预测的姿态数据
    HumanPoseEstimator estimator(model, device, args.get<bool>("pc"));
    
    //读取第一帧数据
    std::copy((uint8_t*)sBuffer, (uint8_t*)sBuffer + hdSize, reinterpret_cast<uint8_t*>(&head));
    if (frame.empty())  frame.create(head.height, head.width, head.type);
    std::copy((uint8_t*)sBuffer + hdSize, (uint8_t*)sBuffer + hdSize + head.length, frame.data);

    estimator.reshape(frame);   // 如果发生网络重塑，请勿进行测量
    cv::Size graphSize{ static_cast<int>(head.width / 4), 60 };
    Presenter presenter(args.get<std::string>("um"), head.height - graphSize.height - 10, graphSize);
    
    int waitDelay = 1;
    double use_time = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (!IsRunning) break;

        t0 = std::chrono::high_resolution_clock::now();
        std::copy((uint8_t*)sBuffer, (uint8_t*)sBuffer + hdSize, reinterpret_cast<uint8_t*>(&head));
        
        //重复帧，不在读取
        if (lastFrameId != head.fid)
        {
            std::copy((uint8_t*)sBuffer + hdSize, (uint8_t*)sBuffer + hdSize + head.length, frame.data);

            estimator.frameToBlobCurr(frame);
            estimator.startCurr();

            if (estimator.readyCurr())
            {
                std::stringstream rawPose;
                poses = estimator.postprocessCurr();
                
                rawPose << "<data>\r\n";
                rawPose << "\t<fid>" << head.fid << "</fid>\r\n";
                rawPose << "\t<count>" << poses.size() << "</count>\r\n";
                rawPose << "\t<name>HumanPoseEstimation2D</name>\r\n";
                rawPose << "\t<keypoints>ears,eyes,nose,neck,shoulders,elbows,wrists,hips,knees,ankles</keypoints>\r\n";
                //rawPose << "\t<keypoints>neck,nose,l_shoulder,l_elbow,l_wrist,l_hip,l_knee,l_ankle,r_shoulder,r_elbow,r_wrist,r_hip,r_knee,r_ankle,r_eye,l_eye,r_ear,l_ear</keypoints>\r\n";

                for (HumanPose const& pose : poses)
                {
                    rawPose << "\t<pose source=\"" << pose.score << "\">\r\n";
                    for (auto const& keypoint : pose.keypoints)
                        rawPose << "\t\t<position>" << keypoint.x << "," << keypoint.y << "</position>\r\n";
                    
                    rawPose << "\t</pose>\r\n";
                }
                rawPose << "</data>";

                std::string pose = rawPose.str();
                if (output)
                    std::cout << "::Source::" << std::endl << pose << std::endl;

                data.mid = 0;
                data.fid = head.fid;
                data.length = pose.length();
                data.format = InferenceDataFormat::XML;
                //写入共享内存
                std::copy((uint8_t*)&data, (uint8_t*)&data + odSize, (uint8_t*)dBuffer);
                std::copy(pose.c_str(), pose.c_str() + pose.length(), (char*)dBuffer + odSize);

                if (show)
                {
                    presenter.drawGraphs(frame);
                    renderHumanPose(poses, frame);
                }
#if TestOutput
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
        if(!output) std::cout << "\33[2K\r[ INFO] " << "Current FrameID:" << head.fid << "  Inference Use:" << use_time << "ms";

        waitDelay = WaitKeyDelay - use_time;
        cv::waitKey(waitDelay <= 0 ? 1 : waitDelay);
        if (show) cv::imshow("Human Pose Estimation 2D (" + device + ")", frame);
        
        lastFrameId = head.fid;
    }

    std::cout << std::endl;
    cv::destroyAllWindows();

    //撤消地址空间内的视图
    if (sBuffer != NULL) UnmapViewOfFile(sBuffer);
    if (dBuffer != NULL) UnmapViewOfFile(dBuffer);
    //关闭共享文件句柄
    if (sMapFile != NULL) CloseHandle(sMapFile);
    if (dMapFile != NULL) CloseHandle(dMapFile);

    std::cout << "[ INFO] Exiting ..." << std::endl;
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
