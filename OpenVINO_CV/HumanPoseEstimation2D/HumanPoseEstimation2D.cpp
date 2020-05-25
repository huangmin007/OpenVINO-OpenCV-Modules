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
#include "input_source.hpp"
#include "output_source.hpp"
#include "static_functions.hpp"

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
    //---------------- 第零步：配置控制台事件处理 --------------
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        std::cerr << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

#pragma region cmdline参数设置解析
    //---------------- 第一步：设计输入参数 --------------
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
    args.add<std::string>("output", 'o', "输出源参数，格式：(shared|console|socket)[:value[:value[:...]]]", false, "shared:pose2d_output.bin");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：human-pose-estimation-0001:FP16:CPU 或 human-pose-estimation-0001:FP16:HETERO:CPU,GPU", false, "human-pose-estimation-0001:FP16:CPU");
    
    args.add<bool>("async", 0, "是否异步分析识别", false, true);
#ifdef _DEBUG
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
    args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif
    args.set_program_name("HumanPoseEstimation2D");
    
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
#pragma endregion

    // --------- 第二点一步：获取或定义参数 -----
    bool show = args.get<bool>("show");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    if (model["path"].empty()) throw std::invalid_argument("AI模型参数错误：" + args.get<std::string>("model"));

    bool isLastFrame = false;
    bool isAsyncMode = args.get<bool>("async");         //总是在SYNC模式下开始执行
    bool isModeChanged = true;                          //更改执行模式时设置为true（SYNC <-> ASYNC）
    
    std::stringstream x_data;
    std::vector<HumanPose> poses;                       //预测的姿态数据
    std::string title = "Human Pose Estimation [" + args.get<std::string>("model") + "]";

    int delay = WaitKeyDelay;
    double total_use_time = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();


    //----------------- 第三步：加载分析模型 ----------------
    HumanPoseEstimator estimator(model["path"], model["device"], false);

    //----------------- 第四步：创建 输入/输出 源 ----------------
    InputSource capture;
    capture.open(args.get<std::string>("input"));
    OutputSource output;
    output.open(args.get<std::string>("output"));

    //读取第一帧数据
    size_t fid = 0;
    cv::Mat curr_frame, next_frame;
    capture.read(curr_frame, fid);
    
    estimator.reshape(curr_frame);   // 如果发生网络重塑，请勿进行测量
    cv::Size graphSize{ static_cast<int>(curr_frame.cols / 4), 60 };
    Presenter presenter("", curr_frame.rows - graphSize.height - 10, graphSize);
    
    while (true)
    {
        if (!IsRunning) break;
        t0 = std::chrono::high_resolution_clock::now();

        //这是第一个异步点
        //在异步模式下，我们捕获帧以填充下一个推断请求
        //在常规模式下，我们捕获到当前推断请求的帧
        size_t next_fid = 0;
        if (!capture.read(next_frame, next_fid))
        {
            if (next_frame.empty())
                isLastFrame = true;
            else
                throw std::logic_error("无法从输入源获取数据帧");
        }

        //异步分析
        if (isAsyncMode) 
        {
            if (isModeChanged)  estimator.frameToBlobCurr(curr_frame);
            if (!isLastFrame)   estimator.frameToBlobNext(next_frame);
        }
        else if (!isModeChanged) {
            estimator.frameToBlobCurr(curr_frame);
        }

        // 主同步点
        // 在真正的异步模式下，我们开始下一个推断请求，同时等待当前完成
        // 在常规模式下，我们启动当前请求，并立即等待其完成
        if (isAsyncMode) 
        {
            if (isModeChanged) estimator.startCurr();
            if (!isLastFrame)  estimator.startNext();
        }
        else if (!isModeChanged) {
            estimator.startCurr();
        }

        if (estimator.readyCurr()) 
        {
            poses = estimator.postprocessCurr();

#pragma region OutputSource
            x_data.str("");
            x_data << "<data>\r\n";
            x_data << "\t<fid>" << fid << "</fid>\r\n";
            x_data << "\t<count>" << poses.size() << "</count>\r\n";
            x_data << "\t<name>HumanPoseEstimation2D</name>\r\n";
            //x_data << "\t<keypoints>ears,eyes,nose,neck,shoulders,elbows,wrists,hips,knees,ankles</keypoints>\r\n";
            x_data << "\t<keypoints>neck,nose,l_shoulder,l_elbow,l_wrist,l_hip,l_knee,l_ankle,r_shoulder,r_elbow,r_wrist,r_hip,r_knee,r_ankle,r_eye,l_eye,r_ear,l_ear</keypoints>\r\n";

            for (HumanPose const& pose : poses)
            {
                x_data << "\t<pose source=\"" << pose.score << "\">\r\n";
                for (auto const& keypoint : pose.keypoints)
                    x_data << "\t\t<position>" << keypoint.x << "," << keypoint.y << "</position>\r\n";

                x_data << "\t</pose>\r\n";
            }
            x_data << "</data>";

            output.set(SourceProperties::OUTPUT_FORMAT, (uint8_t)OutputFormat::XML);
            output.write((uint8_t*)x_data.str().c_str(), x_data.str().length(), fid);
#pragma endregion

            if (show)
            {                
                presenter.drawGraphs(curr_frame);
                renderHumanPose(poses, curr_frame);

                std::ostringstream txt;
                txt << "Frame ID:" << fid << "  Total Use Time: " << total_use_time << " ms";
                cv::putText(curr_frame, txt.str(), cv::Point2f(10, 25), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 0, 0));

                cv::imshow(title, curr_frame);
            }
        }

        if (!show || output.getType() != OutputType::CONSOLE)
            std::cout << "\33[2K\r[ INFO] " << "Frame ID:" << fid << "    Total Use Time:" << total_use_time << "ms";

        if (isLastFrame)  break;
        if (isModeChanged) isModeChanged = false;

        t1 = std::chrono::high_resolution_clock::now();
        total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
        delay = WaitKeyDelay - total_use_time;
        if (delay <= 0) delay = 1;

        fid = next_fid;
        // 在真正的异步模式下，我们将NEXT和CURRENT请求交换给下一个迭代
        curr_frame = next_frame;
        next_frame = cv::Mat();
        if (isAsyncMode) estimator.swapRequest();

        const int key = cv::waitKey(delay) & 255;
        if (key == 'p') {
            delay = (delay == 0) ? 33 : 0;
        }
        else if (27 == key) { // Esc
            break;
        }
        else if (9 == key) { // Tab
            isAsyncMode ^= true;
            isModeChanged = true;
        }
        if(show)    presenter.handleKey(key);
    }
    std::cout << std::endl;

    output.release(); 
    capture.release();
    
    if(show)    cv::destroyAllWindows();

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
