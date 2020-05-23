// FaceDetection.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束
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
#include <ie_extension.h>
#include "samples/common.hpp"

#include "monitors/presenter.h"
#include "detectors.hpp"
#include "visualizer.hpp"
#include "face.hpp"

#include "cmdline.h"
#include "input_source.hpp"
#include "output_source.hpp"
#include "static_functions.hpp"


int main(int argc, char **argv)
{
    LOG_INFO << "分析参数" << std::endl;
#pragma region 参数设置
    cmdline::parser args;
    CreateGeneralCmdLine(args, "InteractiveFaceDetection", "face-detection-adas-0001:FP16:CPU");
    
    args.add<uint8_t>("count", 'c', "最多可同时处理的面部数量", false, 16);

    args.add<std::string>("m_em", 0, "用于<情绪识别检测>的网络模型文件(.xml)和目标设备，格式：(AI模型名称[:精度[:硬件]])", false, ""); //emotions-recognition-retail-0003:FP16:CPU
    args.add<std::string>("m_ag", 0, "用于<年龄/姓别检测>的网络模型文件(.xml)和目标设备，格式：(AI模型名称[:精度[:硬件]])", false, ""); //age-gender-recognition-retail-0013:FP16:CPU;
    args.add<std::string>("m_hp", 0, "用于<头部姿态检测>的网络模型文件(.xml)和目标设备，格式：(AI模型名称[:精度[:硬件]])", false, "head-pose-estimation-adas-0001:FP16:CPU");
    args.add<std::string>("m_lm", 0, "用于<人脸标记检测>的网络模型文件(.xml)和目标设备，格式：(AI模型名称[:精度[:硬件]])", false, "facial-landmarks-35-adas-0002:FP16:CPU");

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

#pragma endregion

    std::cout << "Hello World!\n";
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "[ERROR] Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }
    // ---------------------------- 输入参数的解析和验证 --------------------------------------
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    if (model.size() != 4) throw std::invalid_argument("AI模型参数错误：" + args.get<std::string>("model"));

    std::map<std::string, std::string>  m_em = ParseArgsForModel(args.get<std::string>("m_em"));
    std::map<std::string, std::string> m_ag = ParseArgsForModel(args.get<std::string>("m_ag"));
    std::map<std::string, std::string> m_hp = ParseArgsForModel(args.get<std::string>("m_hp"));
    std::map<std::string, std::string>  m_lm = ParseArgsForModel(args.get<std::string>("m_lm"));

    int count = args.get<uint8_t>("count");

    bool pc = args.get<bool>("pc");         //是否启用每层性能报告
    bool show = args.get<bool>("show");     //是否显示渲染图像
    bool async = args.get<bool>("async");   //是否异步检测

    // --------------------------- 0. VideoCapture ------------------------------------------
    InputSource capture;    //cv::VideoCapture capture;    
    capture.open(args.get<std::string>("input"));
    OutputSource output;
    output.open(args.get<std::string>("output"));

    size_t fid = 0;
    cv::Mat frame, prev_frame, next_frame;
    if (!capture.read(frame, fid))
    {
        LOG("ERROR") << "读取视频帧失败 ..." << std::endl;
        return EXIT_FAILURE;
    }
    const int width = frame.cols;
    const int height = frame.rows;

    // -------- input args
    double fps = -1.f;
    double dx_coef = 1.0;   //使边界框沿 x 轴围绕检测到的面部移动的系数
    double dy_coef = 1.0;   //使边界框沿 y 轴围绕检测到的面部移动的系数
    double bb_enlarge_coef_output_message = 1.2;    //放大/缩小检测到的面部周围的边框尺寸的系数
    bool smooth = true;         //不平滑人物属性
    bool raw_output_message = false;    //将推断结果输出为原始值
    bool dyn_batch_message = false;  //为AI网络启用动态批量大小
    bool show_emotion_bar = true;       //不显示情绪
    std::string cldnn = "";         //GPU自定义内核所需。具有内核描述的.xml文件的绝对路径。
    std::string cpu_library = "";   //CPU自定义层所需。使用内核实现的共享库的绝对路径。

    
    // --------------------------- 1. 加载推理引擎实例 ------------------------------------------
    InferenceEngine::Core ie;
    FaceDetection faceDetector(model["path"], model["device"], 1, false, async, 0.5, 
        raw_output_message, bb_enlarge_coef_output_message, dx_coef, dy_coef);
    AgeGenderDetection ageGenderDetector(m_ag["path"], m_ag["device"], count, dyn_batch_message, async, raw_output_message);
    HeadPoseDetection headPoseDetector(m_hp["path"], m_hp["device"], count, dyn_batch_message, async, raw_output_message);
    EmotionsDetection emotionsDetector(m_em["path"], m_em["device"], count, dyn_batch_message, async, raw_output_message);
    FacialLandmarksDetection facialLandmarksDetector(m_lm["path"], m_lm["device"], count, dyn_batch_message, async, raw_output_message);

    LOG("INFO") << "Loading device [" << model["device"] << "]" <<std::endl;

#pragma region 自定义配置CPU&GPU
    std::set<std::string> loadedDevices;
    std::pair<std::string, std::string> cmdOptions[] = {
        {model["device"], model["path"]},
        {m_ag["device"], m_ag["path"]},
        {m_hp["device"], m_hp["path"]},
        {m_em["device"], m_em["path"]},
        {m_lm["device"], m_lm["path"]}
    };
    for (auto&& option : cmdOptions) 
    {
        auto deviceName = option.first;
        auto networkName = option.second;
        if (deviceName.empty() || networkName.empty())  continue;

        if (loadedDevices.find(deviceName) != loadedDevices.end())  continue;
        
        LOG("INFO") << "Loading device " << deviceName << std::endl;
        //std::cout << ie.GetVersions(deviceName) << std::endl;

        /** 加载CPU设备的扩展 **/
        if ((deviceName.find("CPU") != std::string::npos)) {

            if (!cpu_library.empty()) {
                // CPU（MKLDNN）扩展作为共享库加载，并作为指向基本扩展的指针传递
                auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(cpu_library);
                ie.AddExtension(extension_ptr, "CPU");
                LOG("INFO") << "CPU Extension loaded: " << cpu_library << std::endl;
            }
        }
        else if (!cldnn.empty()) {
            // 加载GPU扩展
            ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cldnn} }, "GPU");
        }

        loadedDevices.insert(deviceName);
    }
#pragma endregion

    if (pc)/** 每层指标 **/
        ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES} });
    
    // ------------ 2. 读取由 Model Optimizer 生成的 IR(Intermediate Representation)（.xml和.bin文件）
    faceDetector.net = ie.LoadNetwork(faceDetector.read(ie), model["device"], {});
    Load(ageGenderDetector).into(ie, m_ag["device"], dyn_batch_message);
    Load(headPoseDetector).into(ie, m_hp["device"], dyn_batch_message);
    Load(emotionsDetector).into(ie, m_em["device"], dyn_batch_message);
    Load(facialLandmarksDetector).into(ie, m_lm["device"], dyn_batch_message);

    // --------------------------- 3. 配置输入和输出 --------------------------------------------
    // --------------------------- 3.1 准备输入Blob --------------------------------------------
    // --------------------------- 3.2 准备输出Blob --------------------------------------------
    // --------------------------- 4. 将模型加载到设备 ------------------------------------------
    // --------------------------- 5. 创建推断请求 ----------------------------------------------
    // --------------------------- 6. 准备输入 -------------------------------------------------
    // --------------------------- 7. 推断 -----------------------------------------------------
    LOG("INFO") << "开始推断 ...... " << std::endl;

    bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() || emotionsDetector.enabled() || facialLandmarksDetector.enabled();

    std::ostringstream out;
    std::list<Face::Ptr> faces;

    int delay = 1;
    size_t id = 0;
    double msrate = -1;
    size_t framesCounter = 0;

    if (fps > 0)  msrate = 1000.f / fps ;

    Visualizer::Ptr visualizer;
    if (show)
    {
        visualizer = std::make_shared<Visualizer>(cv::Size(width, height));
        if (show_emotion_bar && emotionsDetector.enabled()) {
            visualizer->enableEmotionBar(emotionsDetector.emotionsVec);
        }
    }

    // 在第一帧中检测所有脸部，并读取下一帧
    faceDetector.enqueue(frame);
    faceDetector.submitRequest();

    //读取下一帧
    prev_frame = frame.clone();    
    bool frameReadStatus = capture.read(frame, fid);

    const cv::Point THROUGHPUT_METRIC_POSITION{ 10, 45 };
    cv::Size graphSize{ static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60 };
    Presenter presenter("", THROUGHPUT_METRIC_POSITION.y + 15, graphSize);
    
    Timer timer;
    std::stringstream r_data;
    while (true)
    {
        if (!IsRunning) break;

        timer.start("total");
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
                auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                cv::Mat face = prev_frame(clippedRect);
                ageGenderDetector.enqueue(face);
                headPoseDetector.enqueue(face);
                emotionsDetector.enqueue(face);
                facialLandmarksDetector.enqueue(face);
            }
        }
        //运行 年龄/性别识别，头姿势估计，情绪识别和面部地标估计网络
        if (isFaceAnalyticsEnabled)
        {
            ageGenderDetector.submitRequest();
            headPoseDetector.submitRequest();
            emotionsDetector.submitRequest();
            facialLandmarksDetector.submitRequest();
        }

        //如果当前帧不是最后一帧，则读取下一帧
        if (!isLastFrame)
        {
            frameReadStatus = capture.read(next_frame, fid);

            if (!frameReadStatus)
            {
                //capture.open(args.get<std::string>("input"));
                LOG("WARN") << "无法打开输入源文件或相机: " << args.get<std::string>("input") << std::endl;
                //frameReadStatus = capture.read(next_frame);
            }
        }

        if (isFaceAnalyticsEnabled) 
        {
            ageGenderDetector.wait();
            headPoseDetector.wait();
            emotionsDetector.wait();
            facialLandmarksDetector.wait();
        }

        //  后期处理
        std::list<Face::Ptr> prev_faces;
        faces.clear();

        r_data.str("");
        r_data << "<data>" << std::endl;
        r_data << "\t<fid>" << fid << "</fid>" << std::endl;
        r_data << "\t<count>" << prev_detection_results.size() << "</count>" << std::endl;
        r_data << "\t<faces>" << std::endl;
        //处理检测到的每张脸
        for (size_t i = 0; i < prev_detection_results.size(); i++) 
        {
            auto& result = prev_detection_results[i];
            cv::Rect rect = result.location & cv::Rect(0, 0, width, height);

            Face::Ptr face;
            if (smooth) {
                face = matchFace(rect, prev_faces);
                float intensity_mean = calcMean(prev_frame(rect));

                if ((face == nullptr) ||
                    ((std::abs(intensity_mean - face->_intensity_mean) / face->_intensity_mean) > 0.07f)) {
                    face = std::make_shared<Face>(id++, rect);
                }
                else {
                    prev_faces.remove(face);
                }

                face->_intensity_mean = intensity_mean;
                face->_location = rect;
            }
            else {
                face = std::make_shared<Face>(id++, rect);
            }

            r_data << "\t<face label=\"" << result.label << "\" confidence=\"" << result.confidence << "\">" << std::endl;
            r_data << "\t\t<location>" << result.location.x << "," << result.location.y << "," << result.location.width << "," << result.location.height << "</location>" << std::endl;
            
            face->ageGenderEnable((ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch));
            if (face->isAgeGenderEnabled())
            {
                AgeGenderDetection::Result ageGenderResult = ageGenderDetector[i];
                face->updateGender(ageGenderResult.maleProb);
                face->updateAge(ageGenderResult.age);

                r_data << "\t\t<age>" << ageGenderResult.age << "</age>" << std::endl;
                r_data << "\t\t<male>" << ageGenderResult.maleProb << "</male>" << std::endl;
            }

            face->emotionsEnable((emotionsDetector.enabled() && i < emotionsDetector.maxBatch));
            if (face->isEmotionsEnabled())
            {
                face->updateEmotions(emotionsDetector[i]);

                r_data << "\t\t<emotions>" << std::endl;
                for (auto& kv : emotionsDetector[i]) 
                    r_data << "\t\t\t<emotion>" << kv.first << "," << kv.second << "</emotion>" << std::endl;

                r_data << "\t\t</emotions>" << std::endl;
            }
            
            face->headPoseEnable((headPoseDetector.enabled() && i < headPoseDetector.maxBatch));
            if (face->isHeadPoseEnabled())
            {
                face->updateHeadPose(headPoseDetector[i]);
                r_data << "\t\t<pose o=\"ryp\">" << headPoseDetector[i].angle_r << "," << headPoseDetector[i].angle_y << "," << headPoseDetector[i].angle_p << "</pose>" << std::endl;
            }
            
            face->landmarksEnable((facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch));
            if (face->isLandmarksEnabled())
            {
                face->updateLandmarks(facialLandmarksDetector[i]);
                r_data << "\t\t<landmarks>" << std::endl;
                for(auto & value: facialLandmarksDetector[i])
                    r_data << "\t\t\t<position>" << value << "</position>" << std::endl;

                r_data << "\t\t</landmarks>" << std::endl;
            }

            r_data << "\t</face>" << std::endl;
            faces.push_back(face);
        }
        r_data << "\t</faces>" << std::endl;
        r_data << "</data>\r\n";
        output.write((uint8_t*)r_data.str().c_str(), r_data.str().size(), fid);

        presenter.drawGraphs(prev_frame);

        // 可视化结果
        if (show) {
            out.str("");
            out << "Total image throughput: " << std::fixed << std::setprecision(2) << 1000.f / (timer["total"].getSmoothedDuration()) << " fps";
            cv::putText(prev_frame, out.str(), THROUGHPUT_METRIC_POSITION, cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(255, 0, 0), 2);

            // drawing faces
            visualizer->draw(prev_frame, faces);
            cv::imshow("Detection Results", prev_frame);
        }

        prev_frame = frame;
        frame = next_frame;
        next_frame = cv::Mat();

        timer.finish("total");

        if (fps > 0)
            delay = std::max(1, static_cast<int>(msrate - timer["total"].getLastCallDuration()));

        if (isLastFrame)
        {

        }
        else if(show)
        {
            int key = cv::waitKey(delay);
            if (27 == key || 'Q' == key || 'q' == key) {
                break;
            }
            presenter.handleKey(key);
        }
    }


    LOG("INFO") << "Number of processed frames: " << framesCounter << std::endl;
    LOG("INFO") << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << std::endl;

    // 显示性能结果
    if (pc) {
        faceDetector.printPerformanceCounts(getFullDeviceName(ie, model["device"]));
        ageGenderDetector.printPerformanceCounts(getFullDeviceName(ie, m_ag["device"]));
        headPoseDetector.printPerformanceCounts(getFullDeviceName(ie, m_hp["device"]));
        emotionsDetector.printPerformanceCounts(getFullDeviceName(ie, m_em["device"]));
        facialLandmarksDetector.printPerformanceCounts(getFullDeviceName(ie, m_lm["device"]));
    }
    std::cout << presenter.reportMeans() << '\n';
    // --------------------------- 8. 处理输出 --------------------------------------------------

    capture.release();
    cv::destroyAllWindows();

    LOG("INFO") << "Exiting ..." << std::endl;
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
