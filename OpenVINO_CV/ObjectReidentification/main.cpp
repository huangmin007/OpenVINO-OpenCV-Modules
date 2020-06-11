// ObjectReidentification.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "cmdline.h"
//#include "input_source.hpp"
#include "static_functions.hpp"

#include "utils.hpp"
#include "action_detector.hpp"
#include <monitors\presenter.h>
#include "image_grabber.hpp"

using namespace space;

int main(int argc, char** argv)
{
    if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)space::CloseHandlerRoutine, TRUE) == FALSE)
    {
        LOG("ERROR") << "Unable to Install Close Handler Routine!" << std::endl;
        return EXIT_FAILURE;
    }

#pragma region cmdline参数设置解析
    //---------------- 第一步：设计输入参数 --------------confidence
    cmdline::parser args;
    args.add("help", 'h', "参数说明");
    args.add("info", 0, "Inference Engine Infomation");

    args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared)[:value[:value[:...]]]", false, "camera:0:640x480");
    args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
        "示例：face-detection-adas-0001:FP32:CPU 或 face-detection-adas-0001:FP16:GPU", false, "face-detection-adas-0001:FP32:CPU");
    //face-detection-adas-0001,person-detection-retail-0013,face-detection-0105,boxes/Split.0:labels
    args.add<std::string>("output_layers", 'o', "(output layer names)多层网络输出参数，单层使用默认输出，网络层名称，以':'分割，区分大小写，格式：layerName:layerName:...", false, "");
    args.add<float>("conf", 'c', "检测结果的置信度阈值(confidence threshold)", false, 0.5);
    args.add<bool>("reshape", 'r', "重塑输入层，使输入源内存映射到网络输入层实现共享内存数据，不进行数据源缩放和拷贝", false, true);

    //facial-landmarks-35-adas-0002  landmarks-regression-retail-0009
    //face-reidentification-retail-0095  
    args.add<std::string>("fr", 0, "用于 AI识别检测 的 联级网络模型名称/文件(.xml)和目标设备", false, "face-reidentification-retail-0095:FP32:CPU");
    args.add<std::string>("lm", 0, "用于 AI识别检测 的 联级网络模型名称/文件(.xml)和目标设备", false, "landmarks-regression-retail-0009:FP32:CPU");

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
        InferenceEngineInfomation(args.get<std::string>("model"));
        return EXIT_SUCCESS;
    }
    if (!isParser)
    {
        std::cerr << "[ERROR] " << args.error() << std::endl << args.usage();
        return EXIT_SUCCESS;
    }

    bool show = args.get<bool>("show");
    std::map<std::string, std::string> model = ParseArgsForModel(args.get<std::string>("model"));
    std::map<std::string, std::string> fr_model = ParseArgsForModel(args.get<std::string>("fr"));
    std::map<std::string, std::string> lm_model = ParseArgsForModel(args.get<std::string>("lm"));
#pragma endregion

    std::string FLAGS_teacher_id = "";
    int FLAGS_a_top = -1;
    std::string FLAGS_student_ac = "sitting,standing,raising_hand";
    std::string FLAGS_top_ac = "sitting,raising_hand";
    std::string FLAGS_teacher_ac = "standing,writing,demonstrating";

    const auto actions_type = FLAGS_teacher_id.empty()
        ? FLAGS_a_top > 0 ? TOP_K : STUDENT
        : TEACHER;
    const auto actions_map = actions_type == STUDENT
        ? ParseActionLabels(FLAGS_student_ac)
        : actions_type == TOP_K
        ? ParseActionLabels(FLAGS_top_ac)
        : ParseActionLabels(FLAGS_teacher_ac);
    const auto num_top_persons = actions_type == TOP_K ? FLAGS_a_top : -1;
    const auto top_action_id = actions_type == TOP_K
        ? std::distance(actions_map.begin(), find(actions_map.begin(), actions_map.end(), "raising_hand"))
        : -1;

    ImageGrabber cap("cam");
    if (!cap.IsOpened()) {
        LOG("INFO") << "Cannot open the video" << std::endl;
        return 1;
    }

    std::string fd_model_path = model["path"];
    float exp_r_fd = 1.15, t_reg_fd = 0.9f;
    int input_h = 600, input_w = 600;
    float confidence_threshold = 0.6f;

    int min_size_fr = 128;
    try
    {
        InferenceEngine::Core ie;
        LOG("INFO") << "Loading Inference Engine" << std::endl;
        if (model["device"] == "CPU")
        {
            LOG("INFO") << ie.GetVersions("CPU") << std::endl;
            ie.SetConfig({ {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES} }, "CPU");
        }

        // Load face detector
        std::unique_ptr<AsyncDetection<detection::DetectedObject>> face_detector;
        detection::DetectorConfig face_config(fd_model_path);
        face_config.deviceName = model["device"];
        face_config.ie = ie;
        face_config.is_async = true;
        face_config.confidence_threshold = confidence_threshold;
        face_config.input_h = input_h;
        face_config.input_w = input_w;
        face_config.increase_scale_x = exp_r_fd;
        face_config.increase_scale_y = exp_r_fd;
        face_detector.reset(new detection::FaceDetection(face_config));

        // Create face recognizer
        std::unique_ptr<FaceRecognizer> face_recognizer;
        detection::DetectorConfig face_registration_det_config(fd_model_path);
        face_registration_det_config.deviceName = model["device"];
        face_registration_det_config.ie = ie;
        face_registration_det_config.is_async = false;
        face_registration_det_config.confidence_threshold = t_reg_fd;
        face_registration_det_config.increase_scale_x = exp_r_fd;
        face_registration_det_config.increase_scale_y = exp_r_fd;

        //face-reidentification-retail-0095
        CnnConfig reid_config(fr_model["path"]);
        reid_config.deviceName = fr_model["device"];
        if (checkDynamicBatchSupport(ie, fr_model["device"]))
        {
            reid_config.max_batch_size = 16;
            LOG("INFO") << "Batch .." << std::endl;
        }
        else
            reid_config.max_batch_size = 1;
        reid_config.ie = ie;

        //landmarks
        CnnConfig landmarks_config(lm_model["path"]);
        landmarks_config.deviceName = lm_model["device"];
        if (checkDynamicBatchSupport(ie, lm_model["device"]))
            landmarks_config.max_batch_size = 16;
        else
            landmarks_config.max_batch_size = 1;
        landmarks_config.ie = ie;

        face_recognizer.reset(new FaceRecognizerDefault(
            landmarks_config, reid_config, face_registration_det_config,
            "faces_gallery.json", 0.7, min_size_fr, false, false));
        //face_recognizer->LabelExists("");

        // Create tracker for reid
        TrackerParams tracker_reid_params;
        tracker_reid_params.min_track_duration = 1;
        tracker_reid_params.forget_delay = 150;
        tracker_reid_params.affinity_thr = 0.8f;
        tracker_reid_params.averaging_window_size_for_rects = 1;
        tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
        tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_reid_params.drop_forgotten_tracks = false;
        tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_reid_params.objects_type = "face";
        Tracker tracker_reid(tracker_reid_params);

        cv::Mat frame, prev_frame;

        float work_time_ms = 0.f;
        float wait_time_ms = 0.f;
        size_t work_num_frames = 0;
        size_t wait_num_frames = 0;
        size_t total_num_frames = 0;
        const char ESC_KEY = 27;
        const char SPACE_KEY = 32;
        const cv::Scalar green_color(0, 255, 0);
        const cv::Scalar red_color(0, 0, 255);
        const cv::Scalar white_color(255, 255, 255);
        std::vector<std::map<int, int>> face_obj_id_to_action_maps;
        std::map<int, int> top_k_obj_ids;

        int teacher_track_id = -1;

        if (cap.GrabNext()) {
            cap.Retrieve(frame);
        }
        else {
            LOG("ERROR") << "无法读取第一帧" << std::endl;
            return 1;
        }

        face_detector->enqueue(frame);
        face_detector->submitRequest();

        prev_frame = frame.clone();

        bool is_last_frame = false;
        bool is_monitoring_enabled = false;
        auto prev_frame_path = cap.GetVideoPath();

        cv::VideoWriter vid_writer;
        //if (!FLAGS_out_v.empty()) {
        //    vid_writer = cv::VideoWriter(FLAGS_out_v, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.GetFPS(), Visualizer::GetOutputSize(frame.size()));
        //}
        int num_top_persons = -1;
        Visualizer sc_visualizer(show, vid_writer, num_top_persons);
        DetectionsLogger logger(std::cout, false, "", "");

        const int smooth_window_size = static_cast<int>(cap.GetFPS() * 1.0);
        const int smooth_min_length = static_cast<int>(cap.GetFPS() * 1.0);

        cv::Size graphSize{ static_cast<int>(frame.cols / 4), 60 };
        Presenter presenter("", frame.rows - graphSize.height - 10, graphSize);
        LOG("DEBUG") << "Hello" << std::endl;

        while (!is_last_frame)
        {
            logger.CreateNextFrameRecord(cap.GetVideoPath(), work_num_frames, prev_frame.cols, prev_frame.rows);
            auto started = std::chrono::high_resolution_clock::now();

            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }

            presenter.handleKey(key);
            presenter.drawGraphs(prev_frame);
            sc_visualizer.SetFrame(prev_frame);

            

            face_detector->wait();
            detection::DetectedObjects faces = face_detector->fetchResults();
            //LOG("DEBUG") << "Faces:" << faces.size() << std::endl;

            if (!is_last_frame) {
                face_detector->enqueue(frame);
                face_detector->submitRequest();
            }
            auto ids = face_recognizer->Recognize(prev_frame, faces);
            TrackedObjects tracked_face_objects;

            if (show)
                cv::imshow("Test", prev_frame);

            for (size_t i = 0; i < faces.size(); i++) 
            {
                tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
            }
            //LOG("DEBUG") << "tracked_face_objects:" << tracked_face_objects.size() << " work_num_frames:" << work_num_frames << std::endl;
            tracker_reid.Process(prev_frame, tracked_face_objects, work_num_frames);

            const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();
            //LOG("DEBUG") << "Tracked Faces:" << tracked_faces.size() << std::endl;

            auto elapsed = std::chrono::high_resolution_clock::now() - started;
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

            work_time_ms += elapsed_ms;

            std::map<int, int> frame_face_obj_id_to_action;
            for (size_t j = 0; j < tracked_faces.size(); j++) 
            {
                //LOG("DEBUG") << "Tracked.." << std::endl;

                const auto& face = tracked_faces[j];
                std::string face_label = face_recognizer->GetLabelByID(face.label);

                std::string label_to_draw;
                if (face.label != EmbeddingsGallery::unknown_id)
                    label_to_draw += face_label;

                sc_visualizer.DrawObject(face.rect, label_to_draw, red_color, white_color, true);
            }

            sc_visualizer.DrawFPS(1e3f / (work_time_ms / static_cast<float>(work_num_frames) + 1e-6f), red_color);
            ++work_num_frames;

            ++total_num_frames;            
            sc_visualizer.Show();

            prev_frame = frame.clone();
            logger.FinalizeFrameRecord();
        }
        sc_visualizer.Finalize();
    }
    catch (const std::exception &ex)
    {
        LOG("WRROR") << ex.what() << std::endl;
    }
    catch (...)
    {
        LOG("ERROR") << "发生未知/内部异常 ... " << std::endl;
    }

    std::cout << std::endl;

    
    if (show) cv::destroyAllWindows();
    LOG("INFO") << "Exiting ... " << std::endl;

    Sleep(500);
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
