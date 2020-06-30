#include "human_pose_detection.hpp"
#include "peak.hpp"
#include <opencv2/opencv.hpp>

#include "ie_compound_blob.h"


namespace space
{
    HumanPose::HumanPose(const std::vector<cv::Point2f>& keypoints, const float& score) : keypoints(keypoints), score(score) {};

    class FindPeaksBody : public cv::ParallelLoopBody {
    public:
        FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance, std::vector<std::vector<Peak> >& peaksFromHeatMap)
            : heatMaps(heatMaps), minPeaksDistance(minPeaksDistance), peaksFromHeatMap(peaksFromHeatMap) {}

        virtual void operator()(const cv::Range& range) const
        {
            for (int i = range.start; i < range.end; i++) 
            {
                findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
            }
        }

    private:
        const std::vector<cv::Mat>& heatMaps;
        float minPeaksDistance;
        std::vector<std::vector<Peak> >& peaksFromHeatMap;
    };


    HumanPoseDetection::HumanPoseDetection(bool is_debug):is_debug(is_debug)
    {
    };
    HumanPoseDetection::~HumanPoseDetection()
    {
        if (pBuffer != NULL) UnmapViewOfFile(pBuffer);
        if (pMapFile != NULL) CloseHandle(pMapFile);

        pBuffer = NULL;
        pMapFile = NULL;
    };


    void HumanPoseDetection::ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_info)
    {
        std::map<std::string, std::string> model = ParseArgsForModel(model_info);
        return ConfigNetwork(ie, model["path"], model["device"]);
    }
    void HumanPoseDetection::ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device)
    {
        if (infer_count < 1)
            throw std::invalid_argument("Infer Request 数量不能小于 1 .");

        if (infer_count > 1)
            ie_config[InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD] = InferenceEngine::PluginConfigParams::NO;

        LOG("INFO") << "正在配置网络模型 ... " << std::endl;

        //1.
        LOG("INFO") << "\t读取 IR 文件: " << model_path << " [" << device << "]" << std::endl;
        cnnNetwork = ie.ReadNetwork(model_path);

        //2.
        SetNetworkIO();	

        //3.
        LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
        if (ie_config.size() > 0)
            LOG("INFO") << "\t" << ie_config.begin()->first << "=" << ie_config.begin()->second << std::endl;

        execNetwork = ie.LoadNetwork(cnnNetwork, device, ie_config);
        LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
        for (int i = 0; i < infer_count; i++)
        {
            InferenceEngine::InferRequest::Ptr inferPtr = execNetwork.CreateInferRequestPtr();
            requestPtrs.push_back(inferPtr);
        }

        //4.
        LOG("INFO") << "正在配置异步推断回调 ... " << std::endl;
        SetInferCallback();	//设置异步回调

        is_configuration = true;
        LOG("INFO") << "网络模型配置完成 ... " << std::endl;
    }

    void HumanPoseDetection::SetNetworkIO()
    {
        LOG("INFO") << "\t配置网络输入 ... " << std::endl;
        inputsInfo = cnnNetwork.getInputsInfo();
        inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
        inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        
        inputsInfo.begin()->second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::I420);

        //print inputs info
        for (const auto& input : inputsInfo)
        {
            InferenceEngine::SizeVector inputDims = input.second->getTensorDesc().getDims();
            LOG("INFO") << "\tInput Name:[" << input.first << "]  Shape:" << inputDims << "  Precision:[" << input.second->getPrecision() << "]" << " ColorFormat:" << input.second->getPreProcess().getColorFormat() << std::endl;
        }

        //默认支持一层输入，子类可以更改为多层输入
        //多层输入以输入层名.bin文件指向
        //那bin的数据类型，精度怎么解释呢？
        if (inputsInfo.size() != 1)
        {
            LOG("WARN") << "当前模块只支持一个网络层的输入 ... " << std::endl;
            throw std::logic_error("当前模块只支持一个网络层的输入 ... ");
        }


        LOG("INFO") << "\t配置网络输出  ... " << std::endl;
        outputsInfo = cnnNetwork.getOutputsInfo();
        outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
        if (output_layers.size() == 0)
            for (const auto& output : outputsInfo)
                output_layers.push_back(output.first);

        //print outputs info
        for (const auto& output : outputsInfo)
        {
            InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
            LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << outputDims << "  Precision:[" << output.second->getPrecision() << "]" << std::endl;
        }

        //输出调试标题
        InferenceEngine::SizeVector outputDims = outputsInfo.begin()->second->getTensorDesc().getDims();
        debug_title.str("");
        debug_title << "[" << cnnNetwork.getName() << "] Output Size:" << outputsInfo.size() << "  Name:[" << outputsInfo.begin()->first << "]  Shape:" << outputDims << std::endl;
    }

    void HumanPoseDetection::SetInferCallback()
    {
        for (auto& requestPtr : requestPtrs)
        {
            requestPtr->SetCompletionCallback<FCompletionCallback>((FCompletionCallback)
                [&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code)
                {
                    {
                        t1 = std::chrono::high_resolution_clock::now();
                        std::lock_guard<std::shared_mutex> lock(_time_mutex);
                        infer_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
                    };
                    {
                        std::lock_guard<std::shared_mutex> lock(_fps_mutex);
                        fps_count++;
                    };

                    //1.解析数据，外部解析 OR 内部解析
                    ParsingOutputData(request, code);

                    //2.循环再次启动异步推断
                    request->StartAsync(0);
                    t0 = std::chrono::high_resolution_clock::now();

                    //3.更新显示结果
                    if (is_debug) UpdateDebugShow();
                });
        }
        timer.start(1000, [&]
            {
                std::lock_guard<std::shared_mutex> lock(_fps_mutex);
                fps = fps_count;
                fps_count = 0;
            });
    }

    void HumanPoseDetection::RequestInfer(const cv::Mat& frame)
    {
        if (!is_configuration)
            throw std::logic_error("未配置网络对象，不可操作 ...");
        if (frame.empty() || frame.type() != CV_8UC3)
            throw std::logic_error("输入图像帧不能为空帧对象，CV_8UC3 类型 ... ");

        frame_ptr = frame;
        InferenceEngine::StatusCode code;
        InferenceEngine::InferRequest::Ptr requestPtr;

        //查找没有启动的推断对象
        for (auto& request : requestPtrs)
        {
            code = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
            if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED)
            {
                requestPtr = request;
                break;
            }
        }
        if (requestPtr == nullptr) return;

        InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
        LOG("INFO") << "找到一个 InferRequestPtr 对象，Input Address:" << (void*)inputBlob->buffer().as<uint8_t*>() << std::endl;

        //if (inputsInfo.begin()->second->getPreProcess().getColorFormat() == InferenceEngine::ColorFormat::BGR)
        {
            size_t width = frame.cols;
            size_t height = frame.rows;
            size_t channels = frame.channels();

            size_t strideH = frame.step.buf[0];
            size_t strideW = frame.step.buf[1];

            bool is_dense = strideW == channels && strideH == channels * width;
            if (!is_dense) throw std::logic_error("输入的图像帧不支持转换 ... ");

            //重新设置输入层 Blob，将输入图像内存数据指针共享给输入层，做实时或异步推断
            //输入为图像原始尺寸，其实是会存在问题的，如果源图尺寸很大，那处理时间会更长
            InferenceEngine::TensorDesc tensor = inputBlob->getTensorDesc();
            InferenceEngine::TensorDesc n_tensor(tensor.getPrecision(), { 1, channels, height, width }, tensor.getLayout());
            requestPtr->SetBlob(inputsInfo.begin()->first, InferenceEngine::make_shared_blob<uint8_t>(n_tensor, frame.data));
        }

        requestPtr->StartAsync();
        t0 = std::chrono::high_resolution_clock::now();
    }

    void HumanPoseDetection::RequestInfer(const std::string& name)
    {
        InferenceEngine::StatusCode code;
        InferenceEngine::InferRequest::Ptr requestPtr;

        //查找没有启动的推断对象
        for (auto& request : requestPtrs)
        {
            code = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
            if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED)
            {
                requestPtr = request;
                break;
            }
        }
        if (requestPtr == nullptr) return;

        size_t width = 0;
        size_t height = 0;
        static bool createMapFile = false;
        if (!createMapFile)
        {
            if (!GetOnlyReadMapFile(inFile, inBuffer, name.c_str()))
            {
                LOG("ERROR") << "Open Memory File: " << name << " Failed!" << std::endl;
                throw std::logic_error("打开内存文件失败 ... ");
            }
            LOG("INFO") << "Open Memory File: " << name << " Success!" << std::endl;

            frame = (Frame*)inBuffer;
            LOG("INFO") << frame << std::endl;

            size_t width = static_cast<size_t>(frame->width);
            size_t height = static_cast<size_t>(frame->height);
            LOG("INFO") << width << "," << height << std::endl;

            if (frame_ptr.empty())
                frame_ptr.create(height, width, CV_8UC3);

            createMapFile = true;
        }

        uint8_t* framePtr = (uint8_t*)inBuffer + 32;
        size_t offset_y = 0;
        size_t offset_u = offset_y + height * width;
        size_t offset_v = offset_u + offset_u / 2;

        cv::Mat yuv(height, width, CV_8UC2);
        yuv.data = (uint8_t*)inBuffer + 32;

        cv::cvtColor(yuv, frame_ptr, cv::ColorConversionCodes::COLOR_YUV2RGB_Y422);
        cv::imshow("Test", frame_ptr);
        return;

        InferenceEngine::Blob::Ptr y = InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, {1, 1,  height, width}, InferenceEngine::Layout::NHWC }, framePtr + offset_y);
        InferenceEngine::Blob::Ptr u = InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, {1, 1,  height/2, width/2}, InferenceEngine::Layout::NHWC }, framePtr + offset_u);
        InferenceEngine::Blob::Ptr v = InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8, {1, 1,  height/2, width/2}, InferenceEngine::Layout::NHWC }, framePtr + offset_v);

        InferenceEngine::Blob::Ptr input = InferenceEngine::make_shared_blob<InferenceEngine::I420Blob>(y, u, v);

        requestPtr->SetBlob(inputsInfo.begin()->first, input);

        requestPtr->StartAsync();
        t0 = std::chrono::high_resolution_clock::now();
    }

    std::vector<HumanPose> HumanPoseDetection::extractPoses(const std::vector<cv::Mat>& heatMaps, const std::vector<cv::Mat>& pafs) const 
    {
        std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
        FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
        cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())), findPeaksBody);

        int peaksBefore = 0;
        for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) 
        {
            peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
            for (auto& peak : peaksFromHeatMap[heatmapId])
            {
                peak.id += peaksBefore;
            }
        }

        std::vector<HumanPose> poses = groupPeaksToPoses(
            peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
            foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);

        return poses;
    }
   
    void HumanPoseDetection::correctCoordinates(std::vector<HumanPose>& poses, const cv::Size& featureMapsSize,  const cv::Size& imageSize) const 
    {
        CV_Assert(stride % upsampleRatio == 0);

        cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

        float scaleX = imageSize.width / static_cast<float>(fullFeatureMapSize.width);
        float scaleY = imageSize.height / static_cast<float>(fullFeatureMapSize.height);

        for (auto& pose : poses) 
        {
            for (auto& keypoint : pose.keypoints) 
            {
                if (keypoint != cv::Point2f(-1, -1))
                {
                    keypoint.x *= stride / upsampleRatio;
                    keypoint.x *= scaleX;

                    keypoint.y *= stride / upsampleRatio;
                    keypoint.y *= scaleY;
                }
            }
        }
    }

    void HumanPoseDetection::RenderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image)
    {
        CV_Assert(image.type() == CV_8UC3);

        static const cv::Scalar colors[keypointsNumber] = {
            cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0), cv::Scalar(255, 170, 0),
            cv::Scalar(255, 255, 0), cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
            cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85), cv::Scalar(0, 255, 170),
            cv::Scalar(0, 255, 255), cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
            cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255), cv::Scalar(170, 0, 255),
            cv::Scalar(255, 0, 255), cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)
        };
        static const std::pair<int, int> limbKeypointsIds[] = {
            {1, 2},  {1, 5},   {2, 3},
            {3, 4},  {5, 6},   {6, 7},
            {1, 8},  {8, 9},   {9, 10},
            {1, 11}, {11, 12}, {12, 13},
            {1, 0},  {0, 14},  {14, 16},
            {0, 15}, {15, 17}
        };

        const int stickWidth = 4;
        const cv::Point2f absentKeypoint(-1.0f, -1.0f);
        for (const auto& pose : poses)
        {
            CV_Assert(pose.keypoints.size() == keypointsNumber);

            for (size_t keypointIdx = 0; keypointIdx < pose.keypoints.size(); keypointIdx++)
            {
                if (pose.keypoints[keypointIdx] != absentKeypoint) {
                    cv::circle(image, pose.keypoints[keypointIdx], 4, colors[keypointIdx], -1);
                }
            }
        }
        cv::Mat pane = image.clone();
        for (const auto& pose : poses)
        {
            for (const auto& limbKeypointsId : limbKeypointsIds)
            {
                std::pair<cv::Point2f, cv::Point2f> limbKeypoints(pose.keypoints[limbKeypointsId.first],
                    pose.keypoints[limbKeypointsId.second]);
                if (limbKeypoints.first == absentKeypoint || limbKeypoints.second == absentKeypoint) {
                    continue;
                }

                float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
                float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
                cv::Point difference = limbKeypoints.first - limbKeypoints.second;
                double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
                int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
                std::vector<cv::Point> polygon;
                cv::ellipse2Poly(cv::Point2d(meanX, meanY), cv::Size2d(length / 2, stickWidth),
                    angle, 0, 360, 1, polygon);
                cv::fillConvexPoly(pane, polygon, colors[limbKeypointsId.second]);
            }
        }
        cv::addWeighted(image, 0.4, pane, 0.6, 0, image);
    }

    void HumanPoseDetection::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code)
    {
        if (outputsInfo.size() != 2) return;
        
        InferenceEngine::Blob::Ptr pafsBlob;// = request->GetBlob(output_shared_layers[0].first);
        request->GetBlob(output_layers[0].c_str(), pafsBlob, 0);

        InferenceEngine::Blob::Ptr heatMapsBlob;// = request->GetBlob(output_shared_layers[1].first);
        request->GetBlob(output_layers[1].c_str(), heatMapsBlob, 0);

        InferenceEngine::SizeVector heatMapDims = heatMapsBlob->getTensorDesc().getDims();
        
        const float* heatMapsData = heatMapsBlob->buffer().as<float*>();
        const float* pafsData = pafsBlob->buffer().as<float*>();

#if _DEBUG
        std::cout << "Address:" << (void*)heatMapsData << std::endl;
#endif

        int offset = heatMapDims[2] * heatMapDims[3];
        int nPafs = pafsBlob->getTensorDesc().getDims()[1];

        int featureMapHeight = heatMapDims[2];
        int featureMapWidth = heatMapDims[3];

        std::vector<cv::Mat> heatMaps(keypointsNumber);
        for (size_t i = 0; i < heatMaps.size(); i++)
        {
            heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1, (void*)(heatMapsData + i * offset));
            cv::resize(heatMaps[i], heatMaps[i], cv::Size(), upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
        }

        std::vector<cv::Mat> pafs(nPafs);
        for (size_t i = 0; i < pafs.size(); i++)
        {
            pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1, (void*)(pafsData + i * offset));
            cv::resize(pafs[i], pafs[i], cv::Size(), upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
        }

        poses = extractPoses(heatMaps, pafs);
        correctCoordinates(poses, heatMaps[0].size(), frame_ptr.size());

        LOG("INFO") << poses.size() << std::endl;
        ConvertToSharedXML(poses, "pose2d_output.bin");
    }
    
    void HumanPoseDetection::ConvertToSharedXML(const std::vector<HumanPose>& poses, const std::string shared_name)
    {
        static bool has_shared = false;

        if (!has_shared)
        {
            LOG("INFO") << std::endl;
            if (CreateOnlyWriteMapFile(pMapFile, pBuffer, 1024 * 1024 * 2, shared_name.c_str()))
                osd_data = (OutputSourceData*)pBuffer;
            else
                osd_data = new OutputSourceData();

            osd_data->fid = 0;
            has_shared = true;
        }

        osd_data->fid += 1;

        std::stringstream x_data;
        x_data.str("");
        x_data << "<data>\r\n";
        x_data << "\t<fid>" << osd_data->fid << "</fid>\r\n";
        x_data << "\t<count>" << poses.size() << "</count>\r\n";
        x_data << "\t<name>HumanPose2D</name>\r\n";
        for (HumanPose const& pose : poses)
        {
            x_data << "\t<pose confidence=\"" << pose.score << "\">\r\n";
            for (auto const& keypoint : pose.keypoints)
                x_data << "\t\t<position>" << keypoint.x << "," << keypoint.y << "</position>\r\n";

            x_data << "\t</pose>\r\n";
        }
        x_data << "</data>";

        std::string data = x_data.str();
        osd_data->length = data.size();
        osd_data->format = OutputFormat::XML;
        if(pBuffer != NULL)
            std::copy((uint8_t*)data.c_str(), (uint8_t*)data.c_str() + data.length(), (uint8_t*)pBuffer);
    }
    
    void HumanPoseDetection::UpdateDebugShow()
    {
        frame_ptr.copyTo(debug_frame);
        RenderHumanPose(poses, debug_frame);

        std::stringstream use_time;
        use_time << "Detection Count:" << poses.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
        cv::putText(debug_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

        cv::imshow(debug_title.str(), debug_frame);
        cv::waitKey(1);
    }

}