#include "human_pose_detection.hpp"
#include "peak.hpp"
#include <opencv2/opencv.hpp>

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

#if USE_MULTI_INFER
    HumanPoseDetection::HumanPoseDetection(const std::vector<std::string>& output_layers_name, bool is_debug):OpenModelMultiInferBase(output_layers_name, is_debug)
#else
    HumanPoseDetection::HumanPoseDetection(const std::vector<std::string>& output_layers_name, bool is_debug) : OpenModelInferBase(output_layers_name, is_debug)
#endif
    {
    };
    HumanPoseDetection::~HumanPoseDetection()
    {
        if (pBuffer != NULL) UnmapViewOfFile(pBuffer);
        if (pMapFile != NULL) CloseHandle(pMapFile);

        pBuffer = NULL;
        pMapFile = NULL;
    };

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
        
        InferenceEngine::Blob::Ptr pafsBlob;// = request->GetBlob(shared_output_layers[0].first);
        request->GetBlob(shared_output_layers[0].first.c_str(), pafsBlob, 0);

        InferenceEngine::Blob::Ptr heatMapsBlob;// = request->GetBlob(shared_output_layers[1].first);
        request->GetBlob(shared_output_layers[1].first.c_str(), heatMapsBlob, 0);

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
        
        ConvertToSharedXML(poses, "pose2d_output.bin");

        if (is_debug) UpdateDebugShow();
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