#pragma once

#define USE_MULTI_INFER false

#if USE_MULTI_INFER
#include <open_model_multi_infer.hpp>
#else
#include <open_model_infer.hpp>
#endif

#include <static_functions.hpp>
#include <io_data_format.hpp>

namespace space
{
    
    struct HumanPose
    {
        HumanPose(const std::vector<cv::Point2f>& keypoints = std::vector<cv::Point2f>(), const float& score = 0.0f);

        std::vector<cv::Point2f> keypoints;
        float score;
    };

#if USE_MULTI_INFER
    class HumanPoseDetection :  public OpenModelMultiInferBase
#else
    class HumanPoseDetection : public OpenModelInferBase
#endif
    {
    public:
        //HumanPoseDetection(bool is_debug = true);
        HumanPoseDetection(const std::vector<std::string>& output_layers_name, bool is_debug = true);
        ~HumanPoseDetection();

    protected:
        void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
        void UpdateDebugShow() override;

        void ConvertToSharedXML(const std::vector<HumanPose>& poses, const std::string shared_name);

    private:
        cv::Mat debug_frame;
        std::vector<HumanPose> poses;

        HANDLE pMapFile;
        LPVOID pBuffer;
        OutputSourceData *osd_data;

        float minPeaksDistance = 3.0f;
        static const int upsampleRatio = 4;
        static const size_t keypointsNumber = 18;
        float midPointsScoreThreshold = 0.05f;
        float foundMidPointsRatioThreshold = 0.8f;
        int minJointsNumber = 3;
        float minSubsetScore = 0.2f;
        int stride = 8;

        std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps, const std::vector<cv::Mat>& pafs) const;
        void correctCoordinates(std::vector<HumanPose>& poses, const cv::Size& featureMapsSize, const cv::Size& imageSize) const;

        void RenderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image);
    };
}
