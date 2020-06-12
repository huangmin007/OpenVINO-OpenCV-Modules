#pragma once
#include <open_model_infer.hpp>

namespace space
{
    //对象检测结果数据
    struct ObjectResult
    {
        int id;								// ID of the image in the batch
        int label;							// predicted class ID
        float confidence;					// confidence for the predicted class
        cv::Rect location;					// coordinates of the bounding box corner
    };

    class ObjectRecognition :
        public OpenModelInferBase
    {
    public:
        ObjectRecognition(const std::vector<std::string>& input_shared_layers, const std::vector<std::string>& output_layers, bool is_debug = true);
        ~ObjectRecognition();

        void RequestInfer(const cv::Mat& frame);

    protected:
        void SetMemoryShared() override;
        void SetInferCallback() override;
        void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
        void UpdateDebugShow() override;

    private:
        std::vector<std::string> input_shared;
        std::vector<HANDLE> input_shared_handle;
        std::vector<std::pair<std::string, LPVOID>> input_shared_layers;
    
        cv::Mat debug_frame;
        cv::Mat prev_frame;
        std::vector<ObjectResult> results;
    };

}