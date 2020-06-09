#pragma once
/*
#include "object_detection.hpp"
#include <open_model_infer.hpp>

namespace space
{
	class ObjectDetection;
	class ObjectDetection::Result;

    class ObjectRecognition : public OpenModelInferBase
    {
    public:
		ObjectRecognition(const std::vector<std::string>& output_layers_name, bool is_debug = true);
		~ObjectRecognition();

		void RequestInfer(const cv::Mat& frame, const std::vector<ObjectDetection::Result> &results);


	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
		void UpdateDebugShow() override;

		void SetInferCallback() override;

	private:
    };
}
*/