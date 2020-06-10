#pragma once

#include <iostream>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "open_model_infer.hpp"
#include "object_recognition.hpp"

namespace space
{
	class ObjectRecognition;

#pragma region ObjectDetection
	/// <summary>
	/// Object Detection ������ (Open Model Zoo)
	/// </summary>
	class ObjectDetection : public OpenModelInferBase
	{
	public:
		//������������
		struct Result
		{
			int id;								// ID of the image in the batch
			int label;							// predicted class ID
			float confidence;					// confidence for the predicted class
			cv::Rect location;					// coordinates of the bounding box corner
		};

		/// <summary>
		/// �����⹹�캯��
		/// </summary>
		/// <param name="output_layers_name">���������������</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		ObjectDetection(const std::vector<std::string>& output_layers_name, bool is_debug = true);
		~ObjectDetection();

		/// <summary>
		/// ������������
		/// </summary>
		/// <param name="confidence_threshold">������ֵ</param>
		/// <param name="labels">�����ǩ</param>
		void SetParameters(const OpenModelInferBase::Params& params, double confidence_threshold, const std::vector<std::string> labels);

		static void DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string>& labels);

		void AddSubNetwork(ObjectRecognition* sub_network);

	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
		void UpdateDebugShow() override;

	private:

		cv::Mat debug_frame;
		std::vector<std::string> labels;
		double confidence_threshold = 0.5f;

		std::vector<ObjectRecognition*> sub_network;
		std::vector<ObjectDetection::Result> results;
	};
#pragma endregion

#pragma region ObjectRecognition
	/// <summary>
	/// Object Recognition ����ʶ�� (Open Model Zoo)
	/// </summary>
	class ObjectRecognition : public OpenModelInferBase
	{
	public:
		ObjectRecognition(const std::vector<std::string>& output_layers_name, bool is_debug = true);
		~ObjectRecognition();

		void RequestInfer(const cv::Mat& frame, const std::vector<ObjectDetection::Result> results);


	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
		void UpdateDebugShow() override;

		void SetInferCallback() override;

	private:
		cv::Mat prev_frame;
		std::vector<ObjectDetection::Result> results;
		//std::vector<ObjectDetection::Result> prev_results;

		std::string title = "";

		std::vector<float> last_fv;
	};
#pragma endregion
};
