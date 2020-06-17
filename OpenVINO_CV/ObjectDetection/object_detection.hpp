#pragma once

#include <iostream>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "open_model_infer.hpp"

namespace space
{

#pragma region ObjectDetection
	/// <summary>
	/// Object Detection ������ (Open Model Zoo)
	/// </summary>
	class ObjectDetection : public OpenModelInferBase
	{
	public:
		// ���������ò���
		struct Params2
		{
			std::string model;
			std::string device;
			InferenceEngine::Core ie;
			std::map<std::string, std::string> ie_config;

			bool is_async = true;		//�첽�ƶ�
			bool is_reshape = true;		//��������
			size_t batch_count = 1;		//��� batch ����
			size_t infer_count = 1;		//�������ƶ�����

			bool is_mapping_blob = true;					//�Ƿ���ӳ��ָ�������������
			bool auto_request_infer = true;					//������첽�ص�������ɺ��Զ�������һ���ƶ�
			std::vector<std::string> output_layers;			//��Ҫ��������Ĳ�

			float confidence = 0.5f;						//��������ֵ
			std::vector<std::string> labels;				//�����ǩ

			friend std::ostream& operator << (std::ostream& stream, const Params2& params)
			{
				stream << "[Params Async:" << params.is_async <<
					" Reshape:" << params.is_reshape <<
					" BatchCount:" << params.batch_count <<
					" InferCount:" << params.infer_count <<
					" MappingBlob:" << params.is_mapping_blob <<
					" OutputLayers:" << params.output_layers << "]";

				return stream;
			};
		};
		//������������
		struct Result
		{
			int id;								// batch ID
			int label;							// Ԥ����� ID
			float confidence;					// Ԥ����������ֵ
			cv::Rect location;					// �߽��ǵ�����

			friend std::ostream& operator << (std::ostream& stream, const Result& result)
			{
				stream << "[Result ID:" << result.id <<
					" Label:" << result.label <<
					" Confidence:" << result.confidence << 
					" Rect:" << result.location << "]";

				return stream;
			};
		};

		/// <summary>
		/// �����⹹�캯��
		/// </summary>
		/// <param name="output_layers_name">���������������</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		//ObjectDetection(bool is_show = true);
		ObjectDetection(const std::vector<std::string>& output_layers_name, bool is_debug = true);
		~ObjectDetection();

#if 0
		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="params"></param>
		void ConfigNetwork(const Params2 params);
		/// <summary>
		/// �첽�ƶ�������ʵʱ�ύͼ��֡���� is_reshape Ϊ true ʱֻ�����һ�Σ���ε���Ҳ��Ӱ��
		/// </summary>
		/// <param name="frame">ͼ��֡����</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// ��ȡ��һ֡�ƶ����ú�ʱ ms
		/// </summary>
		/// <returns></returns>
		double GetInferUseTime()
		{
			std::shared_lock<std::shared_mutex> lock(_time_mutex);
			return infer_use_time;
		}

		/// <summary>
		/// ��ȡÿ���ƶϽ��������
		/// </summary>
		/// <returns></returns>
		size_t GetFPS()
		{
			std::shared_lock<std::shared_mutex> lock(_fps_mutex);
			return fps;
		}
#endif

		/// <summary>
		/// ������������
		/// </summary>
		/// <param name="confidence_threshold">������ֵ</param>
		/// <param name="labels">�����ǩ</param>
		void SetParameters(const OpenModelInferBase::Params& params, float confidence_threshold, const std::vector<std::string> labels);

		static void DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string>& labels);

	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
		void UpdateDebugShow() override;

	private:
		//[1x1xNx7] FP32
		void PasingO1D1x1xNx7(InferenceEngine::IInferRequest::Ptr request);
		//[Nx5][N]
		void PasingO2DNx5(InferenceEngine::IInferRequest::Ptr request);

		float scaling = 1.2;
		cv::Mat debug_frame;
		std::vector<std::string> labels;
		float confidence_threshold = 0.5f;


		//std::vector<ObjectRecognition*> sub_network;
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
