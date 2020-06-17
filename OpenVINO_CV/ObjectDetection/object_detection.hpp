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
	/// Object Detection 对象检测 (Open Model Zoo)
	/// </summary>
	class ObjectDetection : public OpenModelInferBase
	{
	public:
		// 对象检测配置参数
		struct Params2
		{
			std::string model;
			std::string device;
			InferenceEngine::Core ie;
			std::map<std::string, std::string> ie_config;

			bool is_async = true;		//异步推断
			bool is_reshape = true;		//重塑输入
			size_t batch_count = 1;		//最大 batch 数量
			size_t infer_count = 1;		//创建的推断数量

			bool is_mapping_blob = true;					//是否共享映射指定的输出层数据
			bool auto_request_infer = true;					//如果是异步回调，调完成后，自动请求下一次推断
			std::vector<std::string> output_layers;			//需要处理输出的层

			float confidence = 0.5f;						//可置信阈值
			std::vector<std::string> labels;				//对象标签

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
		//对象检测结果数据
		struct Result
		{
			int id;								// batch ID
			int label;							// 预测分类 ID
			float confidence;					// 预测分类的置信值
			cv::Rect location;					// 边界框角的坐标

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
		/// 对象检测构造函数
		/// </summary>
		/// <param name="output_layers_name">多层的网络输出名称</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		//ObjectDetection(bool is_show = true);
		ObjectDetection(const std::vector<std::string>& output_layers_name, bool is_debug = true);
		~ObjectDetection();

#if 0
		/// <summary>
		/// 配置网络
		/// </summary>
		/// <param name="params"></param>
		void ConfigNetwork(const Params2 params);
		/// <summary>
		/// 异步推断请求，需实时提交图像帧；当 is_reshape 为 true 时只需调用一次，多次调用也不影响
		/// </summary>
		/// <param name="frame">图像帧对象</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// 获取上一帧推断所用耗时 ms
		/// </summary>
		/// <returns></returns>
		double GetInferUseTime()
		{
			std::shared_lock<std::shared_mutex> lock(_time_mutex);
			return infer_use_time;
		}

		/// <summary>
		/// 获取每秒推断结果的数量
		/// </summary>
		/// <returns></returns>
		size_t GetFPS()
		{
			std::shared_lock<std::shared_mutex> lock(_fps_mutex);
			return fps;
		}
#endif

		/// <summary>
		/// 设置其它参数
		/// </summary>
		/// <param name="confidence_threshold">信任阈值</param>
		/// <param name="labels">对象标签</param>
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
	/// Object Recognition 对象识别 (Open Model Zoo)
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
