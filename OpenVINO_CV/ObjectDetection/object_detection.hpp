#pragma once

#include <iostream>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "open_model_inference.hpp"


namespace space
{

#pragma region ObjectDetection
	/// <summary>
	/// Object Detection 对象检测 (Open Model Zoo)
	/// </summary>
	class ObjectDetection : public OpenModelInferBase
	{
	public:
		//对象检测结果数据
		struct Result
		{
			int id;								// ID of the image in the batch
			int label;							// predicted class ID
			float confidence;					// confidence for the predicted class
			cv::Rect location;					// coordinates of the bounding box corner
		};

		/// <summary>
		/// 对象检测构造函数
		/// </summary>
		/// <param name="is_debug">是否运行调试输出状态</param>
		ObjectDetection(bool is_debug = true);
		/// <summary>
		/// 对象检测构造函数
		/// </summary>
		/// <param name="output_layers_name">多层的网络输出名称</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		ObjectDetection(const std::vector<std::string> &output_layers_name, bool is_debug = true);
		~ObjectDetection();

		/// <summary>
		/// 设置其它参数
		/// </summary>
		/// <param name="confidence_threshold">信任阈值</param>
		/// <param name="labels">对象标签</param>
		void SetParameters(double confidence_threshold, const std::vector<std::string> labels);

	protected:
		void UpdateDebugShow() override;

	private:
		/// <summary>
		/// 绘制对象边界
		/// </summary>
		/// <param name="frame"></param>
		/// <param name="results"></param>
		/// <param name="labels"></param>
		void DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string>& labels);
		
		cv::Mat debug_frame;
		std::vector<std::string> labels;
		double confidence_threshold = 0.5f;
	};

#pragma endregion
}
