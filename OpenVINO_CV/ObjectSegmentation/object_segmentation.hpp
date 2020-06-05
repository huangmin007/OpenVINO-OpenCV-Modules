#pragma once

#include <iostream>
#include <Windows.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <open_model_infer.hpp>

namespace space
{
	/// <summary>
	/// Object/Screen/Instance Segmentation 对象/场景/实例分割 (Open Model Zoo)
	/// </summary>
	class ObjectSegmentation:public OpenModelInferBase
	{
	public:
		/// <summary>
		/// 对象/场景/实例分割构造函数
		/// </summary>
		/// <param name="output_layer_names">多层的网络输出名称</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		ObjectSegmentation(const std::vector<std::string>& output_layer_names, bool is_debug = true);
		~ObjectSegmentation();

	protected:
		void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;
		void UpdateDebugShow() override;

	private:
		float blending = 0.5;
		cv::Mat resize_img, mask_img;
		std::vector<cv::Vec3b> colors = 
		{
			{ 128, 64,  128 },{ 232, 35,  244 },{ 70,  70,  70 },{ 156, 102, 102 },
			{ 153, 153, 190 },{ 153, 153, 153 },{ 30, 170, 250 },{ 0,  220, 220 },
			{ 35,  142, 107 },{ 152, 251, 152 },{ 180, 130, 70 },{ 60, 20, 220 },
			{ 0, 0, 255 },{ 142, 0, 0 },{ 70, 0, 0 },{ 100, 60, 0 },{ 90, 0,  0 },
			{ 230, 0, 0 },{ 32, 11, 119 },{ 0, 74, 111 },{ 81, 0, 81 }
		};
	};

}