#pragma once

#include <iostream>
#include <Windows.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


namespace space
{
	/// <summary>
	/// Object/Screen/Instance Segmentation 对象/场景/实例分割 (Open Model Zoo)
	/// </summary>
	class ObjectSegmentation
	{
	public:
		/// <summary>
		/// 对象/场景/实例分割构造函数
		/// </summary>
		/// <param name="isDebug">是否运行调试输出状态</param>
		ObjectSegmentation(bool isDebug = false);
		/// <summary>
		/// 对象/场景/实例分割构造函数
		/// </summary>
		/// <param name="outputLayerNames">多层的网络输出名称</param>
		/// <param name="isDebug"></param>
		/// <returns></returns>
		ObjectSegmentation(const std::vector<std::string>& outputLayerNames, bool isDebug = false);
		~ObjectSegmentation();

		/// <summary>
		/// 配置网络
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="modelInfo">具有一定协议格式的模型参数，格式：(模型名称[:精度[:硬件]])，示例：human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="isAsync">是否执行异步推断</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string& modelInfo, bool isAsync = true);

		/// <summary>
		/// 配置网络
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="modelPath">ai模型文件路径</param>
		/// <param name="device">推断使用的硬件类型</param>
		/// <param name="isAsync">是否执行异步推断</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string& modelPath, const std::string& device, bool isAsync = true);

		/// <summary>
		/// 启动推断
		/// </summary>
		/// <param name="frame">帧图像</param>
		/// <param name="frame_idx">batch index, 帧id索引，这个值是会设置 batch 值，默认 0 不设置 batch 值</param>
		void start(const cv::Mat& frame);

		double getUseTime() const;

	protected:
		bool is_debug;		//是否输出部份调试信息
		bool is_async;				//是否异步推断

		//cnn 网络对象
		InferenceEngine::CNNNetwork cnnNetwork;
		//可执行网络对象
		InferenceEngine::ExecutableNetwork execNetwork;

		//输入数据集信息
		InferenceEngine::InputsDataMap inputsInfo;
		//输入Shapes
		InferenceEngine::ICNNNetwork::InputShapes inputShapes;
		//输出数据集信息
		InferenceEngine::OutputsDataMap outputsInfo;
		//网络输出层的第一层 张量尺寸大小
		InferenceEngine::SizeVector outputSize;

		// 推断请求对象
		InferenceEngine::InferRequest::Ptr requestPtr;

		uint16_t frame_width;		//输入帧图像的宽
		uint16_t frame_height;		//输入帧图像的高

		std::string input_name;		//网络输入名称
		std::string output_name;	//网络输出名称

		void updateDisplay();

	private:
		//指定的输出层名称，一般多层输出才需要指定名称
		std::vector<std::string> output_layer_names;

		std::vector<LPVOID> pBuffers;
		std::vector<HANDLE> pMapFiles;
		//共享网络层输出数据信息
		std::vector<std::pair<std::string, size_t>> shared_layer_infos;

		cv::Mat resize_img, mask_img;
		std::vector<cv::Vec3b> colors = 
		{
			{ 128, 64,  128 },{ 232, 35,  244 },{ 70,  70,  70 },{ 156, 102, 102 },
			{ 153, 153, 190 },{ 153, 153, 153 },{ 30, 170, 250 },{ 0,  220, 220 },
			{ 35,  142, 107 },{ 152, 251, 152 },{ 180, 130, 70 },{ 60, 20, 220 },
			{ 0, 0, 255 },{ 142, 0, 0 },{ 70, 0, 0 },{ 100, 60, 0 },{ 90, 0,  0 },
			{ 230, 0, 0 },{ 32, 11, 119 },{ 0, 74, 111 },{ 81, 0, 81 }
		};

		double total_use_time = 0.0f;
		std::chrono::steady_clock::time_point t0;
		std::chrono::steady_clock::time_point t1;

	};

}