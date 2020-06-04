#pragma once

#include <iostream>
#include <vector>
#include <shared_mutex>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "timer.hpp"

namespace space
{
	typedef InferenceEngine::IInferRequest::CompletionCallback	TCompletionCallback;
	typedef std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>	FCompletionCallback;

	/// <summary>
	/// 开放模型推断基类，基于图像为输入源，异步推断，输出层内存映射共享
	/// <para>自定义 InferRequest 数量对象，跟据其硬件性能</para>
	/// </summary>
	class OpenModelMultiInferBase
	{
	public:
		/// <summary>
		/// 
		/// </summary>
		/// <param name="output_layer_names"></param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		OpenModelMultiInferBase(const std::vector<std::string>& output_layer_names, bool is_debug = true);
		~OpenModelMultiInferBase();

		/// <summary>
		/// 配置网络模型
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="model_info">具有一定协议格式的模型参数，格式：(模型名称[:精度[:硬件]])，示例：human-pose-estimation-0001:FP32:CPU</param>
		void Configuration(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape = true, uint8_t request_count = 2);

		/// <summary>
		/// 配置网络模型
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="model_path">ai模型文件路径</param>
		/// <param name="device">推断使用的硬件类型</param>
		void Configuration(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape = true, uint8_t request_count = 2);

		/// <summary>
		/// 异步推断请求，需实时提交图像帧
		/// </summary>
		/// <param name="frame">图像帧对象</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// 获取平均推断所用耗时 ms
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

	protected:
		/// <summary>
		/// 配置网络输入/输出，是用于子类做其它配置的
		/// </summary>
		virtual void ConfigNetworkIO();

		/// <summary>
		/// 创建输出内存共享，并映射到指定的网络输出层
		/// <para>是否映射到网络输出层，由子类决定</para>
		/// </summary>
		virtual void CreateMemoryShared();

		/// <summary>
		/// 解析输出层数据，可能是二次输出，或是调试显示输出
		/// </summary>
		virtual void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
		{
			if (is_debug) UpdateDebugShow();
		}
		/// <summary>
		/// 更新调试并显示，该函数是用于子类做输出显示的
		/// </summary>
		virtual void UpdateDebugShow() = 0;

		bool is_debug;		//是否输出部份调试信息
		std::stringstream debug_title;	//调试输出窗口标题

		//CNN 网络对象
		InferenceEngine::CNNNetwork cnnNetwork;
		//可执行网络对象
		InferenceEngine::ExecutableNetwork execNetwork;

		//输入层数据信息
		InferenceEngine::InputsDataMap inputsInfo;
		//输出层数据信息
		InferenceEngine::OutputsDataMap outputsInfo;
		//推断请求对象
		std::vector<InferenceEngine::InferRequest::Ptr> requestPtrs;
		
		//int findex = 0;
		//std::vector<cv::Mat> frames;


		//引用输入的帧对象，主要是需要帧的宽，高，通道，类型、数据指针信息
		cv::Mat frame_ptr;
		//共享层内存文件句柄
		std::vector<HANDLE> shared_output_handle;
		//用于操作共享层的数据，共享层名称/内存映射视图指针
		std::vector<std::pair<std::string, LPVOID>> shared_output_layers;

	private:
		

		//需要指定的输出层名称，该参数用于有多层输出的网络
		std::vector<std::string> output_layers;
		//用于创建共享层的信息，共享层名称/空间大小
		std::vector<std::pair<std::string, size_t>> shared_layers_info;

		Timer timer;

		size_t fps = 0;		
		size_t fps_count = 0;
		std::shared_mutex _fps_mutex;

		//获取到推断结果所耗时间
		double infer_use_time = 0.0f;
		std::shared_mutex _time_mutex;

		bool is_reshape = false;		//是否重新调整模型输入
		bool is_configuration = false;	//是否已经配置过
	};
}

