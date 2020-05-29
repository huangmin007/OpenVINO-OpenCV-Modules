#pragma once

#include <iostream>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


namespace space
{

#pragma region ObjectDetection
	/// <summary>
	/// ObjectDetection 对象检测 (Open Model Zoo)
	/// </summary>
	class ObjectDetection
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
		/// <param name="isDebug">是否运行调试输出状态</param>
		ObjectDetection(bool isDebug = false);
		/// <summary>
		/// 对象检测构造函数
		/// </summary>
		/// <param name="outputLayerNames">多层的网络输出名称</param>
		/// <param name="isDebug"></param>
		/// <returns></returns>
		ObjectDetection(const std::vector<std::string> &outputLayerNames, bool isDebug = false);
		~ObjectDetection();

		/// <summary>
		/// 配置网络
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="modelInfo">具有一定协议格式的模型参数，格式：(模型名称[:精度[:硬件]])，示例：human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="isAsync">是否执行异步推断</param>
		/// <param name="confidenceThreshold">数据信任阈值</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string& modelInfo, bool isAsync = true, float confidenceThreshold = 0.5f);

		/// <summary>
		/// 配置网络
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="modelPath">ai模型文件路径</param>
		/// <param name="device">推断使用的硬件类型</param>
		/// <param name="isAsync">是否执行异步推断</param>
		/// <param name="confidenceThreshold">数据信任阈值</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string & modelPath, const std::string &device, bool isAsync = true, float confidenceThreshold = 0.5f);
		
		/// <summary>
		/// 请求推断对象检测
		/// </summary>
		/// <param name="frame">帧图像</param>
		/// <param name="frame_idx">batch index, 帧id索引，这个值是会设置 batch 值，默认 0 不设置 batch 值</param>
		void request(const cv::Mat& frame, int frame_idx = 0);

		/// <summary>
		/// 获取结果类型
		/// </summary>
		/// <param name="results"></param>
		/// <returns></returns>
		virtual bool getResults(std::vector<ObjectDetection::Result>& results);

		/// <summary>
		/// 可执行网络
		/// </summary>
		/// <returns></returns>
		//InferenceEngine::ExecutableNetwork* operator ->();

	protected:
		
		bool is_debug;		//是否输出部份调试信息
		bool is_async;				// 是否异步推断
		float confidence_threshold;	//数据信任阈值

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
		InferenceEngine::SizeVector firstOutputSize;
		
		// 当前帧推断请求对象
		InferenceEngine::InferRequest::Ptr requestPtrCurr;
		// 下一帧推断请求对象
		InferenceEngine::InferRequest::Ptr requestPtrNext;

		uint16_t frame_count;		//输入帧队列数量
		uint16_t frame_width;		//输入帧图像的宽
		uint16_t frame_height;		//输入帧图像的高

		std::string input_name;		//网络输入名称
		uint16_t input_batch;		//网络输入batch要求
		uint16_t input_channels;	//网络输入通道要求
		uint16_t input_width;		//网络输入宽要求
		uint16_t input_height;		//网络输入高要求

		std::string output_name;	//网络输出名称
		uint16_t output_max_count;	//网络输出结果最大数量
		uint16_t output_object_size;//网络输出对象大小

		//是否已经获取结果标记，如果已经获取，需将该变量设置为 false
		//该标记的作用是，不做结果的重复提取
		bool results_flags;		

		/// <summary>
		/// Network I/O Config
		/// </summary>
		virtual void networkIOConfig();

		/// <summary>
		/// cv::Mat U8 to IE Blob
		/// </summary>
		/// <param name="frame">原帧图像数据</param>
		/// <param name="blob"></param>
		/// <param name="batchIndex"></param>
		void matU8ToBlob(const cv::Mat& frame, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);

		/// <summary>
		/// 输入图像帧 to Blob 处理
		/// </summary>
		/// <param name="frame">原帧图像数据</param>
		/// <param name="blob"></param>
		/// <param name="batchIndex"></param>
		virtual void inputFrameToBlob(const cv::Mat& frame, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);

		/// <summary>
		/// 是否存在结果
		/// </summary>
		/// <returns></returns>
		bool hasResults();

	private:
		std::stringstream debug_title;

		//HANDLE pMapFile;
		//PVOID  pBuffer;

		//指定的输出层名称，一般多层输出才需要指定名称
		std::vector<std::string> output_layer_names;

		std::vector<PVOID> pBuffers;
		std::vector<HANDLE> pMapFiles;
		//共享网络层输出数据信息
		std::vector<std::pair<std::string, size_t>> shared_layer_infos;
	};

#pragma endregion
}
