#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

class PersonDetection
{
	
public:
	struct Result
	{
		int label;
		float confidence;
		cv::Rect location;
	};

	/// <summary>
	/// 构造函数
	/// </summary>
	/// <param name="model">模型名称或路径？</param>
	/// <param name="device"></param>
	/// <param name="isAsync"></param>
	/// <returns></returns>
	PersonDetection(const std::string& model, const std::string& device, bool isAsync);
	~PersonDetection();

	/// <summary>
		/// 可执行网络
		/// </summary>
		/// <returns></returns>
	InferenceEngine::ExecutableNetwork* operator ->();

	/// <summary>
	/// 读取网络
	/// </summary>
	/// <param name="ie"></param>
	/// <returns></returns>
	InferenceEngine::CNNNetwork read(InferenceEngine::Core &ie);

	/// <summary>
	/// 帧入队列
	/// </summary>
	/// <param name="frame"></param>
	void enqueue(const cv::Mat& frame);

	/// <summary>
	/// 提交推断请求
	/// </summary>
	void request();

	/// <summary>
	/// 异步等待
	/// </summary>
	void wait();

	/// <summary>
	/// 获取结果
	/// </summary>
	/// <returns></returns>
	bool getResults(std::vector<Result> &results);

protected:
	// 可执行网络对象
	InferenceEngine::ExecutableNetwork execNetwork;
	// 共享的推断请求对象
	InferenceEngine::InferRequest::Ptr requestPtr;
	
	// 是否异步推断
	bool isAsync;		
	// AI网络模型
	std::string model;	
	// AI网络模型加载到指定的硬件
	std::string device;

	//网络输入名称
	std::string network_input_name;
	uint16_t network_input_width;		//网络输入宽要求
	uint16_t network_input_height;		//网络输入高要求
	//网络输出名称
	std::string network_output_name;
	uint16_t network_output_max_count;
	uint16_t network_output_object_size;

	//输入图像的宽
	uint16_t input_frame_width;
	//输入图像的高
	uint16_t input_frame_height;

	uint8_t enquedFrames;		//队列数量
	bool results_fetched;		

	float confidenceThreshold;	//置信度阈值

private:
	InferenceEngine::CNNNetwork cnnNetwork;

	
};

