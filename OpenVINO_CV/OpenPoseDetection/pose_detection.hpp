#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

class PoseDetection
{
public:
	struct Result
	{
		int label;
		float confidence;
		std::vector<cv::Point2f> keypoints;
	};
	PoseDetection(const std::string& model, const std::string& device, bool isAsync);
	~PoseDetection();

	/// <summary>
	/// 读取网络
	/// </summary>
	/// <param name="ie"></param>
	/// <returns></returns>
	InferenceEngine::CNNNetwork read(InferenceEngine::Core& ie);

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
	bool getResults(std::vector<Result>& results, cv::Mat &img);

protected:
	// 可执行网络对象
	InferenceEngine::ExecutableNetwork execNetwork;
	// 共享的推断请求对象
	InferenceEngine::InferRequest::Ptr requestPtr;

	InferenceEngine::CNNNetwork cnnNetwork;

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
	int H, W;
	const int POSE_PAIRS[3][20][2] = {
		{   // COCO body
			{ 1,2 },{ 1,5 },{ 2,3 },
			{ 3,4 },{ 5,6 },{ 6,7 },
			{ 1,8 },{ 8,9 },{ 9,10 },
			{ 1,11 },{ 11,12 },{ 12,13 },
			{ 1,0 },{ 0,14 },
			{ 14,16 },{ 0,15 },{ 15,17 }
		},
		{   // MPI body
			{ 0,1 },{ 1,2 },{ 2,3 },
			{ 3,4 },{ 1,5 },{ 5,6 },
			{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
			{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
		},
		{   // hand
			{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
			{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
			{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
			{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
			{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
		} };
};

