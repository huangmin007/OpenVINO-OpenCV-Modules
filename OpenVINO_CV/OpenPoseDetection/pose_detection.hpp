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
	/// ��ȡ����
	/// </summary>
	/// <param name="ie"></param>
	/// <returns></returns>
	InferenceEngine::CNNNetwork read(InferenceEngine::Core& ie);

	/// <summary>
	/// ֡�����
	/// </summary>
	/// <param name="frame"></param>
	void enqueue(const cv::Mat& frame);

	/// <summary>
	/// �ύ�ƶ�����
	/// </summary>
	void request();

	/// <summary>
	/// �첽�ȴ�
	/// </summary>
	void wait();

	/// <summary>
	/// ��ȡ���
	/// </summary>
	/// <returns></returns>
	bool getResults(std::vector<Result>& results, cv::Mat &img);

protected:
	// ��ִ���������
	InferenceEngine::ExecutableNetwork execNetwork;
	// ������ƶ��������
	InferenceEngine::InferRequest::Ptr requestPtr;

	InferenceEngine::CNNNetwork cnnNetwork;

	// �Ƿ��첽�ƶ�
	bool isAsync;
	// AI����ģ��
	std::string model;
	// AI����ģ�ͼ��ص�ָ����Ӳ��
	std::string device;

	//������������
	std::string network_input_name;
	uint16_t network_input_width;		//���������Ҫ��
	uint16_t network_input_height;		//���������Ҫ��
	//�����������
	std::string network_output_name;
	uint16_t network_output_max_count;
	uint16_t network_output_object_size;

	//����ͼ��Ŀ�
	uint16_t input_frame_width;
	//����ͼ��ĸ�
	uint16_t input_frame_height;

	uint8_t enquedFrames;		//��������
	bool results_fetched;

	float confidenceThreshold;	//���Ŷ���ֵ
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

