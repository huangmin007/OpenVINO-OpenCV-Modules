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
	/// ���캯��
	/// </summary>
	/// <param name="model">ģ�����ƻ�·����</param>
	/// <param name="device"></param>
	/// <param name="isAsync"></param>
	/// <returns></returns>
	PersonDetection(const std::string& model, const std::string& device, bool isAsync);
	~PersonDetection();

	/// <summary>
		/// ��ִ������
		/// </summary>
		/// <returns></returns>
	InferenceEngine::ExecutableNetwork* operator ->();

	/// <summary>
	/// ��ȡ����
	/// </summary>
	/// <param name="ie"></param>
	/// <returns></returns>
	InferenceEngine::CNNNetwork read(InferenceEngine::Core &ie);

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
	bool getResults(std::vector<Result> &results);

protected:
	// ��ִ���������
	InferenceEngine::ExecutableNetwork execNetwork;
	// ������ƶ��������
	InferenceEngine::InferRequest::Ptr requestPtr;
	
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
	InferenceEngine::CNNNetwork cnnNetwork;

	
};

