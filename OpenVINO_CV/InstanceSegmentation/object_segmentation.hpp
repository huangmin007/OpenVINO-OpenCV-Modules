#pragma once

#include <iostream>
#include <Windows.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


namespace space
{
	/// <summary>
	/// Object/Screen/Instance Segmentation ����/����/ʵ���ָ� (Open Model Zoo)
	/// </summary>
	class ObjectSegmentation
	{
	public:
		/// <summary>
		/// ����/����/ʵ���ָ�캯��
		/// </summary>
		/// <param name="isDebug">�Ƿ����е������״̬</param>
		ObjectSegmentation(bool isDebug = false);
		/// <summary>
		/// ����/����/ʵ���ָ�캯��
		/// </summary>
		/// <param name="outputLayerNames">���������������</param>
		/// <param name="isDebug"></param>
		/// <returns></returns>
		ObjectSegmentation(const std::vector<std::string>& outputLayerNames, bool isDebug = false);
		~ObjectSegmentation();

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="modelInfo">����һ��Э���ʽ��ģ�Ͳ�������ʽ��(ģ������[:����[:Ӳ��]])��ʾ����human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="isAsync">�Ƿ�ִ���첽�ƶ�</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string& modelInfo, bool isAsync = true);

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="modelPath">aiģ���ļ�·��</param>
		/// <param name="device">�ƶ�ʹ�õ�Ӳ������</param>
		/// <param name="isAsync">�Ƿ�ִ���첽�ƶ�</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string& modelPath, const std::string& device, bool isAsync = true);

		/// <summary>
		/// �����ƶ�
		/// </summary>
		/// <param name="frame">֡ͼ��</param>
		/// <param name="frame_idx">batch index, ֡id���������ֵ�ǻ����� batch ֵ��Ĭ�� 0 ������ batch ֵ</param>
		void start(const cv::Mat& frame);

		double getUseTime() const;

	protected:
		bool is_debug;		//�Ƿ�������ݵ�����Ϣ
		bool is_async;				//�Ƿ��첽�ƶ�

		//cnn �������
		InferenceEngine::CNNNetwork cnnNetwork;
		//��ִ���������
		InferenceEngine::ExecutableNetwork execNetwork;

		//�������ݼ���Ϣ
		InferenceEngine::InputsDataMap inputsInfo;
		//����Shapes
		InferenceEngine::ICNNNetwork::InputShapes inputShapes;
		//������ݼ���Ϣ
		InferenceEngine::OutputsDataMap outputsInfo;
		//���������ĵ�һ�� �����ߴ��С
		InferenceEngine::SizeVector outputSize;

		// �ƶ��������
		InferenceEngine::InferRequest::Ptr requestPtr;

		uint16_t frame_width;		//����֡ͼ��Ŀ�
		uint16_t frame_height;		//����֡ͼ��ĸ�

		std::string input_name;		//������������
		std::string output_name;	//�����������

		void updateDisplay();

	private:
		//ָ������������ƣ�һ�����������Ҫָ������
		std::vector<std::string> output_layer_names;

		std::vector<LPVOID> pBuffers;
		std::vector<HANDLE> pMapFiles;
		//������������������Ϣ
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