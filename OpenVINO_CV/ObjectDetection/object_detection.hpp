#pragma once

#include <iostream>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


namespace space
{

#pragma region ObjectDetection
	/// <summary>
	/// Object Detection ������ (Open Model Zoo)
	/// </summary>
	class ObjectDetection
	{
	public:
		//������������
		struct Result
		{
			int id;								// ID of the image in the batch
			int label;							// predicted class ID
			float confidence;					// confidence for the predicted class
			cv::Rect location;					// coordinates of the bounding box corner
		};

		/// <summary>
		/// �����⹹�캯��
		/// </summary>
		/// <param name="isDebug">�Ƿ����е������״̬</param>
		ObjectDetection(bool isDebug = false);
		/// <summary>
		/// �����⹹�캯��
		/// </summary>
		/// <param name="outputLayerNames">���������������</param>
		/// <param name="isDebug"></param>
		/// <returns></returns>
		ObjectDetection(const std::vector<std::string> &outputLayerNames, bool isDebug = false);
		~ObjectDetection();

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="modelInfo">����һ��Э���ʽ��ģ�Ͳ�������ʽ��(ģ������[:����[:Ӳ��]])��ʾ����human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="isAsync">�Ƿ�ִ���첽�ƶ�</param>
		/// <param name="confidenceThreshold">����������ֵ</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string& modelInfo, bool isAsync = true, float confidenceThreshold = 0.5f);

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="modelPath">aiģ���ļ�·��</param>
		/// <param name="device">�ƶ�ʹ�õ�Ӳ������</param>
		/// <param name="isAsync">�Ƿ�ִ���첽�ƶ�</param>
		/// <param name="confidenceThreshold">����������ֵ</param>
		/// <returns></returns>
		bool config(InferenceEngine::Core& ie, const std::string & modelPath, const std::string &device, bool isAsync = true, float confidenceThreshold = 0.5f);
		
		/// <summary>
		/// �����ƶ϶�����
		/// </summary>
		/// <param name="frame">֡ͼ��</param>
		/// <param name="frame_idx">batch index, ֡id���������ֵ�ǻ����� batch ֵ��Ĭ�� 0 ������ batch ֵ</param>
		void request(const cv::Mat& frame, int frame_idx = 0);

		/// <summary>
		/// ��ȡ�������
		/// </summary>
		/// <param name="results"></param>
		/// <returns></returns>
		virtual bool getResults(std::vector<ObjectDetection::Result>& results);

		/// <summary>
		/// ��ִ������
		/// </summary>
		/// <returns></returns>
		//InferenceEngine::ExecutableNetwork* operator ->();

		void start();

		void updateDebugShow(cv::Mat &frame);

	protected:
		
		bool is_debug;		//�Ƿ�������ݵ�����Ϣ
		bool is_async;				//�Ƿ��첽�ƶ�
		float confidence_threshold;	//����������ֵ

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
		InferenceEngine::SizeVector outputSizeVector;
		
		// ��ǰ֡�ƶ��������
		InferenceEngine::InferRequest::Ptr requestPtr;
		// ��һ֡�ƶ��������
		//InferenceEngine::InferRequest::Ptr requestPtrNext;

		uint16_t frame_width;		//����֡ͼ��Ŀ�
		uint16_t frame_height;		//����֡ͼ��ĸ�

		std::string input_name;		//������������
		uint16_t input_batch;		//��������batchҪ��
		uint16_t input_channels;	//��������ͨ��Ҫ��
		uint16_t input_width;		//���������Ҫ��
		uint16_t input_height;		//���������Ҫ��

		std::string output_name;	//�����������
		uint16_t output_max_count;	//�����������������
		uint16_t output_object_size;//������������С

		//�Ƿ��Ѿ���ȡ�����ǣ�����Ѿ���ȡ���轫�ñ�������Ϊ false
		//�ñ�ǵ������ǣ�����������ظ���ȡ
		bool results_flags;		
		bool has_results = false;

		/// <summary>
		/// cv::Mat U8 to IE Blob
		/// </summary>
		/// <param name="frame">ԭ֡ͼ������</param>
		/// <param name="blob"></param>
		/// <param name="batchIndex"></param>
		void matU8ToBlob(const cv::Mat& frame, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);

		

	private:
		std::stringstream debug_title;

		double total_use_time = 0.0f;
		std::chrono::steady_clock::time_point t0;
		std::chrono::steady_clock::time_point t1;

		//ָ������������ƣ�һ�����������Ҫָ������
		std::vector<std::string> output_layer_names;

		std::vector<LPVOID> pBuffers;
		std::vector<HANDLE> pMapFiles;
		//������������������Ϣ
		std::vector<std::pair<std::string, size_t>> shared_layer_infos;
	};

#pragma endregion
}
