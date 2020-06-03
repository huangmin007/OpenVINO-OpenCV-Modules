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
	//typedef InferenceEngine::IInferRequest::CompletionCallback	TCompletionCallback;
	//typedef std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>		FCompletionCallback;

	/// <summary>
	/// ����ģ���ƶϻ��࣬����ͼ��Ϊ����Դ���첽�ƶϣ�������ڴ�ӳ�乲��
	/// <para>����Ĭ��ֻ������һ�� InferRequest ��������������Ѵ������</para>
	/// </summary>
	class OpenModelInferBase
	{
	public:
		/// <summary>
		/// ����ģ���ƶϹ��캯��
		/// </summary>
		/// <param name="output_layer_names">��Ҫָ������������ƣ��ò��������ж�����������</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		OpenModelInferBase(const std::vector<std::string> &output_layer_names, bool is_debug = true);
		~OpenModelInferBase();

		/// <summary>
		/// ��������ģ��
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="model_info">����һ��Э���ʽ��ģ�Ͳ�������ʽ��(ģ������[:����[:Ӳ��]])��ʾ����human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="is_reshape">�Ƿ���������㣬���µ������룬ָ��ԭ֡���ݳߴ硢����ָ��λ����Ϊ���룬���첽�ƶ�����</param>
		void Configuration(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape = true);

		/// <summary>
		/// ��������ģ��
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="model_path">aiģ���ļ�·��</param>
		/// <param name="device">�ƶ�ʹ�õ�Ӳ������</param>
		/// <param name="is_reshape">�Ƿ���������㣬���µ������룬ָ��ԭ֡���ݳߴ硢����ָ��λ����Ϊ���룬���첽�ƶ�����</param>
		void Configuration(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape = true);

		/// <summary>
		/// �첽�ƶ�������ʵʱ�ύͼ��֡���� ResizeInput �෴
		/// <para>��Ҫʵʱ���øú����������ܵ��� ResizeInput ����</para>
		/// </summary>
		/// <param name="frame">ͼ��֡����</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// ��ȡ��һ֡�ƶ����ú�ʱ ms
		/// </summary>
		/// <returns></returns>
		double GetInferUseTime()
		{
			std::shared_lock<std::shared_mutex> lock(_time_mutex);
			return infer_use_time;
		}

		/// <summary>
		/// ��ȡÿ���ƶϽ��������
		/// </summary>
		/// <returns></returns>
		size_t GetFPS()
		{
			std::shared_lock<std::shared_mutex> lock(_fps_mutex);
			return fps;
		}

	protected:

		/// <summary>
		/// ������������/������������������������õ�
		/// </summary>
		virtual void ConfigNetworkIO();

		/// <summary>
		/// ������������ݣ������Ƕ�����������ǵ�����ʾ���
		/// </summary>
		virtual void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
		{
			if (is_debug) UpdateDebugShow();
		}

		/// <summary>
		/// ���µ��Բ���ʾ���ú��������������������ʾ��
		/// </summary>
		virtual void UpdateDebugShow() = 0;

		bool is_debug;		//�Ƿ�������ݵ�����Ϣ
		bool is_reshape = false;		//�Ƿ����µ���ģ������
		std::stringstream debug_title;	//����������ڱ���

		//CNN �������
		InferenceEngine::CNNNetwork cnnNetwork;
		//��ִ���������
		InferenceEngine::ExecutableNetwork execNetwork;

		//�����������Ϣ
		InferenceEngine::InputsDataMap inputsInfo;
		//�����������Ϣ
		InferenceEngine::OutputsDataMap outputsInfo;
		// �ƶ��������
		InferenceEngine::InferRequest::Ptr requestPtr;


		//���������֡������Ҫ����Ҫ֡�Ŀ��ߣ�ͨ�������͡�����ָ����Ϣ
		cv::Mat frame_ptr;
		//������ڴ��ļ����
		std::vector<HANDLE> shared_output_handle;
		//���ڲ������������ݣ����������/�ڴ�ӳ����ͼָ��
		std::vector<std::pair<std::string, LPVOID>> shared_output_layers;
		
	private:
		/// <summary>
		/// ��������ڴ湲����ӳ�䵽ָ�������������
		/// <para>�Ƿ�ӳ�䵽��������㣬���������</para>
		/// </summary>
		void CreateOutputShared();
		/// <summary>
		/// �ڴ����ӳ��
		/// </summary>
		/// <param name="shared"></param>
		void MemoryOutputMapping(const std::pair<std::string, LPVOID>& shared);

		std::string device;
		InferenceEngine::Core ie;

		//��Ҫָ������������ƣ��ò��������ж�����������
		std::vector<std::string> output_layers;
		//���ڴ�����������Ϣ�����������/�ռ��С
		std::vector<std::pair<std::string, size_t>> shared_layers_info;

		//��ȡ���ƶϽ������ʱ��
		double infer_use_time = 0.0f;
		std::shared_mutex _time_mutex;

		Timer timer;
		size_t fps = 0;
		size_t fps_count = 0;
		std::shared_mutex _fps_mutex;

		std::chrono::steady_clock::time_point t0;
		std::chrono::steady_clock::time_point t1;

		bool is_configuration = false;	//�Ƿ��Ѿ����ù�
	};

}
