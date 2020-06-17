#pragma once

#include <iostream>
#include <vector>
#include <shared_mutex>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "timer.hpp"
#define _TEST false

#define SHARED_RESERVE_BYTE_SIZE 32

namespace space
{

	class �ʼ�
	{
	public:
		std::map<std::string, std::string> plugin_config =
		{
			//ͨ�ò���ֵ:InferenceEngine::PluginConfigParams::NO/YES

			//������������������CPU�Ͻ���������߳�
			{InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, InferenceEngine::PluginConfigParams::NO},
			//����ÿ���̵߳�CPU�׺���ѡ�������(YES:���̶̹߳������ģ����ʺϾ�̬��׼)(NUMA:��radrad�̶���NUMA�ڵ㣬���ʺ�ʵ�ʵ�)(NO:CPU�����߳��޹̶�)
			{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO},

			//�Ż�CPUִ�����������������ѡ��Ӧ��ֵһ��ʹ�ã�
			//CPU_THROUGHPUT_NUMA:��������������������ӦNUMA��������صķ���
			//CPU_THROUGHPUT_AUTO:�������ٵ�����������ܣ���������˽�Ŀ��������ӵ�ж��ٸ��ںˣ��Լ���ѵ��������Ƕ��٣������������ֲ��ѡ��
			//ָ��������ֵ�����������������
			{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "CPU_THROUGHPUT_NUMA/CPU_THROUGHPUT_AUTO"},


			//�Ż�GPU���ִ�������������,��ѡ��Ӧ��ֵһ��ʹ�ã�
			//GPU_THROUGHPUT_AUTO:�������ٵ�������ĳЩ����¿��ܻ�������ܣ���ѡ������Ϊopencl�������ý�������ʾ���Ӷ�����CPU���ض��������Ž�������
			//������ֵ���������������
			{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "GPU_THROUGHPUT_AUTO"},

			//�������ܼ�����ѡ�������,������ֵһ��ʹ�ã�
			//PluginConfigParams::YES or PluginConfigParams::NO
			{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "YES/NO"},


			//�Ƿ�����̬������
			{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES},
			//�ؼ��ֶ���������Ķ�̬���� �ɽ��ܵ�ֵ��
			//-1 ������������ 
			//>0 ���Ƶ�ֱ��ֵ��Ҫ��������δ�СΪ��Сֵ�����������ƣ�ԭʼ���Σ�
			{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "-1"},

			{InferenceEngine::PluginConfigParams::KEY_DUMP_QUANTIZED_GRAPH_AS_DOT, "?"},
			{InferenceEngine::PluginConfigParams::KEY_DUMP_QUANTIZED_GRAPH_AS_IR, "?"},

			//�������������е��߳�(���߳�),��ѡ��Ӧ��ֵһ��ʹ�ã�
			//PluginConfigParams::YES or PluginConfigParams::NO
			{InferenceEngine::PluginConfigParams::KEY_SINGLE_THREAD, "YES/NO"},

			//ָʾ������������ļ�,��ֵӦ���Ǿ��в���ض����õ��ļ���
			{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, ""},

			//����ת����������Զ������ں�,��ѡ��Ӧ��ֵһ��ʹ�ã�
			//PluginConfigParams::YES or PluginConfigParams::NO (default)
			{InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, ""},

			//���Ʋ����ɻ�ʹ�õ����ܵ���,ѡ��ֵ��
			//TUNING_DISABLED: (default)
			//TUNING_USE_EXISTING:ʹ�õ����ļ��е���������
			//TUNING_CREATE:Ϊ�����ļ��в����ڵĲ���������������
			//TUNING_UPDATE:ִ�зǵ������£�����ɾ����Ч/�����õ�����
			//TUNING_RETUNE:Ϊ���в��������������ݣ���ʹ�Ѿ�����
			//����ֵ TUNING_CREATE �� TUNING_RETUNE������ļ������ڣ��򽫴������ļ�
			{InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_CREATE/TUNING_USE_EXISTING/TUNING_DISABLED/TUNING_UPDATE/TUNING_RETUNE"},
			{InferenceEngine::PluginConfigParams::KEY_TUNING_FILE, ""},

			//������־����
			{InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "LOG_NONE/LOG_ERROR/LOG_WARNING/LOG_INFO/LOG_DEBUG/LOG_TRACE"},

			//setting of required device to execute on values: 
			//�豸ID�� 0 ��ʼ-��һ���豸��1-�ڶ����豸����������
			{InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, ""},


			//Ϊ��ͬ�Ŀ�ִ���������ͬ������첽�������ö�ռģʽ��
			//��ʱ�б�Ҫ���Ⲣ�й���ͬһ�豸�ĳ���Ԥ������
			//���磬��2������CPU�豸������ִ�г���һ��-��Hetero����У���һ��-�ڴ�CPU����С�
			//���ߵĲ���ִ�п��ܻᵼ�³���Ԥ������������ѵ�CPUʹ���ʡ�
			//ͨ������ִ�г���һ����һ����������Ӧ�����Ч�ʸ��ߡ�
			//Ĭ������£����������������ѡ������ΪYES�����ڳ�������������������ѡ������ΪNO��
			//��ע�⣬����ΪYES�����CPU�����ܣ�����Ĵ��ļ��е���һ�����ü���
			{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, ""},

			//�����ڲ�����ͼ��ת��
			//Ӧ�ô��ݸ�LoadNetwork�����������ڲ�ͼԪת������Ӧ��������Ϣ��ֵ�ǲ�����չ����������ļ������ơ�
			//�ļ�<dot_file_name>_init.dot��<dot_file_name>_perf.dot�������
			{InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, ""},
		};
	};

	//�ƶ���ɻص�����
	typedef InferenceEngine::IInferRequest::CompletionCallback	TCompletionCallback;
	typedef std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>		FCompletionCallback;
	typedef std::function<void(InferenceEngine::IInferRequest::Ptr, InferenceEngine::StatusCode)>		FPCompletionCallback;

	/// <summary>
	/// ����ģ���ƶϻ��࣬����ͼ��Ϊ����Դ���첽�ƶϣ�������ڴ�ӳ�乲��
	/// <para>����Ĭ��ֻ������һ�� InferRequest ��������������Ѵ������</para>
	/// </summary>
	class OpenModelInferBase
	{
	public:
		/// <summary>
		/// OpenModelInferBase Params
		/// </summary>
		struct Params
		{
			size_t batch_size = 1;				//�����������С
			size_t infer_count = 1;				//�������ƶ���������
			bool is_mapping_blob = true;		//�Ƿ���ӳ��ָ�������������
			//IE ����������
			std::map<std::string, std::string> ie_config;
			//�����ƶ�����ⲿ�ص�
			FPCompletionCallback request_callback = nullptr;	
			//�첽�ص���ɺ��Զ�������һ���ƶ�
			bool auto_request_infer = true;

			friend std::ostream& operator << (std::ostream& stream, const OpenModelInferBase::Params& params)
			{
				stream << "[BatchSize:" << params.batch_size <<
					" InferCount:" << params.infer_count << std::boolalpha <<
					" SharedBlob:" << params.is_mapping_blob << "]";

				return stream;
			};
		};

		/// <summary>
		/// ����ģ���ƶϹ��캯��
		/// </summary>
		/// <param name="output_layer_names">��Ҫָ������������ƣ��ò��������ж�����������</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		OpenModelInferBase(const std::vector<std::string>& output_layer_names, bool is_debug = true);
		~OpenModelInferBase();

		/// <summary>
		/// ������ز��������� ConfigNetwork ֮ǰ����
		/// </summary>
		/// <param name="params"></param>
		void SetParameters(const Params& params);

		/// <summary>
		/// ��������ģ��
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="model_info">����һ��Э���ʽ��ģ�Ͳ�������ʽ��(ģ������[:����[:Ӳ��]])��ʾ����human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="is_reshape">�Ƿ���������㣬���µ������룬ָ��ԭ֡���ݳߴ硢����ָ��λ����Ϊ����(�����ڴ����ݸ���)�����첽�ƶ�����</param>
		void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape = true);

		/// <summary>
		/// ��������ģ��
		/// </summary>
		/// <param name="ie">ie�ƶ��������</param>
		/// <param name="model_path">aiģ���ļ�·��</param>
		/// <param name="device">�ƶ�ʹ�õ�Ӳ������</param>
		/// <param name="is_reshape">�Ƿ���������㣬���µ������룬ָ��ԭ֡���ݳߴ硢����ָ��λ����Ϊ����(�����ڴ����ݸ���)�����첽�ƶ�����</param>
		void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape = true);

		/// <summary>
		/// �첽�ƶ�������ʵʱ�ύͼ��֡���� is_reshape Ϊ true ʱֻ�����һ�Σ���ε���Ҳ��Ӱ��
		/// </summary>
		/// <param name="frame">ͼ��֡����</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// ��ȡ��ǰ�����ִ������
		/// </summary>
		/// <returns></returns>
		const InferenceEngine::ExecutableNetwork GetExecutableNetwork() const
		{
			return execNetwork;
		}
		/// <summary>
		/// ��ȡ���������ӳ����Ϣ
		/// </summary>
		/// <returns></returns>
		const std::vector<std::pair<std::string, LPVOID>> GetSharedOutputLayers() const
		{
			return output_shared_layers;
		}

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
		/// ������������/���
		/// </summary>
		virtual void SetNetworkIO();

		/// <summary>
		/// �����ƶ������첽�ص�
		/// </summary>
		virtual void SetInferCallback();

		/// <summary>
		/// �����ڴ湲����ָ����������������ݹ��� 
		/// </summary>
		virtual void SetMemoryShared();

		/// <summary>
		/// ����ָ������������ݣ�Ҳ�����Ƕ�������/��������ǵ�����ʾ���
		/// <param> ���Ҫ�����ڲ���������������ʵ�ָú�������������ⲿ�������ûص����� SetCompletionCallback </param>
		/// </summary>
		virtual void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) = 0;

		/// <summary>
		/// ���µ��Բ���ʾ���ú��������������������ʾ��
		/// </summary>
		virtual void UpdateDebugShow() = 0;

		size_t batch_size = 1;				//�������С
		size_t infer_count = 1;				//�������ƶ���������
		bool is_mapping_blob = true;		//�Ƿ���ӳ��ָ�������������
		bool is_configuration = false;		//�Ƿ��Ѿ�������������
		// IE ����������
		std::map<std::string, std::string> ie_config = { };


		bool is_debug = true;			//�Ƿ�������ݵ�����Ϣ
		bool is_reshape = true;			//�Ƿ����µ���ģ������
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
		std::vector<InferenceEngine::InferRequest::Ptr> requestPtrs;

		FPCompletionCallback request_callback = nullptr;

		//���������֡������Ҫ����Ҫ֡�Ŀ��ߣ�ͨ�������͡�����ָ����Ϣ
		cv::Mat frame_ptr;
		//���ڲ������������ݣ����������/�ڴ�ӳ����ͼָ��
		std::vector<std::pair<std::string, LPVOID>> output_shared_layers;
		
		//���ڲ����ļ�ʱ��
		Timer timer;
		//FPS
		size_t fps = 0;
		size_t fps_count = 0;
		std::shared_mutex _fps_mutex;

		//��ȡ���ƶϽ������ʱ��
		double infer_use_time = 0.0f;
		std::shared_mutex _time_mutex;
		std::chrono::steady_clock::time_point t0;
		std::chrono::steady_clock::time_point t1;

	private:

		//��Ҫָ������������ƣ��ò��������ж�����������
		std::vector<std::string> output_layers;
		std::vector<HANDLE> output_shared_handle;	//�����ڴ��ļ����
		//���ڴ����ڴ湲�����Ϣ��ָ�����������������/����洢�ռ��С
		std::vector<std::pair<std::string, size_t>> shared_layers_info;

#if _TEST
		//���ڲ��� Timer ��׼ȷ�Ա���
		double test_use_time = 0.0f;
		std::chrono::steady_clock::time_point t2;
		std::chrono::steady_clock::time_point t3;
#endif

	};

}
