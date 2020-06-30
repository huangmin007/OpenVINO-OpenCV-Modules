#pragma once

#include <iostream>
#include <Windows.h>
#include <shared_mutex>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include <cmdline.h>
#include <timer.hpp>
#include <static_functions.hpp>

namespace space
{
	/// <summary>
	/// Object Detection ������ (Open Model Zoo)
	/// </summary>
	class ObjectDetection
	{
	public:
		// ���������ò���
		struct Params
		{
		public:
			InferenceEngine::Core ie;

			std::string model;
			std::string device = "CPU";
			std::map<std::string, std::string> ie_config;

			bool is_async = true;		//�첽�ƶ�
			bool is_reshape = true;		//��������
			size_t batch_count = 1;		//��� batch ����
			size_t infer_count = 1;		//�������ƶ�����

			bool is_mapping_blob = true;					//�Ƿ���ӳ��ָ�������������
			bool auto_request_infer = true;					//������첽�ص�������ɺ��Զ�������һ���ƶ�
			std::vector<std::string> output_layers;			//��Ҫ��������Ĳ�

			float confidence = 0.5f;						//��������ֵ
			std::vector<std::string> labels;				//�����ǩ

			void Parse(const cmdline::parser& args)
			{
				std::map<std::string, std::string> model_info = ParseArgsForModel(args.get<std::string>("model"));

				model = model_info["path"];
				device = model_info["device"];
				is_reshape = args.get<bool>("reshape");
				confidence = args.get<float>("conf");
				output_layers = SplitString(args.get<std::string>("output_layers"), ':');
			}

			bool Check()
			{
				if (model.empty())
				{
					throw std::invalid_argument("����ģ�Ͳ���Ϊ��.");
					return false;
				}
				if (batch_count < 1)
				{
					throw std::invalid_argument("Batch Count ����С�� 1 .");
					return false;
				}
				if (infer_count < 1)
				{
					throw std::invalid_argument("Infer Request ��������С�� 1 .");
					return false;
				}

				return true;
			}
			
			void UpdateIEConfig()
			{
				if (batch_count > 1)
					ie_config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
				
				if (infer_count > 1)
					ie_config[InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD] = InferenceEngine::PluginConfigParams::NO;
			}

			explicit Params() {};
			explicit Params(InferenceEngine::Core ie) :ie(ie) {};

			friend std::ostream& operator << (std::ostream& stream, const Params& params)
			{
				stream << std::boolalpha << "[Params" <<
					" Async:" << params.is_async <<
					" Reshape:" << params.is_reshape <<
					" BatchCount:" << params.batch_count <<
					" InferCount:" << params.infer_count <<
					" MappingBlob:" << params.is_mapping_blob <<
					//" OutputLayers:" << params.output_layers << 
					"]";

				return stream;
			};
		};
		//������������
		struct Result
		{
			int id;								// batch ID
			int label;							// Ԥ����� ID
			float confidence;					// Ԥ����������ֵ
			cv::Rect location;					// �߽��ǵ�����

			friend std::ostream& operator << (std::ostream& stream, const Result& result)
			{
				stream << "[Result ID:" << result.id <<
					" Label:" << result.label <<
					" Confidence:" << result.confidence << 
					" Rect:" << result.location << "]";

				return stream;
			};
		};

		/// <summary>
		/// �����⹹�캯��
		/// </summary>
		/// <param name="output_layers_name">���������������</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		ObjectDetection(bool is_show = true);
		~ObjectDetection();

		/// <summary>
		/// ��������
		/// </summary>
		/// <param name="params"></param>
		void ConfigNetwork(const Params& params);

		/// <summary>
		/// �첽�ƶ�������ʵʱ�ύͼ��֡���� is_reshape Ϊ true ʱֻ�����һ�Σ���ε���Ҳ��Ӱ��
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

		static void DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string>& labels);

	protected:
		/// <summary>
		/// ��������IO
		/// </summary>
		virtual void SetNetworkIO();
		
		/// <summary>
		/// �����ƶ������첽�ص�
		/// </summary>
		virtual void SetInferCallback();

		/// <summary>
		/// ����ָ������������ݣ�Ҳ�����Ƕ�������/��������ǵ�����ʾ���
		/// <param> ���Ҫ�����ڲ���������������ʵ�ָú�������������ⲿ�������ûص����� SetCompletionCallback </param>
		/// </summary>
		virtual void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status);

		/// <summary>
		/// ���µ��Բ���ʾ���ú��������������������ʾ��
		/// </summary>
		virtual void UpdateDebugShow();

		Params params;
		bool is_configuration = false;		//�Ƿ��Ѿ�������������
		std::vector<std::tuple<std::string, HANDLE, LPVOID>> shared_layers;	//�ڴ湲�������

		//���������֡������Ҫ����Ҫ֡�Ŀ��ߣ�ͨ�������͡�����ָ����Ϣ��ǳ��������
		cv::Mat frame_ptr;

		bool is_show = true;			//�Ƿ�������ݵ�����Ϣ
		cv::Mat show_frame;				//������Ⱦ��ʾ֡����ȿ�������
		std::stringstream title;		//����������ڱ���


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
		//[Nx5][N]
		void PasingO2DNx5(InferenceEngine::IInferRequest::Ptr request);
		//[1x1xNx7] FP32
		void PasingO1D1x1xNx7(InferenceEngine::IInferRequest::Ptr request);

		float scaling = 1.2;
		std::vector<ObjectDetection::Result> results;
	};

};
