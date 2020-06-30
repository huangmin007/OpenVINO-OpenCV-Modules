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
	/// Object Detection 对象检测 (Open Model Zoo)
	/// </summary>
	class ObjectDetection
	{
	public:
		// 对象检测配置参数
		struct Params
		{
		public:
			InferenceEngine::Core ie;

			std::string model;
			std::string device = "CPU";
			std::map<std::string, std::string> ie_config;

			bool is_async = true;		//异步推断
			bool is_reshape = true;		//重塑输入
			size_t batch_count = 1;		//最大 batch 数量
			size_t infer_count = 1;		//创建的推断数量

			bool is_mapping_blob = true;					//是否共享映射指定的输出层数据
			bool auto_request_infer = true;					//如果是异步回调，调完成后，自动请求下一次推断
			std::vector<std::string> output_layers;			//需要处理输出的层

			float confidence = 0.5f;						//可置信阈值
			std::vector<std::string> labels;				//对象标签

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
					throw std::invalid_argument("网络模型不能为空.");
					return false;
				}
				if (batch_count < 1)
				{
					throw std::invalid_argument("Batch Count 不能小于 1 .");
					return false;
				}
				if (infer_count < 1)
				{
					throw std::invalid_argument("Infer Request 数量不能小于 1 .");
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
		//对象检测结果数据
		struct Result
		{
			int id;								// batch ID
			int label;							// 预测分类 ID
			float confidence;					// 预测分类的置信值
			cv::Rect location;					// 边界框角的坐标

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
		/// 对象检测构造函数
		/// </summary>
		/// <param name="output_layers_name">多层的网络输出名称</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		ObjectDetection(bool is_show = true);
		~ObjectDetection();

		/// <summary>
		/// 配置网络
		/// </summary>
		/// <param name="params"></param>
		void ConfigNetwork(const Params& params);

		/// <summary>
		/// 异步推断请求，需实时提交图像帧；当 is_reshape 为 true 时只需调用一次，多次调用也不影响
		/// </summary>
		/// <param name="frame">图像帧对象</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// 获取上一帧推断所用耗时 ms
		/// </summary>
		/// <returns></returns>
		double GetInferUseTime()
		{
			std::shared_lock<std::shared_mutex> lock(_time_mutex);
			return infer_use_time;
		}

		/// <summary>
		/// 获取每秒推断结果的数量
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
		/// 设置网络IO
		/// </summary>
		virtual void SetNetworkIO();
		
		/// <summary>
		/// 设置推断请求异步回调
		/// </summary>
		virtual void SetInferCallback();

		/// <summary>
		/// 解析指定的输出层数据；也可做是二次输入/输出，或是调试显示输出
		/// <param> 如果要在类内部解析，在子类内实现该函数；如果在类外部解析设置回调函数 SetCompletionCallback </param>
		/// </summary>
		virtual void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status);

		/// <summary>
		/// 更新调试并显示，该函数是用于子类做输出显示的
		/// </summary>
		virtual void UpdateDebugShow();

		Params params;
		bool is_configuration = false;		//是否已经设置网络配置
		std::vector<std::tuple<std::string, HANDLE, LPVOID>> shared_layers;	//内存共享输出层

		//引用输入的帧对象，主要是需要帧的宽，高，通道，类型、数据指针信息，浅拷贝对象
		cv::Mat frame_ptr;

		bool is_show = true;			//是否输出部份调试信息
		cv::Mat show_frame;				//用于渲染显示帧，深度拷贝对象
		std::stringstream title;		//调试输出窗口标题


		//CNN 网络对象
		InferenceEngine::CNNNetwork cnnNetwork;
		//可执行网络对象
		InferenceEngine::ExecutableNetwork execNetwork;
		//输入层数据信息
		InferenceEngine::InputsDataMap inputsInfo;
		//输出层数据信息
		InferenceEngine::OutputsDataMap outputsInfo;
		// 推断请求对象
		std::vector<InferenceEngine::InferRequest::Ptr> requestPtrs;


		//用于测量的计时器
		Timer timer;
		//FPS
		size_t fps = 0;
		size_t fps_count = 0;
		std::shared_mutex _fps_mutex;

		//获取到推断结果所耗时间
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
