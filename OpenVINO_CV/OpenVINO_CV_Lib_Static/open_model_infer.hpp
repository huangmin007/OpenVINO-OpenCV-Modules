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

	class 笔记
	{
	public:
		std::map<std::string, std::string> plugin_config =
		{
			//通用布尔值:InferenceEngine::PluginConfigParams::NO/YES

			//限制推理引擎用于在CPU上进行推理的线程
			{InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, InferenceEngine::PluginConfigParams::NO},
			//设置每个线程的CPU亲和性选项的名称(YES:将线程固定到核心，最适合静态基准)(NUMA:将radrad固定到NUMA节点，最适合实际的)(NO:CPU推理线程无固定)
			{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO},

			//优化CPU执行以最大化吞吐量。此选项应与值一起使用：
			//CPU_THROUGHPUT_NUMA:创建所需数量的流以适应NUMA并避免相关的罚款
			//CPU_THROUGHPUT_AUTO:创建最少的流以提高性能，如果您不了解目标计算机将拥有多少个内核（以及最佳的流数量是多少），这是最可移植的选项
			//指定正整数值将创建请求的流数量
			{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "CPU_THROUGHPUT_NUMA/CPU_THROUGHPUT_AUTO"},


			//优化GPU插件执行以最大化吞吐量,此选项应与值一起使用：
			//GPU_THROUGHPUT_AUTO:创建最少的流，在某些情况下可能会提高性能，此选项允许为opencl队列启用节流阀提示，从而减少CPU负载而不会显着降低性能
			//正整数值创建请求的流数量
			{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "GPU_THROUGHPUT_AUTO"},

			//设置性能计数器选项的名称,与以下值一起使用：
			//PluginConfigParams::YES or PluginConfigParams::NO
			{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "YES/NO"},


			//是否开启动态批处理
			{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES},
			//关键字定义批处理的动态限制 可接受的值：
			//-1 不限制批处理 
			//>0 限制的直接值。要处理的批次大小为最小值（新批次限制，原始批次）
			{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "-1"},

			{InferenceEngine::PluginConfigParams::KEY_DUMP_QUANTIZED_GRAPH_AS_DOT, "?"},
			{InferenceEngine::PluginConfigParams::KEY_DUMP_QUANTIZED_GRAPH_AS_IR, "?"},

			//控制推理引擎中的线程(单线程),此选项应与值一起使用：
			//PluginConfigParams::YES or PluginConfigParams::NO
			{InferenceEngine::PluginConfigParams::KEY_SINGLE_THREAD, "YES/NO"},

			//指示插件加载配置文件,该值应该是具有插件特定配置的文件名
			{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, ""},

			//启用转储插件用于自定义层的内核,此选项应与值一起使用：
			//PluginConfigParams::YES or PluginConfigParams::NO (default)
			{InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, ""},

			//控制插件完成或使用的性能调整,选择值：
			//TUNING_DISABLED: (default)
			//TUNING_USE_EXISTING:使用调整文件中的现有数据
			//TUNING_CREATE:为调整文件中不存在的参数创建调整数据
			//TUNING_UPDATE:执行非调整更新，例如删除无效/已弃用的数据
			//TUNING_RETUNE:为所有参数创建调整数据，即使已经存在
			//对于值 TUNING_CREATE 和 TUNING_RETUNE，如果文件不存在，则将创建该文件
			{InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_CREATE/TUNING_USE_EXISTING/TUNING_DISABLED/TUNING_UPDATE/TUNING_RETUNE"},
			{InferenceEngine::PluginConfigParams::KEY_TUNING_FILE, ""},

			//设置日志级别
			{InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "LOG_NONE/LOG_ERROR/LOG_WARNING/LOG_INFO/LOG_DEBUG/LOG_TRACE"},

			//setting of required device to execute on values: 
			//设备ID从 0 开始-第一个设备，1-第二个设备，依此类推
			{InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, ""},


			//为不同的可执行网络和相同插件的异步请求启用独占模式。
			//有时有必要避免并行共享同一设备的超额预订请求。
			//例如，有2个用于CPU设备的任务执行程序：一个-在Hetero插件中，另一个-在纯CPU插件中。
			//两者的并行执行可能会导致超额预订，而不是最佳的CPU使用率。
			//通过单个执行程序一个接一个地运行相应任务的效率更高。
			//默认情况下，对于其他情况，此选项设置为YES，对于常规情况（单插件），此选项设置为NO。
			//请注意，设置为YES会禁用CPU流功能（请参阅此文件中的另一个配置键）
			{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, ""},

			//启用内部基本图的转储
			//应该传递给LoadNetwork方法以启用内部图元转储和相应的配置信息。值是不带扩展名的输出点文件的名称。
			//文件<dot_file_name>_init.dot和<dot_file_name>_perf.dot会产生。
			{InferenceEngine::PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, ""},
		};
	};

	//推断完成回调对象
	typedef InferenceEngine::IInferRequest::CompletionCallback	TCompletionCallback;
	typedef std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>		FCompletionCallback;
	typedef std::function<void(InferenceEngine::IInferRequest::Ptr, InferenceEngine::StatusCode)>		FPCompletionCallback;

	/// <summary>
	/// 开放模型推断基类，基于图像为输入源，异步推断，输出层内存映射共享
	/// <para>基本默认只创建了一个 InferRequest 对象，子类可以自已创建多个</para>
	/// </summary>
	class OpenModelInferBase
	{
	public:
		/// <summary>
		/// OpenModelInferBase Params
		/// </summary>
		struct Params
		{
			size_t batch_size = 1;				//设置批处理大小
			size_t infer_count = 1;				//创建的推断请求数量
			bool is_mapping_blob = true;		//是否共享映射指定的输出层数据
			//IE 引擎插件配置
			std::map<std::string, std::string> ie_config;
			//设置推断完成外部回调
			FPCompletionCallback request_callback = nullptr;	
			//异步回调完成后，自动请求下一次推断
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
		/// 开放模型推断构造函数
		/// </summary>
		/// <param name="output_layer_names">需要指定的输出层名称，该参数用于有多层输出的网络</param>
		/// <param name="is_debug"></param>
		/// <returns></returns>
		OpenModelInferBase(const std::vector<std::string>& output_layer_names, bool is_debug = true);
		~OpenModelInferBase();

		/// <summary>
		/// 设置相关参数，需在 ConfigNetwork 之前设置
		/// </summary>
		/// <param name="params"></param>
		void SetParameters(const Params& params);

		/// <summary>
		/// 配置网络模型
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="model_info">具有一定协议格式的模型参数，格式：(模型名称[:精度[:硬件]])，示例：human-pose-estimation-0001:FP32:CPU</param>
		/// <param name="is_reshape">是否重塑输入层，重新调整输入，指按原帧数据尺寸、数据指针位置做为输入(减少内存数据复制)，做异步推断请求</param>
		void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape = true);

		/// <summary>
		/// 配置网络模型
		/// </summary>
		/// <param name="ie">ie推断引擎对象</param>
		/// <param name="model_path">ai模型文件路径</param>
		/// <param name="device">推断使用的硬件类型</param>
		/// <param name="is_reshape">是否重塑输入层，重新调整输入，指按原帧数据尺寸、数据指针位置做为输入(减少内存数据复制)，做异步推断请求</param>
		void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape = true);

		/// <summary>
		/// 异步推断请求，需实时提交图像帧；当 is_reshape 为 true 时只需调用一次，多次调用也不影响
		/// </summary>
		/// <param name="frame">图像帧对象</param>
		void RequestInfer(const cv::Mat& frame);

		/// <summary>
		/// 获取当前对象的执行网络
		/// </summary>
		/// <returns></returns>
		const InferenceEngine::ExecutableNetwork GetExecutableNetwork() const
		{
			return execNetwork;
		}
		/// <summary>
		/// 获取共享输出层映射信息
		/// </summary>
		/// <returns></returns>
		const std::vector<std::pair<std::string, LPVOID>> GetSharedOutputLayers() const
		{
			return output_shared_layers;
		}

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

	protected:

		/// <summary>
		/// 设置网络输入/输出
		/// </summary>
		virtual void SetNetworkIO();

		/// <summary>
		/// 设置推断请求异步回调
		/// </summary>
		virtual void SetInferCallback();

		/// <summary>
		/// 设置内存共享，将指定的网络输出层数据共享 
		/// </summary>
		virtual void SetMemoryShared();

		/// <summary>
		/// 解析指定的输出层数据；也可做是二次输入/输出，或是调试显示输出
		/// <param> 如果要在类内部解析，在子类内实现该函数；如果在类外部解析设置回调函数 SetCompletionCallback </param>
		/// </summary>
		virtual void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) = 0;

		/// <summary>
		/// 更新调试并显示，该函数是用于子类做输出显示的
		/// </summary>
		virtual void UpdateDebugShow() = 0;

		size_t batch_size = 1;				//批处理大小
		size_t infer_count = 1;				//创建的推断请求数量
		bool is_mapping_blob = true;		//是否共享映射指定的输出层数据
		bool is_configuration = false;		//是否已经设置网络配置
		// IE 引擎插件配置
		std::map<std::string, std::string> ie_config = { };


		bool is_debug = true;			//是否输出部份调试信息
		bool is_reshape = true;			//是否重新调整模型输入
		std::stringstream debug_title;	//调试输出窗口标题

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

		FPCompletionCallback request_callback = nullptr;

		//引用输入的帧对象，主要是需要帧的宽，高，通道，类型、数据指针信息
		cv::Mat frame_ptr;
		//用于操作共享层的数据，网络层名称/内存映射视图指针
		std::vector<std::pair<std::string, LPVOID>> output_shared_layers;
		
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

		//需要指定的输出层名称，该参数用于有多层输出的网络
		std::vector<std::string> output_layers;
		std::vector<HANDLE> output_shared_handle;	//共享内存文件句柄
		//用于创建内存共享的信息，指定的网络输出层名称/分配存储空间大小
		std::vector<std::pair<std::string, size_t>> shared_layers_info;

#if _TEST
		//用于测量 Timer 的准确性变量
		double test_use_time = 0.0f;
		std::chrono::steady_clock::time_point t2;
		std::chrono::steady_clock::time_point t3;
#endif

	};

}
