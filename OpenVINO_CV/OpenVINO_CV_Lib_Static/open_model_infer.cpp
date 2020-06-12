#include "pch.h"
#include "static_functions.hpp"
#include "open_model_infer.hpp"
#include <typeinfo>
#include <shared_mutex>

namespace space
{
	OpenModelInferBase::OpenModelInferBase(const std::vector<std::string>& output_layers, bool is_debug)
		: output_layers(output_layers), is_debug(is_debug)
	{
		//强制异步请求
		//ie_config[InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = InferenceEngine::PluginConfigParams::YES;

		//指定CPU插件应用于推理的线程数。零（默认）表示使用所有（逻辑）内核
		//ie_config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = "0";
	}

	OpenModelInferBase::~OpenModelInferBase()
	{
		timer.stop();

		LOG("INFO") << "OpenModelInferBase 正在关闭/清理共享内存  ... " << std::endl;
		for (auto& shared : output_shared_layers)
		{
			UnmapViewOfFile(shared.second);
			shared.second = NULL;
		}
		for (auto& handle : output_shared_handle)
		{
			CloseHandle(handle);
			handle = NULL;
		}

		output_shared_handle.clear();
		output_shared_layers.clear();
	}

	void OpenModelInferBase::SetParameters(const Params& params)
	{
		if (is_configuration)
			throw std::logic_error("需要在网络配置之前设置参数 ... ");

		this->batch_size = params.batch_size;
		this->infer_count = params.infer_count;
		this->is_mapping_blob = params.is_mapping_blob;
		this->request_callback = params.request_callback;
		
		for (auto& kv : params.ie_config)
			this->ie_config[kv.first] = kv.second;

		LOG("INFO") << "Parameters:" << params << std::endl;
	}

	void OpenModelInferBase::ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape)
	{
		std::map<std::string, std::string> model = ParseArgsForModel(model_info);
		return ConfigNetwork(ie, model["path"], model["device"], is_reshape);
	}
	void OpenModelInferBase::ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape)
	{
		if (batch_size < 1) 
			throw std::invalid_argument("Batch Size 不能小于 1 .");
		if (infer_count < 1)
			throw std::invalid_argument("Infer Request 数量不能小于 1 .");

		if (batch_size > 1)
			ie_config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
		if(infer_count > 1)
			ie_config[InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD] = InferenceEngine::PluginConfigParams::NO;

		this->is_reshape = is_reshape;
		LOG("INFO") << "正在配置网络模型 ... " << std::endl;

		// 1
		LOG("INFO") << "\t读取 IR 文件: " << model_path << " [" << device << "]" << std::endl;
		cnnNetwork = ie.ReadNetwork(model_path);
		cnnNetwork.setBatchSize(batch_size);

		//2.
		SetNetworkIO();	//配置网络输入/输出

		//3.
		LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
		if(ie_config.size() > 0)
			LOG("INFO") << "\t" << ie_config.begin()->first << "=" << ie_config.begin()->second << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device, ie_config);
		LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
		for (int i = 0; i < infer_count; i++)
		{
			InferenceEngine::InferRequest::Ptr inferPtr = execNetwork.CreateInferRequestPtr();
			requestPtrs.push_back(inferPtr);
		}

		//4.
		LOG("INFO") << "正在配置异步推断回调 ... " << std::endl;
		SetInferCallback();	//设置异步回调
		//5.
		LOG("INFO") << "正在创建/映射共享内存 ... " << std::endl;
		SetMemoryShared();	//创建内存共享

		is_configuration = true;
		LOG("INFO") << "网络模型配置完成 ... " << std::endl;
	}

	void OpenModelInferBase::SetNetworkIO()
	{
		LOG("INFO") << "\t配置网络输入 ... Reshape:[" << std::boolalpha << is_reshape << "]" << std::endl;
		inputsInfo = cnnNetwork.getInputsInfo();
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		if (is_reshape)
		{
			inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
			inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
		}
		else
		{
			inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NCHW);
			inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
		}

		//print inputs info
		for (const auto& input : inputsInfo)
		{
			InferenceEngine::SizeVector inputDims = input.second->getTensorDesc().getDims();
			LOG("INFO") << "\tInput Name:[" << input.first << "]  Shape:" << inputDims << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
		}

		//默认支持一层输入，子类可以更改为多层输入
		//多层输入以输入层名.bin文件指向
		//那bin的数据类型，精度怎么解释呢？
		if (inputsInfo.size() != 1)
		{
			LOG("WARN") << "当前模块只支持一个网络层的输入 ... " << std::endl;
			throw std::logic_error("当前模块只支持一个网络层的输入 ... ");
		}


		LOG("INFO") << "\t配置网络输出  ... " << std::endl;
		outputsInfo = cnnNetwork.getOutputsInfo();
		outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
		if (output_layers.size() == 0)
			for (const auto& output : outputsInfo)
				output_layers.push_back(output.first);

		//print outputs info
		for (const auto& output : outputsInfo)
		{
			InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
			std::vector<std::string>::iterator iter = find(output_layers.begin(), output_layers.end(), output.first);
			LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << outputDims << "  Precision:[" << output.second->getPrecision() << "]" <<
				(iter != output_layers.end() ? (is_mapping_blob ? "  \t[√]共享  [√]映射" : "  \t[√]共享  [×]映射") : "") << std::endl;
		}

		//将输出层添加到共享
		for (auto& output : outputsInfo)
		{
			int size = GetPrecisionOfSize(output.second->getTensorDesc().getPrecision());
			InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
			std::vector<std::string>::iterator it = find(output_layers.begin(), output_layers.end(), output.first);

			if (it != output_layers.end())
			{
				size_t shared_size = 1;
				for (auto& v : outputDims) shared_size *= v;
				shared_layers_info.push_back({ output.first, shared_size * size });
			}
		}


		//输出调试标题
		InferenceEngine::SizeVector outputDims = outputsInfo.find(shared_layers_info.begin()->first)->second->getTensorDesc().getDims();
		debug_title.str("");
		debug_title << "[" << cnnNetwork.getName() << "] Output Size:" << outputsInfo.size() << "  Name:[" << shared_layers_info.begin()->first << "]  Shape:" << outputDims << std::endl;
	}

	void OpenModelInferBase::SetInferCallback()
	{
		for (auto &requestPtr:requestPtrs)
		{
			requestPtr->SetCompletionCallback<FCompletionCallback>((FCompletionCallback)
				[&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code)
				{
					{
						t1 = std::chrono::high_resolution_clock::now();
						std::lock_guard<std::shared_mutex> lock(_time_mutex);
						infer_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
					};
					{
						std::lock_guard<std::shared_mutex> lock(_fps_mutex);
						fps_count++;
					};

					//1.解析数据，外部解析 OR 内部解析
					if (request_callback != nullptr)
						request_callback(request, code);
					else
						ParsingOutputData(request, code);

					//2.循环再次启动异步推断
					if (!this->is_reshape)
					{
						InferenceEngine::Blob::Ptr inputBlob;
						request->GetBlob(inputsInfo.begin()->first.c_str(), inputBlob, 0);
						MatU8ToBlob<uint8_t>(frame_ptr, inputBlob);
					};
					//request->SetBatch(batch_size, 0);
					request->StartAsync(0);
					t0 = std::chrono::high_resolution_clock::now();

					//3.更新显示结果
					if (is_debug) UpdateDebugShow();
				});
		}
		timer.start(1000, [&]
			{
#if _TEST
				t3 = std::chrono::high_resolution_clock::now();
				test_use_time = std::chrono::duration_cast<ms>(t3 - t2).count();
				std::cout << " Test Timer Count:" << test_use_time << std::endl;
				t2 = t3;
#endif
				{
					std::lock_guard<std::shared_mutex> lock(_fps_mutex);
					fps = fps_count;
					fps_count = 0;
				};
			});
	}
	
	void OpenModelInferBase::SetMemoryShared()
	{
		for (const auto& shared : shared_layers_info)
		{
			LPVOID pBuffer;
			HANDLE pMapFile;

			if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second + SHARED_RESERVE_BYTE_SIZE, shared.first.c_str()))
			{
				output_shared_handle.push_back(pMapFile);
				output_shared_layers.push_back({ shared.first , pBuffer });

				//将输出层映射到内存共享
				//重新设置指定输出层 Blob, 将指定输出层数据指向为共享指针，实现该层数据共享
				//if(is_mapping_blob && infer_count == 1)
				//	MemorySharedBlob(requestPtrs[0], output_shared_layers.back(), SHARED_RESERVE_BYTE_SIZE);

				//多个推断对象，输出映射到同一个共享位置？？
				if (is_mapping_blob)
					for (auto& requestPtr : requestPtrs)
						MemorySharedBlob(requestPtr, output_shared_layers.back(), SHARED_RESERVE_BYTE_SIZE);

				//LOG("INGO") << "共享内存 " << shared.first << " 映射：" << std::boolalpha << is_mapping_blob << std::endl;
			}
			else
			{
				LOG("ERROR") << "共享内存块创建失败 ... " << std::endl;
				throw std::logic_error("共享内存块创建失败 ... ");
			}
		}
	}

	void OpenModelInferBase::RequestInfer(const cv::Mat& frame)
	{
		if (!is_configuration)
			throw std::logic_error("未配置网络对象，不可操作 ...");
		if (frame.empty() || frame.type() != CV_8UC3)
			throw std::logic_error("输入图像帧不能为空帧对象，CV_8UC3 类型 ... ");

		frame_ptr = frame;
		InferenceEngine::StatusCode code;
		InferenceEngine::InferRequest::Ptr requestPtr;

		//查找没有启动的推断对象
		for (auto& request : requestPtrs)
		{
			code = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
			if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED)
			{
				requestPtr = request;
				break;
			}
		}
		if (requestPtr == nullptr) return;

		InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
		LOG("INFO") << "找到一个 InferRequestPtr 对象，Input Address:" << (void*)inputBlob->buffer().as<uint8_t*>() << std::endl;

		if (is_reshape)
		{
			size_t width = frame.cols;
			size_t height = frame.rows;
			size_t channels = frame.channels();

			size_t strideH = frame.step.buf[0];
			size_t strideW = frame.step.buf[1];

			bool is_dense = strideW == channels && strideH == channels * width;
			if (!is_dense) throw std::logic_error("输入的图像帧不支持转换 ... ");

			//重新设置输入层 Blob，将输入图像内存数据指针共享给输入层，做实时或异步推断
			//输入为图像原始尺寸，其实是会存在问题的，如果源图尺寸很大，那处理时间会更长
			InferenceEngine::TensorDesc tensor = inputBlob->getTensorDesc();
			InferenceEngine::TensorDesc n_tensor(tensor.getPrecision(), { 1, channels, height, width }, tensor.getLayout());
			requestPtr->SetBlob(inputsInfo.begin()->first, InferenceEngine::make_shared_blob<uint8_t>(n_tensor, frame.data));
		}
		else
		{
			MatU8ToBlob<uint8_t>(frame, inputBlob);
		}

		//requestPtr->SetBatch(batch_size);
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();
	}
}