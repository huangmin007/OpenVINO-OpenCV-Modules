#include "pch.h"
#include "open_model_multi_infer.hpp"
#include "static_functions.hpp"

namespace space
{
	OpenModelMultiInferBase::OpenModelMultiInferBase(const std::vector<std::string>& output_layer_names, bool is_debug)
		:output_layers(output_layer_names), is_debug(is_debug)
	{};
	OpenModelMultiInferBase::~OpenModelMultiInferBase()
	{
		timer.stop();
		for (auto& shared : shared_output_layers)
		{
			UnmapViewOfFile(shared.second);
			shared.second = NULL;
		}
		for (auto& handle : shared_output_handle)
		{
			CloseHandle(handle);
			handle = NULL;
		}

		shared_output_handle.clear();
		shared_output_layers.clear();

		LOG("INFO") << "OpenModelMultiInferBase Dispose Shared ...." << std::endl;
	}

	void OpenModelMultiInferBase::Configuration(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape, uint8_t request_count)
	{
		std::map<std::string, std::string> model = ParseArgsForModel(model_info);
		return Configuration(ie, model["path"], model["device"], is_reshape, request_count);
	}
	void OpenModelMultiInferBase::Configuration(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape, uint8_t request_count)
	{
		if (request_count < 1)
			throw std::logic_error("创建的推断请求对象不能为 0 个，最少需要 1 个 ...");

		this->is_reshape = is_reshape;
		LOG("INFO") << "正在配置网络模型 ... " << std::endl;

		LOG("INFO") << "\t读取 IR 文件: " << model_path << " [" << device << "]" << std::endl;
		cnnNetwork = ie.ReadNetwork(model_path);
		cnnNetwork.setBatchSize(is_reshape ? 1 : 4);

		//配置网络输入/输出
		ConfigNetworkIO();

		std::map<std::string, std::string> config = { };
		bool isPossibleDynBatch = device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos;
		if (isPossibleDynBatch)
		{
			config[InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM] = "8";
			config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT] = "-1";

			std::cout << "Batch Size:" << cnnNetwork.getBatchSize() << std::endl;
			if(cnnNetwork.getBatchSize() > 1)
				config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
		}

		LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device);//, config);
		LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
		for (int i = 0; i < request_count; i++)
			requestPtrs.push_back(execNetwork.CreateInferRequestPtr());

		for (static uint8_t i = 0; i < requestPtrs.size(); i++)
		{
			requestPtrs[i]->SetCompletionCallback<FCompletionCallback>((FCompletionCallback)
				[&](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code)
				{
					{
						std::lock_guard<std::shared_mutex> lock(_fps_mutex);
						fps_count++;
					}
#if _DEBUG
					InferenceEngine::Blob::Ptr data;
					request->GetBlob(outputsInfo.begin()->first.c_str(), data, 0);
					std::cout << "Result::" << (void*)(data->buffer().as<float*>()) << std::endl;
#endif
					ParsingOutputData(request, code);

					if (!this->is_reshape)
					{
						InferenceEngine::Blob::Ptr inputBlob;
						request->GetBlob(inputsInfo.begin()->first.c_str(), inputBlob, 0);
						MatU8ToBlob<uint8_t>(frame_ptr, inputBlob);
					}

					request->StartAsync(0);
				});
		}

		timer.start(1000, [&]
			{
				std::lock_guard<std::shared_mutex> lock(_fps_mutex);
				fps = fps_count;
				fps_count = 0;
			});

		//创建输出共享
		CreateMemoryShared();
		is_configuration = true;
	}

	void OpenModelMultiInferBase::ConfigNetworkIO()
	{
		LOG("INFO") << "\t配置网络输入  ... " << std::endl;
		inputsInfo = cnnNetwork.getInputsInfo();
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		if (is_reshape)
		{
			inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
			inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
		}
		//print inputs info
		for (const auto& input : inputsInfo)
		{
			InferenceEngine::SizeVector inputDims = input.second->getTensorDesc().getDims();
			std::stringstream shape; shape << "[";
			for (int i = 0; i < inputDims.size(); i++)
				shape << inputDims[i] << (i != inputDims.size() - 1 ? "x" : "]");

			LOG("INFO") << "\Input Name:[" << input.first << "]  Shape:" << shape.str() << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
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
		//一层网络输出
		if (outputsInfo.size() == 1)
		{
			output_layers.push_back(outputsInfo.begin()->first);
			int size = GetPrecisionOfSize(outputsInfo.begin()->second->getTensorDesc().getPrecision());
			InferenceEngine::SizeVector outputDims = outputsInfo.begin()->second->getTensorDesc().getDims();

			//将输出层添加到共享
			size_t shared_size = 1;
			for (auto& v : outputDims) shared_size *= v;
			shared_layers_info.clear();
			shared_layers_info.push_back({ outputsInfo.begin()->first, shared_size * size });
		}

		//print outputs info
		for (const auto& output : outputsInfo)
		{
			InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
			std::stringstream shape; shape << "[";
			for (int i = 0; i < outputDims.size(); i++)
				shape << outputDims[i] << (i != outputDims.size() - 1 ? "x" : "]");

			std::vector<std::string>::iterator iter = find(output_layers.begin(), output_layers.end(), output.first);
			LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << shape.str() << "  Precision:[" << output.second->getPrecision() << "]" << (iter != output_layers.end() ? "  \t[√]" : "") << std::endl;
		}

		//多层网络输出
		if (outputsInfo.size() > 1)
		{
			if (output_layers.size() < 2)
				THROW_IE_EXCEPTION << "当前网络模型为多层输出，但未选定网络输出名称 ... ";

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
		}


		//输出调试标题
		InferenceEngine::SizeVector outputDims = outputsInfo.find(shared_layers_info.begin()->first)->second->getTensorDesc().getDims();
		std::stringstream shape; shape << "[";
		for (int i = 0; i < outputDims.size(); i++)
			shape << outputDims[i] << (i != outputDims.size() - 1 ? "x" : "]");
		debug_title.str("");
		debug_title << "[" << cnnNetwork.getName() << "] Output Size:" << outputsInfo.size() << "  Name:[" << shared_layers_info.begin()->first << "]  Shape:" << shape.str() << std::endl;
	}

	void OpenModelMultiInferBase::CreateMemoryShared()
	{
		LOG("INFO") << "正在创建/映射共享内存块 ... " << std::endl;
		for (const auto& shared : shared_layers_info)
		{
			LPVOID pBuffer;
			HANDLE pMapFile;

			if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second, shared.first.c_str()))
			{
				shared_output_layers.push_back({ shared.first , pBuffer });

				//将输出层映射到内存共享(有些输出层精度不一样，会出现“无法创建共享blob!blob类型不能用于存储当前精度的对象”)
				//重新设置指定输出层 Blob, 将指定输出层数据指向为共享指针，实现该层数据共享
				try
				{
					//MemorySharedBlob({ shared.first , pBuffer });
				}
				catch (const std::exception& ex)
				{
					LOG("WARN") << "映射共享内存 [" << shared.first << "] 错误：" << ex.what() << std::endl;
				}
			}
			else
			{
				LOG("ERROR") << "共享内存块创建失败 ... " << std::endl;
				throw std::logic_error("共享内存块创建失败 ... ");
			}
		}
	}

	void OpenModelMultiInferBase::RequestInfer(const cv::Mat& frame)
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
			InferenceEngine::TensorDesc inputTensorDesc = inputsInfo.begin()->second->getTensorDesc();
			InferenceEngine::TensorDesc tDesc(inputTensorDesc.getPrecision(), { 1, channels, height, width }, inputTensorDesc.getLayout());
			requestPtr->SetBlob(inputsInfo.begin()->first, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));
		}
		else
		{
			InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
			MatU8ToBlob<uint8_t>(frame, inputBlob);
		}

		requestPtr->StartAsync();
	}
}