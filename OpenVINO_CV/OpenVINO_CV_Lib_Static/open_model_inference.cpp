#include "pch.h"
#include "static_functions.hpp"
#include "open_model_inference.hpp"
#include <typeinfo>

namespace space
{
	OpenModelInferBase::OpenModelInferBase(bool is_debug) :is_debug(is_debug)
	{
	}
	OpenModelInferBase::OpenModelInferBase(const std::vector<std::string>& output_layer_names, bool is_debug)
		: output_layers(output_layer_names), is_debug(is_debug)
	{
	}

	OpenModelInferBase::~OpenModelInferBase()
	{
		for (auto& shared : shared_output_layers)
			UnmapViewOfFile(shared.second);
		for (auto& handle : shared_output_handle)
			CloseHandle(handle);

		shared_output_handle.clear();
		shared_output_layers.clear();

		LOG("INFO") << "OpenModelInferBase Dispose Shared ...." << std::endl;
	}

	void OpenModelInferBase::Configuration(InferenceEngine::Core& ie, const std::string& model_info)
	{
		std::map<std::string, std::string> model = ParseArgsForModel(model_info);
		return Configuration(ie, model["path"], model["device"]);
	}
	void OpenModelInferBase::Configuration(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device)
	{
		this->ie = ie;
		this->device = device;
		LOG("INFO") << "正在配置网络模型 ... " << std::endl;

		LOG("INFO") << "\t读取 IR 文件: " << model_path << " [" << device << "]" << std::endl;
		cnnNetwork = ie.ReadNetwork(model_path);
		cnnNetwork.setBatchSize(1);

		//配置网络输入/输出
		ConfigNetworkIO();

		LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device);
		LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
		requestPtr = execNetwork.CreateInferRequestPtr();	//子类可以创建更多的推断对象，这个得看硬件性能配置了

		//创建输出共享
		CreateOutputShared();

		is_configuration = true;
	}

	size_t OpenModelInferBase::GetPrecisionOfSize(const InferenceEngine::Precision &precision)
	{
		switch (precision)
		{
		case InferenceEngine::Precision::U64:	//uint64_t
		case InferenceEngine::Precision::I64:	//int64_t
			return sizeof(uint64_t);

		case InferenceEngine::Precision::FP32:	//float
			return sizeof(float);

		case InferenceEngine::Precision::I32:	//int32_t
			return sizeof(int32_t);

		case InferenceEngine::Precision::U16:	//uint16_t
		case InferenceEngine::Precision::I16:	//int16_t
		case InferenceEngine::Precision::Q78:	//int16_t, uint16_t
		case InferenceEngine::Precision::FP16:	//int16_t, uint16_t	
			return sizeof(uint16_t);

		case InferenceEngine::Precision::U8:	//uint8_t
		case InferenceEngine::Precision::BOOL:	//uint8_t
		case InferenceEngine::Precision::I8:	//int8_t
		case InferenceEngine::Precision::BIN:	//int8_t, uint8_t
			return sizeof(uint8_t);
		}

		return sizeof(uint8_t);
	}

	void OpenModelInferBase::ConfigNetworkIO()
	{
		LOG("INFO") << "\t配置网络输入  ... " << std::endl;
		inputsInfo = cnnNetwork.getInputsInfo();
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);

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
		if(outputsInfo.size() > 1)
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
		for(int i = 0; i < outputDims.size(); i ++)
			shape << outputDims[i] << (i != outputDims.size() - 1 ? "x" : "]");
		debug_title.str("");
		debug_title << "[" << cnnNetwork.getName() << "] Output Size:" << outputsInfo.size() << "  Name:[" << shared_layers_info.begin()->first << "]  Shape:" << shape.str() << std::endl;
	}

	void OpenModelInferBase::MemoryOutputMapping(const std::pair<std::string, LPVOID> &shared)
	{
		const InferenceEngine::Precision precision = outputsInfo.find(shared.first)->second->getPrecision();

		switch (precision)
		{
		case InferenceEngine::Precision::U64:	//uint64_t
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<uint64_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (uint64_t*)shared.second));
			break;
		case InferenceEngine::Precision::I64:	//int64_t
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int64_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (int64_t*)shared.second));
			break;
		case InferenceEngine::Precision::FP32:	//float
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<float>(outputsInfo.find(shared.first)->second->getTensorDesc(), (float*)shared.second));
			break;
		case InferenceEngine::Precision::I32:	//int32_t
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int32_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (int32_t*)shared.second));
			break;
		case InferenceEngine::Precision::U16:	//uint16_t
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<uint16_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (uint16_t*)shared.second));
			break;
		case InferenceEngine::Precision::I16:	//int16_t
		case InferenceEngine::Precision::Q78:	//int16_t, uint16_t
		case InferenceEngine::Precision::FP16:	//int16_t, uint16_t	
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int16_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (int16_t*)shared.second));
			break;
		case InferenceEngine::Precision::U8:	//uint8_t
		case InferenceEngine::Precision::BOOL:	//uint8_t
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<uint8_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (uint8_t*)shared.second));
			break;
		case InferenceEngine::Precision::I8:	//int8_t
		case InferenceEngine::Precision::BIN:	//int8_t, uint8_t
			requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int8_t>(outputsInfo.find(shared.first)->second->getTensorDesc(), (int8_t*)shared.second));
			break;
		default:
			LOG("WARN") << "共享映射数据，未处理的输出精度：" << precision << std::endl;
		}
	}

	void OpenModelInferBase::CreateOutputShared()
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
					//input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();
					MemoryOutputMapping({ shared.first , pBuffer });
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

	void OpenModelInferBase::ReshapeInput(const cv::Mat& frame)
	{
		LOG("INFO") << "重新设置输入层参数 ... " << std::endl;

		if (!is_configuration)
			throw std::logic_error("未配置网络对象，不可操作 ...");
		if(frame.empty())
			throw std::logic_error("输入图像帧不能为空帧对象 ... ");
		if (frame.type() != CV_8UC3)
			throw std::logic_error("输入图像帧不支持 CV_8UC3 类型 ... ");

		frame_ptr = frame;
		size_t width = frame.cols;
		size_t height = frame.rows;
		size_t channels = frame.channels();

		size_t strideH = frame.step.buf[0];
		size_t strideW = frame.step.buf[1];

		bool is_dense = strideW == channels && strideH == channels * width;
		if (!is_dense)
			throw std::logic_error("输入的图像帧不支持转换 ... ");

		//重新修改输入层参数
		inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
		//cnnNetwork.reshape();

		//重新创建推断对象
		LOG("INFO") << "重新创建可执行网络 ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device);
		LOG("INFO") << "重新创建推断请求对象 ... " << std::endl;
		requestPtr = execNetwork.CreateInferRequestPtr();

		//重新设置输入层 Blob，将输入图像内存数据指针共享给输入层，做实时或异步推断
		//输入为图像原始尺寸，其实是会存在问题的，如果源图尺寸很大，那处理时间会更长
		InferenceEngine::TensorDesc inputTensorDesc = inputsInfo.begin()->second->getTensorDesc();
		InferenceEngine::TensorDesc tDesc(inputTensorDesc.getPrecision(), { 1, channels, height, width }, inputTensorDesc.getLayout());
		requestPtr->SetBlob(inputsInfo.begin()->first, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));

		//重新映射内存
		LOG("INFO") << "重新内存映射网络输出层 ... " << std::endl;
		for (auto& shared : shared_output_layers) MemoryOutputMapping(shared);

		//异步推断完成调用
		requestPtr->SetCompletionCallback(
			[&] {
				t1 = std::chrono::high_resolution_clock::now();
				infer_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

				//解析数据
				ParsingOutputData();

				//循环再次启动异步推断
				requestPtr->StartAsync();
				t0 = std::chrono::high_resolution_clock::now();
			});

		//开始异步推断
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();

		is_reshape = true;
	}

	void OpenModelInferBase::RequestInfer(const cv::Mat& frame)
	{
		if (!is_configuration)
			throw std::logic_error("未配置网络对象，不可操作 ...");
		if (frame.empty())
			throw std::logic_error("输入图像帧不能为空帧对象 ... ");
		if (is_reshape)
			throw std::logic_error("已重设过输入尺寸参数 ResizeInput，不可操作 ...");

		frame_ptr = frame;
		InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
		MatU8ToBlob<uint8_t>(frame, inputBlob);

		//设置异步回调
		static bool is_async_callback = false;
		if (!is_async_callback)
		{
			requestPtr->SetCompletionCallback(
				[&] {
					t1 = std::chrono::high_resolution_clock::now();
					infer_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

					ParsingOutputData();
				});

			is_async_callback = true;
		}

		//开始异步推断
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();
		requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
	}

	double OpenModelInferBase::GetInferUseTime(){	return infer_use_time;	}

	void OpenModelInferBase::ParsingOutputData()
	{
		if (is_debug) UpdateDebugShow();
	}

}