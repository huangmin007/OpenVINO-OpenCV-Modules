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
			throw std::logic_error("�������ƶ����������Ϊ 0 ����������Ҫ 1 �� ...");

		this->is_reshape = is_reshape;
		LOG("INFO") << "������������ģ�� ... " << std::endl;

		LOG("INFO") << "\t��ȡ IR �ļ�: " << model_path << " [" << device << "]" << std::endl;
		cnnNetwork = ie.ReadNetwork(model_path);
		cnnNetwork.setBatchSize(is_reshape ? 1 : 4);

		//������������/���
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

		LOG("INFO") << "���ڴ�����ִ������ ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device);//, config);
		LOG("INFO") << "���ڴ����ƶ�������� ... " << std::endl;
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

		//�����������
		CreateMemoryShared();
		is_configuration = true;
	}

	void OpenModelMultiInferBase::ConfigNetworkIO()
	{
		LOG("INFO") << "\t������������  ... " << std::endl;
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

		//Ĭ��֧��һ�����룬������Ը���Ϊ�������
		//����������������.bin�ļ�ָ��
		//��bin���������ͣ�������ô�����أ�
		if (inputsInfo.size() != 1)
		{
			LOG("WARN") << "��ǰģ��ֻ֧��һ������������ ... " << std::endl;
			throw std::logic_error("��ǰģ��ֻ֧��һ������������ ... ");
		}


		LOG("INFO") << "\t�����������  ... " << std::endl;
		outputsInfo = cnnNetwork.getOutputsInfo();
		outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
		//һ���������
		if (outputsInfo.size() == 1)
		{
			output_layers.push_back(outputsInfo.begin()->first);
			int size = GetPrecisionOfSize(outputsInfo.begin()->second->getTensorDesc().getPrecision());
			InferenceEngine::SizeVector outputDims = outputsInfo.begin()->second->getTensorDesc().getDims();

			//����������ӵ�����
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
			LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << shape.str() << "  Precision:[" << output.second->getPrecision() << "]" << (iter != output_layers.end() ? "  \t[��]" : "") << std::endl;
		}

		//����������
		if (outputsInfo.size() > 1)
		{
			if (output_layers.size() < 2)
				THROW_IE_EXCEPTION << "��ǰ����ģ��Ϊ����������δѡ������������� ... ";

			//����������ӵ�����
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


		//������Ա���
		InferenceEngine::SizeVector outputDims = outputsInfo.find(shared_layers_info.begin()->first)->second->getTensorDesc().getDims();
		std::stringstream shape; shape << "[";
		for (int i = 0; i < outputDims.size(); i++)
			shape << outputDims[i] << (i != outputDims.size() - 1 ? "x" : "]");
		debug_title.str("");
		debug_title << "[" << cnnNetwork.getName() << "] Output Size:" << outputsInfo.size() << "  Name:[" << shared_layers_info.begin()->first << "]  Shape:" << shape.str() << std::endl;
	}

	void OpenModelMultiInferBase::CreateMemoryShared()
	{
		LOG("INFO") << "���ڴ���/ӳ�乲���ڴ�� ... " << std::endl;
		for (const auto& shared : shared_layers_info)
		{
			LPVOID pBuffer;
			HANDLE pMapFile;

			if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second, shared.first.c_str()))
			{
				shared_output_layers.push_back({ shared.first , pBuffer });

				//�������ӳ�䵽�ڴ湲��(��Щ����㾫�Ȳ�һ��������֡��޷���������blob!blob���Ͳ������ڴ洢��ǰ���ȵĶ���)
				//��������ָ������� Blob, ��ָ�����������ָ��Ϊ����ָ�룬ʵ�ָò����ݹ���
				try
				{
					//MemorySharedBlob({ shared.first , pBuffer });
				}
				catch (const std::exception& ex)
				{
					LOG("WARN") << "ӳ�乲���ڴ� [" << shared.first << "] ����" << ex.what() << std::endl;
				}
			}
			else
			{
				LOG("ERROR") << "�����ڴ�鴴��ʧ�� ... " << std::endl;
				throw std::logic_error("�����ڴ�鴴��ʧ�� ... ");
			}
		}
	}

	void OpenModelMultiInferBase::RequestInfer(const cv::Mat& frame)
	{
		if (!is_configuration)
			throw std::logic_error("δ����������󣬲��ɲ��� ...");
		if (frame.empty() || frame.type() != CV_8UC3)
			throw std::logic_error("����ͼ��֡����Ϊ��֡����CV_8UC3 ���� ... ");

		frame_ptr = frame;
		InferenceEngine::StatusCode code;
		InferenceEngine::InferRequest::Ptr requestPtr;

		//����û���������ƶ϶���
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
			if (!is_dense) throw std::logic_error("�����ͼ��֡��֧��ת�� ... ");

			//������������� Blob��������ͼ���ڴ�����ָ�빲��������㣬��ʵʱ���첽�ƶ�
			//����Ϊͼ��ԭʼ�ߴ磬��ʵ�ǻ��������ģ����Դͼ�ߴ�ܴ��Ǵ���ʱ������
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