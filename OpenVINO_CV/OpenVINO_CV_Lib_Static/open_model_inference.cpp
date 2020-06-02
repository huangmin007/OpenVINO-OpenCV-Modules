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
		LOG("INFO") << "������������ģ�� ... " << std::endl;

		LOG("INFO") << "\t��ȡ IR �ļ�: " << model_path << " [" << device << "]" << std::endl;
		cnnNetwork = ie.ReadNetwork(model_path);
		cnnNetwork.setBatchSize(1);

		//������������/���
		ConfigNetworkIO();

		LOG("INFO") << "���ڴ�����ִ������ ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device);
		LOG("INFO") << "���ڴ����ƶ�������� ... " << std::endl;
		requestPtr = execNetwork.CreateInferRequestPtr();	//������Դ���������ƶ϶�������ÿ�Ӳ������������

		//�����������
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
		LOG("INFO") << "\t������������  ... " << std::endl;
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

			//���������ӵ�����
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
		if(outputsInfo.size() > 1)
		{
			if (output_layers.size() < 2)
				THROW_IE_EXCEPTION << "��ǰ����ģ��Ϊ����������δѡ������������� ... ";

			//���������ӵ�����
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
			LOG("WARN") << "����ӳ�����ݣ�δ�����������ȣ�" << precision << std::endl;
		}
	}

	void OpenModelInferBase::CreateOutputShared()
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
					//input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type*>();
					MemoryOutputMapping({ shared.first , pBuffer });
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

	void OpenModelInferBase::ReshapeInput(const cv::Mat& frame)
	{
		LOG("INFO") << "���������������� ... " << std::endl;

		if (!is_configuration)
			throw std::logic_error("δ����������󣬲��ɲ��� ...");
		if(frame.empty())
			throw std::logic_error("����ͼ��֡����Ϊ��֡���� ... ");
		if (frame.type() != CV_8UC3)
			throw std::logic_error("����ͼ��֡��֧�� CV_8UC3 ���� ... ");

		frame_ptr = frame;
		size_t width = frame.cols;
		size_t height = frame.rows;
		size_t channels = frame.channels();

		size_t strideH = frame.step.buf[0];
		size_t strideW = frame.step.buf[1];

		bool is_dense = strideW == channels && strideH == channels * width;
		if (!is_dense)
			throw std::logic_error("�����ͼ��֡��֧��ת�� ... ");

		//�����޸���������
		inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
		//cnnNetwork.reshape();

		//���´����ƶ϶���
		LOG("INFO") << "���´�����ִ������ ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device);
		LOG("INFO") << "���´����ƶ�������� ... " << std::endl;
		requestPtr = execNetwork.CreateInferRequestPtr();

		//������������� Blob��������ͼ���ڴ�����ָ�빲�������㣬��ʵʱ���첽�ƶ�
		//����Ϊͼ��ԭʼ�ߴ磬��ʵ�ǻ��������ģ����Դͼ�ߴ�ܴ��Ǵ���ʱ������
		InferenceEngine::TensorDesc inputTensorDesc = inputsInfo.begin()->second->getTensorDesc();
		InferenceEngine::TensorDesc tDesc(inputTensorDesc.getPrecision(), { 1, channels, height, width }, inputTensorDesc.getLayout());
		requestPtr->SetBlob(inputsInfo.begin()->first, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));

		//����ӳ���ڴ�
		LOG("INFO") << "�����ڴ�ӳ����������� ... " << std::endl;
		for (auto& shared : shared_output_layers) MemoryOutputMapping(shared);

		//�첽�ƶ���ɵ���
		requestPtr->SetCompletionCallback(
			[&] {
				t1 = std::chrono::high_resolution_clock::now();
				infer_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();

				//��������
				ParsingOutputData();

				//ѭ���ٴ������첽�ƶ�
				requestPtr->StartAsync();
				t0 = std::chrono::high_resolution_clock::now();
			});

		//��ʼ�첽�ƶ�
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();

		is_reshape = true;
	}

	void OpenModelInferBase::RequestInfer(const cv::Mat& frame)
	{
		if (!is_configuration)
			throw std::logic_error("δ����������󣬲��ɲ��� ...");
		if (frame.empty())
			throw std::logic_error("����ͼ��֡����Ϊ��֡���� ... ");
		if (is_reshape)
			throw std::logic_error("�����������ߴ���� ResizeInput�����ɲ��� ...");

		frame_ptr = frame;
		InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
		MatU8ToBlob<uint8_t>(frame, inputBlob);

		//�����첽�ص�
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

		//��ʼ�첽�ƶ�
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