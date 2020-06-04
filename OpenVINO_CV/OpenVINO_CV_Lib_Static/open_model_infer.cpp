#include "pch.h"
#include "static_functions.hpp"
#include "open_model_infer.hpp"
#include <typeinfo>
#include <shared_mutex>

namespace space
{
	OpenModelInferBase::OpenModelInferBase(const std::vector<std::string>& output_layer_names, bool is_debug)
		: output_layers(output_layer_names), is_debug(is_debug)
	{
	}

	OpenModelInferBase::~OpenModelInferBase()
	{
		timer.stop();

		LOG("INFO") << "���ڹر�/�������ڴ�  ... " << std::endl;
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
	}

	void OpenModelInferBase::Configuration(InferenceEngine::Core& ie, const std::string& model_info, bool is_reshape)
	{
		std::map<std::string, std::string> model = ParseArgsForModel(model_info);
		return Configuration(ie, model["path"], model["device"], is_reshape);
	}
	void OpenModelInferBase::Configuration(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device, bool is_reshape)
	{
		if (batch_size < 1) 
			throw std::invalid_argument("Batch Size ����С�� 1 .");

		if (batch_size > 1)
			ie_config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;

		this->is_reshape = is_reshape;
		LOG("INFO") << "������������ģ�� ... " << std::endl;

		// 1
		LOG("INFO") << "\t��ȡ IR �ļ�: " << model_path << " [" << device << "]" << std::endl;
		cnnNetwork = ie.ReadNetwork(model_path);
		cnnNetwork.setBatchSize(batch_size);

		//2.
		ConfigNetworkIO();	//������������/���

		//3.
		LOG("INFO") << "���ڴ�����ִ������ ... " << std::endl;
		execNetwork = ie.LoadNetwork(cnnNetwork, device, ie_config);
		LOG("INFO") << "���ڴ����ƶ�������� ... " << std::endl;
		requestPtr = execNetwork.CreateInferRequestPtr();

		//4.
		LOG("INFO") << "���������첽�ƶϻص� ... " << std::endl;
		SetRequestCallback();	//�����첽�ص�
		//5.
		LOG("INFO") << "���ڴ���/ӳ�乲���ڴ� ... " << std::endl;
		CreateMemoryShared();	//�����ڴ湲��

		is_configuration = true;
		LOG("INFO") << "����ģ��������� ... " << std::endl;
	}

	void OpenModelInferBase::ConfigNetworkIO()
	{
		LOG("INFO") << "\t������������ ... " << std::endl;
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
			LOG("INFO") << "\Input Name:[" << input.first << "]  Shape:" << inputDims << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
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
		//1.һ���������
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
		
		//2.print outputs info
		for (const auto& output : outputsInfo)
		{
			InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
			std::vector<std::string>::iterator iter = find(output_layers.begin(), output_layers.end(), output.first);
			LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << outputDims << "  Precision:[" << output.second->getPrecision() << "]" << 
				(iter != output_layers.end() ? (is_shared_blob ? "  \t[��]����  [��]ӳ��" : "  \t[��]����  [��]ӳ��") : "") << std::endl;
		}
		
		//3.����������
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
		debug_title.str("");
		debug_title << "[" << cnnNetwork.getName() << "] Output Size:" << outputsInfo.size() << "  Name:[" << shared_layers_info.begin()->first << "]  Shape:" << outputDims << std::endl;
	}

	void OpenModelInferBase::SetRequestCallback()
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

				//1.��������
				ParsingOutputData(request, code);

				//2.ѭ���ٴ������첽�ƶ�
				if (!this->is_reshape)
				{
					InferenceEngine::Blob::Ptr inputBlob;
					request->GetBlob(inputsInfo.begin()->first.c_str(), inputBlob, 0);
					MatU8ToBlob<uint8_t>(frame_ptr, inputBlob);
				};
				request->StartAsync(0);
				t0 = std::chrono::high_resolution_clock::now();

				//3.������ʾ���
				if (is_debug) UpdateDebugShow();
			});

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
	
	void OpenModelInferBase::CreateMemoryShared()
	{
		for (const auto& shared : shared_layers_info)
		{
			LPVOID pBuffer;
			HANDLE pMapFile;

			if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second, shared.first.c_str()))
			{
				shared_output_layers.push_back({ shared.first , pBuffer });

				//�������ӳ�䵽�ڴ湲��
				//��������ָ������� Blob, ��ָ�����������ָ��Ϊ����ָ�룬ʵ�ָò����ݹ���
				if(is_shared_blob)
					MemorySharedBlob(requestPtr, shared_output_layers.back());
			}
			else
			{
				LOG("ERROR") << "�����ڴ�鴴��ʧ�� ... " << std::endl;
				throw std::logic_error("�����ڴ�鴴��ʧ�� ... ");
			}
		}
	}

	void OpenModelInferBase::RequestInfer(const cv::Mat& frame)
	{
		if (!is_configuration)
			throw std::logic_error("δ����������󣬲��ɲ��� ...");
		if (frame.empty() || frame.type() != CV_8UC3)
			throw std::logic_error("����ͼ��֡����Ϊ��֡����CV_8UC3 ���� ... ");

		frame_ptr = frame;
		InferenceEngine::StatusCode code = requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
		//�ƶ�û�������ͼ�������ִ��
		if (code != InferenceEngine::StatusCode::INFER_NOT_STARTED)	return;

		if (is_reshape)
		{
			size_t width = frame.cols;
			size_t height = frame.rows;
			size_t channels = frame.channels();

			size_t strideH = frame.step.buf[0];
			size_t strideW = frame.step.buf[1];

			bool is_dense = strideW == channels && strideH == channels * width;
			if (!is_dense) throw std::logic_error("�����ͼ��֡��֧��ת�� ... ");
			
			//������������� Blob��������ͼ���ڴ�����ָ�빲�������㣬��ʵʱ���첽�ƶ�
			//����Ϊͼ��ԭʼ�ߴ磬��ʵ�ǻ��������ģ����Դͼ�ߴ�ܴ��Ǵ���ʱ������
			InferenceEngine::TensorDesc tensor = inputsInfo.begin()->second->getTensorDesc();
			InferenceEngine::TensorDesc n_tensor(tensor.getPrecision(), { 1, channels, height, width }, tensor.getLayout());
			requestPtr->SetBlob(inputsInfo.begin()->first, InferenceEngine::make_shared_blob<uint8_t>(n_tensor, frame.data));
		}
		else		
		{
			InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
			MatU8ToBlob<uint8_t>(frame, inputBlob);
		}
		
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();
	}
}