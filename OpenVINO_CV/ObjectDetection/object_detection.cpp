#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_detection.hpp"


namespace space
{
	ObjectDetection::ObjectDetection(bool isDebug) :is_debug(isDebug), confidence_threshold(0.5) {};
	ObjectDetection::ObjectDetection(const std::vector<std::string>& outputLayerNames, bool isDebug):
		output_layer_names(outputLayerNames),is_debug(isDebug),confidence_threshold(0.5){};
	ObjectDetection::~ObjectDetection()
	{
		for (auto& buffer : pBuffers) UnmapViewOfFile(buffer);
		for (auto& file : pMapFiles) CloseHandle(file);

		pBuffers.clear();
		pMapFiles.clear();
	};

	bool ObjectDetection::config(InferenceEngine::Core& ie, const std::string& modelInfo, bool isAsync, float confidenceThreshold)
	{
		std::map<std::string, std::string> model = ParseArgsForModel(modelInfo);
		return config(ie, model["path"], model["device"], isAsync, confidenceThreshold);
	};

	bool ObjectDetection::config(InferenceEngine::Core& ie, const std::string& modelPath, const std::string& device, bool isAsync, float confidenceThreshold)
	{
		try
		{
			LOG("INFO") << "����������������/��� ... " << std::endl;

			this->is_async = isAsync;
			this->confidence_threshold = confidenceThreshold;

			LOG("INFO") << "\t��ȡ IR �ļ�: " << modelPath << std::endl;
			cnnNetwork = ie.ReadNetwork(modelPath);
			cnnNetwork.setBatchSize(1);

			//-------------------��������----------------------
			networkIOConfig();

			//-------------------- �������� -------------------
			LOG("INFO") << "���ڴ�����ִ������ ... " << std::endl;
			execNetwork = ie.LoadNetwork(cnnNetwork, device);

			LOG("INFO") << "���ڴ����ƶ�������� ... " << std::endl;
			requestPtrCurr = execNetwork.CreateInferRequestPtr();
			requestPtrNext = execNetwork.CreateInferRequestPtr();

			//-------------------------�������ݹ���---------------------------------------
			LOG("INFO") << "���ڴ��������ڴ�� ... " << std::endl;
			for (const auto &shared:shared_layer_infos)
			{
				LPVOID pBuffer;
				HANDLE pMapFile;
				
				if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second, shared.first.c_str()))
				{
					pBuffers.push_back(pBuffer);
					pMapFiles.push_back(pMapFile);
				}
				else
				{
					LOG("ERROR") << "�����ڴ�鴴��ʧ�� ... " << std::endl;
					THROW_IE_EXCEPTION << "�����ڴ�鴴��ʧ�� ... ";
					break;
				}
			}
		}
		catch (const std::exception& error)
		{
			LOG("ERROR") << error.what() << std::endl;
			return false;
		}
		catch (...)
		{
			LOG("ERROR") << "����δ֪/�ڲ��쳣 ... " << std::endl;
			return false;
		}

		return true;
	}


	void ObjectDetection::networkIOConfig()
	{
		//-------------------- ����������������Ϣ -------------------
		LOG("INFO") << "\t������������  ... " << std::endl;
		inputsInfo = cnnNetwork.getInputsInfo();
		inputShapes = cnnNetwork.getInputShapes();
		//���õ�һ������뾫��Ϊ U8������Ϊͼ������ uint8_t
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		//print inputs info
		for (const auto& input : inputsInfo)
		{
			InferenceEngine::SizeVector inputDims = input.second->getTensorDesc().getDims();

			std::stringstream shape; shape << "[";
			for (int i = 0; i < inputDims.size(); i++)
				shape << inputDims[i] << (i != inputDims.size() - 1 ? "x" : "]");

			LOG("INFO") << "\tOutput Name:[" << input.first << "]  Shape:" << shape.str() << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
		}

		if (inputsInfo.size() != 1)
			THROW_IE_EXCEPTION << "ͨ�ö�����ģ��ֻ֧��һ������������ ... ";

		input_name = inputsInfo.begin()->first;
		input_batch = static_cast<uint16_t>(getTensorBatch(inputsInfo.begin()->second->getTensorDesc()));
		input_channels = static_cast<uint16_t>(getTensorChannels(inputsInfo.begin()->second->getTensorDesc()));
		input_height = static_cast<uint16_t>(getTensorHeight(inputsInfo.begin()->second->getTensorDesc()));
		input_width = static_cast<uint16_t>(getTensorWidth(inputsInfo.begin()->second->getTensorDesc()));

		debug_title.str("");
		debug_title << "Input Name:[" << input_name << "]  Shape:[" << input_batch << "x" << input_channels << "x" << input_width << "x" << input_height << "]";

		//------------------------------------------- ���������������Ϣ -----------------------------------
		LOG("INFO") << "\t�����������  ... " << std::endl;
		outputsInfo = cnnNetwork.getOutputsInfo();
		//��һ���������ȣ�Ϊ FP32 float �����ͣ�ʹ�ñ�����Ϣ����
		outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
		//print outputs info
		for (const auto& output : outputsInfo)
		{
			InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
			std::stringstream shape; shape << "[";
			for (int i = 0; i < outputDims.size(); i++)
				shape << outputDims[i] << (i != outputDims.size() - 1 ? "x" : "]");

			std::vector<std::string>::iterator iter = find(output_layer_names.begin(), output_layer_names.end(), output.first);
			LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << shape.str() << "  Precision:[" << output.second->getPrecision() << "]" << (iter != output_layer_names.end() ? "  \t[��]" : "") << std::endl;
		}

		//һ���������
		if (outputsInfo.size() == 1)
		{
			//outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
			//outputsInfo.begin()->second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(outputsInfo.begin()->second->getDims()));

			output_name = outputsInfo.begin()->first;
			firstOutputSize = outputsInfo.begin()->second->getTensorDesc().getDims();

			size_t shared_size = 1;
			for (auto& v : firstOutputSize) shared_size *= v;
			shared_layer_infos.clear();
			shared_layer_infos.push_back({ output_name, shared_size * 4 });

			if (firstOutputSize.size() == 4)
			{
				output_max_count = firstOutputSize[2];
				output_object_size = firstOutputSize[3];
			}
			else if (firstOutputSize.size() == 2)
			{
				output_max_count = firstOutputSize[0];
				output_object_size = firstOutputSize[1];
			}
			else
			{
				THROW_IE_EXCEPTION << "�����������ά�Ȳ�֧�� " << firstOutputSize.size() << " ... ";
			}
		}
		//����������
		else
		{
			if(output_layer_names.size() < 2)
				THROW_IE_EXCEPTION << "��ǰ����ģ��Ϊ����������δѡ������������� ... ";

			for (auto& output : outputsInfo)
			{
				InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
				
				std::vector<std::string>::iterator it = find(output_layer_names.begin(), output_layer_names.end(), output.first);
				if (it != output_layer_names.end())
				{
					size_t shared_size = 1;
					for (auto& v : outputDims) shared_size *= v;
					shared_layer_infos.push_back({ output.first, shared_size * 4 });
				}
			}
		}

	}

	void ObjectDetection::request(const cv::Mat& frame, int frame_idx)
	{
		//if (frame_count >= 16)
		//{
		//	LOG("WARN") << "����֡�����������ó��� 16 ֡" << std::endl;
		//	return;
		//}

		frame_width = frame.cols;
		frame_height = frame.rows;

		InferenceEngine::Blob::Ptr inputBlobPtr = requestPtrNext->GetBlob(input_name);
		inputFrameToBlob(frame, inputBlobPtr, frame_idx);

		if (is_async)
		{
			requestPtrNext->StartAsync();
			requestPtrNext->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
		}
		else
		{
			requestPtrNext->Infer();
		}

		//frame_count++;
		results_flags = false;
		requestPtrCurr.swap(requestPtrNext);
	}

	void ObjectDetection::inputFrameToBlob(const cv::Mat& frame, InferenceEngine::Blob::Ptr& blob, int batchIndex)
	{
		matU8ToBlob(frame, blob, batchIndex);
	}
	
	void ObjectDetection::matU8ToBlob(const cv::Mat& frame, InferenceEngine::Blob::Ptr& blob, int batchIndex)
	{
		//InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();

		//const size_t width = blobSize[3];
		//const size_t height = blobSize[2];
		//const size_t channels = blobSize[1];

		if (static_cast<uint16_t>(frame.channels()) != input_channels)
			THROW_IE_EXCEPTION << "���������ͼ���ͨ��������ƥ�� ... ";

		uint8_t* blob_data = blob->buffer().as<uint8_t*>();

		cv::Mat resized_image(frame);
		if (static_cast<int>(input_width) != frame.size().width || 
			static_cast<int>(input_height) != frame.size().height)
		{
			cv::resize(frame, resized_image, cv::Size(input_width, input_height));
		}

		int batchOffset = batchIndex * input_width * input_height * input_channels;
		if (input_channels == 3)
		{
			int offset = 0;
			for (int h = 0; h < input_height; h++)
			{
				uint8_t* row_pixels = resized_image.data + h * resized_image.step;
				for (int w = 0; w < input_width; w++)
				{
					offset = h * input_width + w + batchOffset;

					blob_data[offset] = row_pixels[0];
					blob_data[offset + (input_width * input_height * 1)] = row_pixels[1];
					blob_data[offset + (input_width * input_height * 2)] = row_pixels[2];

					row_pixels += 3;
				}
			}
		}
		else if(input_channels == 1)
		{
			for (size_t h = 0; h < input_height; h++)
			{
				for (size_t w = 0; w < input_width; w++)
				{
					blob_data[batchOffset + h * input_width + w] = resized_image.at<uchar>(h, w);
				}
			}
		}
		else 
		{
			THROW_IE_EXCEPTION << "ͨ����������֧�� ... ";
		}

		if (is_debug)
			cv::imshow(debug_title.str(), resized_image);
	}

	bool ObjectDetection::hasResults()
	{
		if (is_async)
			return !results_flags && requestPtrCurr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY) == InferenceEngine::StatusCode::OK;
		
		return !results_flags;
	}


	bool ObjectDetection::getResults(std::vector<ObjectDetection::Result> &results)
	{
		if (!hasResults()) return false;

		results_flags = true;		
		//frame_count = frame_count <= 0 ? 0 : frame_count - 1;

		//copy to shared memory
		for (int i = 0; i < pBuffers.size(); i++)
		{
			InferenceEngine::Blob::Ptr bolb = requestPtrCurr->GetBlob(shared_layer_infos[i].first);

			const float* buffer = bolb->buffer().as<float*>();
			std::copy(buffer, buffer + bolb->size(), (float*)pBuffers[i]);
		}

		if (!is_debug) return true;
		results.clear();

		//�������
		if (outputsInfo.size() == 1)
		{
			const float* buffer = requestPtrCurr->GetBlob(output_name)->buffer().as<float*>();

			//[1x1xNx7] FP32
			if (firstOutputSize.size() == 4 && firstOutputSize[0] == 1 && firstOutputSize[1] == 1 && firstOutputSize[3] == 7)
			{
				for (int i = 0; i < output_max_count; i++)
				{
					float image_id = buffer[i * output_object_size + 0];
					if (image_id < 0) break;

					int offset = i * output_object_size;

					Result rt;
					rt.id = static_cast<int>(image_id);
					rt.label = static_cast<int>(buffer[offset + 1]);
					rt.confidence = std::min(std::max(0.0f, buffer[offset + 2]), 1.0f);

					if (rt.confidence <= confidence_threshold) continue;

					rt.location.x = static_cast<int>(buffer[offset + 3] * frame_width);
					rt.location.y = static_cast<int>(buffer[offset + 4] * frame_height);
					rt.location.width = static_cast<int>(buffer[offset + 5] * frame_width - rt.location.x);
					rt.location.height = static_cast<int>(buffer[offset + 6] * frame_height - rt.location.y);

					results.push_back(rt);
				}
			}

			return true;
		}

		//������
		if (shared_layer_infos.size() == 2)
		{
			InferenceEngine::SizeVector outputSize = outputsInfo[shared_layer_infos[0].first]->getTensorDesc().getDims();

			//[Nx5][N]
			//[Nx5] FP32	= float
			//[N]	I32		= int32_t
			if (outputSize.size() == 2 && outputSize.back() == 5)
			{
				//[Nx5]
				const float* buffer0 = requestPtrCurr->GetBlob(shared_layer_infos[0].first)->buffer().as<float*>();
				//[N]
				const float* buffer1 = requestPtrCurr->GetBlob(shared_layer_infos[1].first)->buffer().as<float*>();

				for (int i = 0; i < outputSize[0]; i++)
				{
					Result rt;
					rt.label = buffer1[i];
					if (rt.label < 0) break;

					int offset = i * outputSize[1];
					rt.confidence = buffer0[offset + 4];
					if (rt.confidence <= confidence_threshold) continue;

					rt.location.x = static_cast<int>(buffer0[offset + 0] / input_width * frame_width);
					rt.location.y = static_cast<int>(buffer0[offset + 1] / input_height * frame_height);
					rt.location.width = static_cast<int>(buffer0[offset + 2] / input_width * frame_width - rt.location.x);
					rt.location.height = static_cast<int>(buffer0[offset + 3] / input_height * frame_height - rt.location.y);

					results.push_back(rt);
				}

				return true;
			}

		}
		else
		{
			LOG("WARN") << "����ģʽ��δ��������������ݣ�output layouts: " << outputsInfo.size() << std::endl;
		}

		return true;
	}

}
