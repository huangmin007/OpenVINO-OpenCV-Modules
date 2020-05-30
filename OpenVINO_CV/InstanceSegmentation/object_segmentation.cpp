
#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_segmentation.hpp"

#include "samples/common.hpp"
#include <samples/ocv_common.hpp>

namespace space
{
	ObjectSegmentation::ObjectSegmentation(bool isDebug) :is_debug(isDebug){};
	ObjectSegmentation::ObjectSegmentation(const std::vector<std::string>& outputLayerNames, bool isDebug) :
		output_layer_names(outputLayerNames), is_debug(isDebug) {};
	ObjectSegmentation::~ObjectSegmentation()
	{
		for (auto& buffer : pBuffers) UnmapViewOfFile(buffer);
		for (auto& file : pMapFiles) CloseHandle(file);

		pBuffers.clear();
		pMapFiles.clear();
	};

	bool ObjectSegmentation::config(InferenceEngine::Core& ie, const std::string& modelInfo, bool isAsync)
	{
		std::map<std::string, std::string> model = ParseArgsForModel(modelInfo);
		return config(ie, model["path"], model["device"], isAsync);
	};

	bool ObjectSegmentation::config(InferenceEngine::Core& ie, const std::string& modelPath, const std::string& device, bool isAsync)
	{
		LOG("INFO") << "����������������/��� ... " << std::endl;
		this->is_async = isAsync;

		try
		{
			LOG("INFO") << "\t��ȡ IR �ļ�: " << modelPath << std::endl;
			cnnNetwork = ie.ReadNetwork(modelPath);
			cnnNetwork.setBatchSize(1);

			//inputShapes = cnnNetwork.getInputShapes();
			//inputShapes.begin()->second[0] = 1;
			//cnnNetwork.reshape(inputShapes);
			//input_name = inputShapes.begin()->first;

#pragma region ����������������Ϣ
			//-------------------- ����������������Ϣ -------------------
			LOG("INFO") << "\t������������  ... " << std::endl;
			inputsInfo = cnnNetwork.getInputsInfo();
			inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
			inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
			inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
			input_name = inputsInfo.begin()->first;

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
				THROW_IE_EXCEPTION << "ͨ�ö���ָ�ģ��ֻ֧��һ������������ ... ";
			
			//------------------------------------------- ���������������Ϣ -----------------------------------
			LOG("INFO") << "\t�����������  ... " << std::endl;
			outputsInfo = cnnNetwork.getOutputsInfo();
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
				output_name = outputsInfo.begin()->first;
				outputSize = outputsInfo.begin()->second->getTensorDesc().getDims();

				size_t shared_size = 1;
				for (auto& v : outputSize) shared_size *= v;
				shared_layer_infos.clear();
				shared_layer_infos.push_back({ output_name, shared_size * 4 });

				std::cout << "B:" << getTensorBatch(outputsInfo.begin()->second->getTensorDesc()) <<
					" C:" << getTensorChannels(outputsInfo.begin()->second->getTensorDesc()) <<
					" H:" << getTensorHeight(outputsInfo.begin()->second->getTensorDesc()) <<
					" W:" << getTensorWidth(outputsInfo.begin()->second->getTensorDesc()) << std::endl;
			}
			//����������
			else
			{
				if (output_layer_names.size() < 2)
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
#pragma endregion

			//-------------------- �������� -------------------
			LOG("INFO") << "���ڴ�����ִ������ ... " << std::endl;
			execNetwork = ie.LoadNetwork(cnnNetwork, device);

			LOG("INFO") << "���ڴ����ƶ�������� ... " << std::endl;
			requestPtr = execNetwork.CreateInferRequestPtr();

			//-------------------------�������ݹ���---------------------------------------
			LOG("INFO") << "���ڴ��������ڴ�� ... " << std::endl;
			for (const auto& shared : shared_layer_infos)
			{
				LPVOID pBuffer;
				HANDLE pMapFile;

				if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second, shared.first.c_str()))
				{
					pBuffers.push_back(pBuffer);
					pMapFiles.push_back(pMapFile);

					//float* buffer = requestPtr->GetBlob(shared.first)->buffer().as<float*>();
					//buffer = (float*)pBuffer;
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

	void ObjectSegmentation::start(const cv::Mat& frame)
	{
		if (CV_8UC3 != frame.type())
			throw std::runtime_error("����ͼƬӦ֧�� CV_8UC3 ����");

		frame_height = frame.size().height;
		frame_width = frame.size().width;

		size_t channels = frame.channels();
		size_t strideH = frame.step.buf[0];
		size_t strideW = frame.step.buf[1];

		bool is_dense = strideW == channels && strideH == channels * frame_width;
		if (!is_dense) THROW_IE_EXCEPTION << "��֧�ִӷǳ��� cv::Mat ת��";

		//������ͼ���ڴ�����ָ�빲����ƶ����棬��ʵʱ���첽�ƶ�
		InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
			{ 1, channels, frame_height, frame_width }, InferenceEngine::Layout::NHWC);
		requestPtr->SetBlob(input_name, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));

		for (int i = 0; i < shared_layer_infos.size(); i ++)
		{
			//InferenceEngine::TensorDesc output_Desc(InferenceEngine::Precision::FP32,
			//	outputsInfo.begin()->second->getDims(), outputsInfo.begin()->second->getLayout());

			std::string o_name = shared_layer_infos[i].first;
			requestPtr->SetBlob(o_name, 
				InferenceEngine::make_shared_blob<float>(
					outputsInfo.find(o_name)->second->getTensorDesc(), 
					(float*)pBuffers[i]));
		}

		//InferenceEngine::TensorDesc output_Desc(InferenceEngine::Precision::FP32,
		//	outputsInfo.begin()->second->getDims(), outputsInfo.begin()->second->getLayout());
		//InferenceEngine::make_shared_blob<float>(output_Desc, (float*)pBuffers[0]);
		//requestPtr->SetBlob(output_name, InferenceEngine::make_shared_blob<float>(output_Desc, (float*)pBuffers[0]));

		requestPtr->SetCompletionCallback(
			[&]
			{
				t1 = std::chrono::high_resolution_clock::now();
				total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();				
				std::cout << "\33[2K\r[ INFO] " << "Total Use Time: " << total_use_time;

				updateDisplay();

				//InferenceEngine::StatusCode code = requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
				//std::cout << "complete... " << code << std::endl;

				requestPtr->StartAsync();
				t0 = std::chrono::high_resolution_clock::now();
			});

		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();

		InferenceEngine::StatusCode code = requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
		std::cout << "Code::" << code << std::endl;
	}

	double ObjectSegmentation::getUseTime() const
	{
		return total_use_time;
	}

	void ObjectSegmentation::updateDisplay()
	{
		//copy to shared memory
		//for (int i = 0; i < pBuffers.size(); i++)
		{
			//InferenceEngine::Blob::Ptr bolb = requestPtr->GetBlob(shared_layer_infos[i].first);

			//const float* buffer = bolb->buffer().as<float*>();
			//std::copy(buffer, buffer + bolb->size(), (float*)pBuffers[i]);
		}

		if (!is_debug) return;
		
		//�������
		//[B,H,W] = [1,H,W]
		//[B,C,H,W] = [1,3,H,W]
		if (outputsInfo.size() == 1)
		{
			int width, height, channels;
			if (outputSize.size() == 3)
			{
				channels = 0;
				height = outputSize[1];
				width = outputSize[2];
			}
			else if (outputSize.size() == 4)
			{
				channels = outputSize[1];
				height = outputSize[2];
				width = outputSize[3];
			}
			else
			{
				return;
			}

			//const float* buffer = requestPtr->GetBlob(output_name)->buffer().as<float*>();
			//buffer = (float*)pBuffers[0];
			const float* buffer = (float*)pBuffers[0];

			if (mask_img.empty())
				mask_img.create(height, width, CV_8UC3);
			
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					//��� id
					size_t classId = 0;					
					if (channels == 0)
					{
						classId = static_cast<size_t>(buffer[h * width + w]);
					}
					else
					{
						float maxProb = -1.0f;
						//�ڶ��ά�ȵ������У��ҳ����ֵ����ֵ��Ӧ��ͨ������
						for (int id = 0; id < channels; id++)
						{
							float prob = buffer[id * height * width + h * width + w];

							if (prob > maxProb)
							{
								classId = id;
								maxProb = prob;
							}
						}
					}
					//���е���ɫ���������������
					while (classId >= colors.size())
					{
						static std::mt19937 rng;
						static std::uniform_int_distribution<int> distr(0, 255);

						cv::Vec3b color(distr(rng), distr(rng), distr(rng));
						colors.push_back(color);
					}

					mask_img.at<cv::Vec3b>(h, w) = colors[classId];
				}
			}

			cv::resize(mask_img, resize_img, cv::Size(frame_width, frame_height));
			//resize_img = resize_img * 0.1f;
			
			cv::imshow("Debug Output", resize_img);
			cv::waitKey(1);

			return;
		}
		//������
		if (shared_layer_infos.size() == 2)
		{
			InferenceEngine::SizeVector outputSize = outputsInfo[shared_layer_infos[0].first]->getTensorDesc().getDims();

			if (outputSize.size() == 2 && outputSize.back() == 5)
			{
				const float* buffer0 = requestPtr->GetBlob(shared_layer_infos[0].first)->buffer().as<float*>();
				const float* buffer1 = requestPtr->GetBlob(shared_layer_infos[1].first)->buffer().as<float*>();

				for (int i = 0; i < outputSize[0]; i++)
				{

				}

				return;
			}

		}
		else
		{
			LOG("WARN") << "����ģʽ��δ��������������ݣ�output layouts: " << outputsInfo.size() << std::endl;
		}

		return;
	}

}
