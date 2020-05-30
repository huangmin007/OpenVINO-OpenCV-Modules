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
			LOG("INFO") << "正在配置网络输入/输出 ... " << std::endl;

			this->is_async = isAsync;
			this->confidence_threshold = confidenceThreshold;

			LOG("INFO") << "\t读取 IR 文件: " << modelPath << std::endl;
			cnnNetwork = ie.ReadNetwork(modelPath);
			cnnNetwork.setBatchSize(1);

#pragma region NetowrkIOConfig
			//-------------------- 配置网络层的输入信息 -------------------
			LOG("INFO") << "\t配置网络输入  ... " << std::endl;
			inputsInfo = cnnNetwork.getInputsInfo();
			inputShapes = cnnNetwork.getInputShapes();
			//设置第一层的输入精度为 U8，输入为图像数据 uint8_t
			inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);

			inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
			inputsInfo.begin()->second->setLayout(InferenceEngine::Layout::NHWC);
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

			if (inputsInfo.size() != 1 || inputShapes.begin()->second.size() != 4)
				THROW_IE_EXCEPTION << "通用对象检测模块只支持一个网络层的输入，且输入格式为[BxCxHxW] ... ";

			input_name = inputsInfo.begin()->first;
			input_batch = inputShapes.begin()->second[0];
			input_channels = inputShapes.begin()->second[1];
			input_height = inputShapes.begin()->second[2];
			input_width = inputShapes.begin()->second[3];

			debug_title.str("");
			debug_title << "Input Name:[" << input_name << "]  Shape:[" << input_batch << "x" << input_channels << "x" << input_height << "x" << input_width << "]";

			//------------------------------------------- 配置网络层的输出信息 -----------------------------------
			LOG("INFO") << "\t配置网络输出  ... " << std::endl;
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
				LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << shape.str() << "  Precision:[" << output.second->getPrecision() << "]" << (iter != output_layer_names.end() ? "  \t[√]" : "") << std::endl;
			}

			//一层网络输出
			if (outputsInfo.size() == 1)
			{
				output_name = outputsInfo.begin()->first;
				outputSizeVector = outputsInfo.begin()->second->getTensorDesc().getDims();

				size_t shared_size = 1;
				for (auto& v : outputSizeVector) shared_size *= v;
				shared_layer_infos.clear();
				shared_layer_infos.push_back({ output_name, shared_size * 4 });

				if (outputSizeVector.size() == 4)
				{
					output_max_count = outputSizeVector[2];
					output_object_size = outputSizeVector[3];
				}
				else if (outputSizeVector.size() == 2)
				{
					output_max_count = outputSizeVector[0];
					output_object_size = outputSizeVector[1];
				}
				else
				{
					THROW_IE_EXCEPTION << "网络输出张量维度不支持 " << outputSizeVector.size() << " ... ";
				}
			}
			//多层网络输出
			else
			{
				if (output_layer_names.size() < 2)
					THROW_IE_EXCEPTION << "当前网络模型为多层输出，但未选定网络输出名称 ... ";

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

			//-------------------- 加载网络 -------------------
			LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
			execNetwork = ie.LoadNetwork(cnnNetwork, device);

			LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
			requestPtr = execNetwork.CreateInferRequestPtr();

			//-------------------------配置数据共享---------------------------------------
			LOG("INFO") << "正在创建/映射共享内存块 ... " << std::endl;
			for (const auto &shared:shared_layer_infos)
			{
				LPVOID pBuffer;
				HANDLE pMapFile;
				
				if (CreateOnlyWriteMapFile(pMapFile, pBuffer, shared.second, shared.first.c_str()))
				{
					pBuffers.push_back(pBuffer);
					pMapFiles.push_back(pMapFile);

					//将输出层映射到内存共享
					//重新设置指定输出层 Blob, 将指定输出层数据指向为共享指针，实现该层数据共享
					requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<float>(
						outputsInfo.find(shared.first)->second->getTensorDesc(), (float*)pBuffer));
				}
				else
				{
					LOG("ERROR") << "共享内存块创建失败 ... " << std::endl;
					THROW_IE_EXCEPTION << "共享内存块创建失败 ... ";
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
			LOG("ERROR") << "发生未知/内部异常 ... " << std::endl;
			return false;
		}

		start();

		return true;
	}

	void ObjectDetection::start()
	{
		//设置异步推理完成回调
		//可以读取包含推理结果的输出，并将新输入再次用于重复异步请求
		requestPtr->SetCompletionCallback(
			[&]
			{
				t1 = std::chrono::high_resolution_clock::now();
				total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();
				//std::cout << "\33[2K\r[ INFO] " << "Total Use Time: " << total_use_time;

				//if (is_debug)updateDebugShow();

				requestPtr->StartAsync();
				t0 = std::chrono::high_resolution_clock::now();
			});
	}


	void ObjectDetection::request(const cv::Mat& frame, int frame_idx)
	{
		frame_width = frame.cols;
		frame_height = frame.rows;

		InferenceEngine::Blob::Ptr inputBlobPtr = requestPtr->GetBlob(input_name);
		//matU8ToBlob(frame, inputBlobPtr, frame_idx);

		InferenceEngine::TensorDesc inputTensorDesc = inputsInfo.begin()->second->getTensorDesc();
		InferenceEngine::TensorDesc tDesc(inputTensorDesc.getPrecision(), { 1, 3, frame_height, frame_width }, inputTensorDesc.getLayout());
		requestPtr->SetBlob(input_name, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));

		//requestPtr->Infer();
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();
	}

	void ObjectDetection::matU8ToBlob(const cv::Mat& frame, InferenceEngine::Blob::Ptr& blob, int batchIndex)
	{
		//InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();

		//const size_t width = blobSize[3];
		//const size_t height = blobSize[2];
		//const size_t channels = blobSize[1];

		if (static_cast<uint16_t>(frame.channels()) != input_channels)
			THROW_IE_EXCEPTION << "网络输入和图像的通道数必须匹配 ... ";

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
			THROW_IE_EXCEPTION << "通道数量不受支持 ... ";
		}

		if (is_debug)
			cv::imshow(debug_title.str(), resized_image);
	}

	bool ObjectDetection::getResults(std::vector<ObjectDetection::Result> &results)
	{
		if (!is_debug) return false;
		//if (!has_results) return false;

		has_results = false;
		results.clear();

		//单层输出
		if (outputsInfo.size() == 1)
		{
			const float* buffer = (float*)pBuffers[0];// requestPtr->GetBlob(output_name)->buffer().as<float*>();

			//[1x1xNx7] FP32
			if (outputSizeVector.size() == 4 && outputSizeVector[0] == 1 && outputSizeVector[1] == 1 && outputSizeVector[3] == 7)
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

		//多层输出
		if (shared_layer_infos.size() == 2)
		{
			InferenceEngine::SizeVector outputSize = outputsInfo[shared_layer_infos[0].first]->getTensorDesc().getDims();

			//[Nx5][N]
			//[Nx5] FP32	= float
			//[N]	I32		= int32_t
			if (outputSize.size() == 2 && outputSize.back() == 5)
			{
				//[Nx5]
				const float* buffer1 = (float*)pBuffers[1];
				//const float* buffer0 = requestPtr->GetBlob(shared_layer_infos[0].first)->buffer().as<float*>();
				//[N]
				const float* buffer0 = (float*)pBuffers[1];
				//const float* buffer0 = requestPtr->GetBlob(shared_layer_infos[0].first)->buffer().as<float*>();

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
			LOG("WARN") << "调试模式下未分析网络输出数据，output layouts: " << outputsInfo.size() << std::endl;
		}

		return true;
	}

}
