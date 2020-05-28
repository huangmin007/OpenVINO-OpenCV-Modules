#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_detection.hpp"


namespace space
{
	ObjectDetection::ObjectDetection(bool isDebug):
		is_debug(isDebug),
		confidence_threshold(0.5)
	{};
	ObjectDetection::~ObjectDetection() 
	{
		//if(pFile != NULL) 
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
			LOG("INFO") << "正在配置网络输入/输出 ... " << modelPath << std::endl;

			this->is_async = isAsync;
			this->confidence_threshold = confidenceThreshold;

			LOG("INFO") << "\t读取 IR 文件 ... " << std::endl;
			cnnNetwork = ie.ReadNetwork(modelPath);
			cnnNetwork.setBatchSize(1);

			//-------------------配置网络----------------------
			networkIOConfig();

			//-------------------- 加载网络 -------------------
			LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
			execNetwork = ie.LoadNetwork(cnnNetwork, device);

			LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
			requestPtrCurr = execNetwork.CreateInferRequestPtr();
			requestPtrNext = execNetwork.CreateInferRequestPtr();
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

		return true;
	}

	void ObjectDetection::networkIOConfig()
	{
		//-------------------- 配置网络层的输入信息 -------------------[BxCxHxW] name:"input",shape: [1x3x320x544] 
		LOG("INFO") << "\t配置网络输入  ... " << std::endl;
		inputsInfo = cnnNetwork.getInputsInfo();
		inputShapes = cnnNetwork.getInputShapes();
		if (inputsInfo.size() == 1)
		{
			inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		}
		else
		{

			for (auto& input : inputsInfo)
				input.second->setPrecision(InferenceEngine::Precision::U8);
		}
		input_name = inputsInfo.begin()->first;
		input_batch = static_cast<uint16_t>(getTensorBatch(inputsInfo.begin()->second->getTensorDesc()));
		input_channels = static_cast<uint16_t>(getTensorChannels(inputsInfo.begin()->second->getTensorDesc()));
		input_height = static_cast<uint16_t>(getTensorHeight(inputsInfo.begin()->second->getTensorDesc()));
		input_width = static_cast<uint16_t>(getTensorWidth(inputsInfo.begin()->second->getTensorDesc()));

		if (is_debug)
		{
			debug_title << "Input Name:"<< input_name << "  Shape:[" << input_batch << "x" << input_channels << "x" << input_width + "x" + input_height << "]\0";
			LOG("INFO") << debug_title.str() << std::endl;
		}
		for (auto& shape : inputShapes)
		{
			std::stringstream size; size << "[";
			for (int i = 0; i < shape.second.size(); i++)
				size << shape.second[i] << (i != shape.second.size() - 1 ? "x" : "]");

			LOG("INFO") << "\tInput Name:" << shape.first << "   Shape:" << size.str() << std::endl;
		}

		//-------------------- 配置网络层的输出信息 -------------------
		LOG("INFO") << "\t配置网络输出  ... " << std::endl;
		outputsInfo = cnnNetwork.getOutputsInfo();
		if (outputsInfo.size() == 1)
		{
			outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
			outputsInfo.begin()->second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(outputsInfo.begin()->second->getDims()));
		}
		else
		{
			for (auto& output : outputsInfo)
				output.second->setPrecision(InferenceEngine::Precision::FP32);
		}

		outputShapes = outputsInfo.begin()->second->getTensorDesc().getDims();
		output_name = outputsInfo.begin()->first;
		output_max_count = outputShapes[2];
		output_object_size = outputShapes[3];
		LOG("INFO") << "\tOutput name:" << output_name << " shape:["
			<< outputShapes[0] << "x" << outputShapes[1] << "x" << outputShapes[2] << "x" << outputShapes[3] << "]" << std::endl;

		//-------------------------配置数据共享---------------------------------------

	}

	//void ObjectDetection::createOutputShared()
	//{

	//}

	void ObjectDetection::request(const cv::Mat& frame, int frame_idx)
	{
		//if (frame_count >= 16)
		//{
		//	LOG("WARN") << "队列帧总数量，不得超出 16 帧" << std::endl;
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

		//results.clear();
		//const float* buffer = requestPtrCurr->GetBlob(output_name)->buffer().as<float*>();
		//std::cout << "Result::" << buffer << std::endl;
		/*
		//[1x1x200x7]
    for (int i = 0; i < output_max_count && output_object_size == 7; i++)
    {
        float image_id = buffer[i * output_object_size + 0];
        if (image_id < 0) break;

        int offset = i * output_object_size;

        Result rt;
        rt.label = buffer[offset + 1];
        rt.confidence = std::min(std::max(0.0f, buffer[offset + 2]), 1.0f);

        if (rt.confidence <= confidence_threshold) continue;

        rt.location.x = static_cast<int>(buffer[offset + 3] * frame_width);
        rt.location.y = static_cast<int>(buffer[offset + 4] * frame_height);
        rt.location.width = static_cast<int>(buffer[offset + 5] * frame_width - rt.location.x);
        rt.location.height = static_cast<int>(buffer[offset + 6] * frame_height - rt.location.y);

        results.push_back(rt);
    }
		*/

		return true;
	}

}
