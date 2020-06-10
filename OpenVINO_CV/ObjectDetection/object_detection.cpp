#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_detection.hpp"


namespace space
{
	ObjectDetection::ObjectDetection(const std::vector<std::string>& output_layers_name, bool is_debug)
		:OpenModelInferBase(output_layers_name, is_debug), confidence_threshold(0.5){};
	
	ObjectDetection::~ObjectDetection(){};

	void ObjectDetection::SetParameters(const OpenModelInferBase::Params &params, double confidence_threshold, const std::vector<std::string> labels)
	{
		OpenModelInferBase::SetParameters(params);

		this->labels = labels;
		this->confidence_threshold = confidence_threshold;
	}

	/// <summary>
	/// 绘制检测对象的边界，以及标签
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="results"></param>
	/// <param name="labels"></param>
	void ObjectDetection::DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string>& labels)
	{
		size_t length = results.size();
		if (length <= 0) return;

		std::string label;
		std::stringstream txt;
		size_t label_length = labels.size();

		for (const auto& result : results)
		{
			space::DrawObjectBound(frame, result.location);

			txt.str("");
			label = (label_length <= 0 || result.label < 0 || result.label >= label_length) ?
				std::to_string(result.label) : labels[result.label];
			txt << std::setprecision(3) << "BID:" << result.id << "  Label:" << label << "  Conf:" << result.confidence;

			cv::putText(frame, txt.str(), cv::Point(result.location.x, result.location.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
		}
	};

	
	void ObjectDetection::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
	{
		static InferenceEngine::SizeVector outputShape = outputsInfo.begin()->second->getTensorDesc().getDims();

		//单层输出
		if (output_shared_layers.size() == 1)
		{
			InferenceEngine::Blob::Ptr data;
			request->GetBlob(output_shared_layers.begin()->first.c_str(), data, 0);
			const float* buffer = data->buffer().as<float*>();

			//[1x1xNx7] FP32
			if (outputShape.size() == 4 && outputShape[0] == 1 && outputShape[1] == 1 && outputShape[3] == 7)
			{
				//需要拷贝，不能破坏原有内存图像数据
				frame_ptr.copyTo(debug_frame);

				size_t max_count = outputShape[2];
				size_t object_size = outputShape[3];
				
				results.clear();
				for (int i = 0; i < max_count; i++)
				{
					int offset = i * object_size;
					float batch_id = buffer[offset];
					if (batch_id < 0) break;

					Result rt;
					rt.id = static_cast<int>(batch_id);
					rt.label = static_cast<int>(buffer[offset + 1]);
					rt.confidence = std::min(std::max(0.0f, buffer[offset + 2]), 1.0f);

					if (rt.confidence <= confidence_threshold) continue;

					int x = static_cast<int>(buffer[offset + 3] * debug_frame.cols);
					int y = static_cast<int>(buffer[offset + 4] * debug_frame.rows);
					int width = static_cast<int>(buffer[offset + 5] * debug_frame.cols - x);
					int height = static_cast<int>(buffer[offset + 6] * debug_frame.rows - y);
					
					rt.location = ScaleRectangle(cv::Rect(x, y, width, height), 1.2, 1.0, 1.0);

					results.push_back(rt);
				}
			}
			else if (outputShape.size() == 4 && outputShape[1] == 1001)
			{
				int maxIdx = 0;
				float maxP = 0.0;
				int nclasses = 1001;
				float sum = 1.0;

				for (int i = 0; i < nclasses; i++)
				{
					float e = exp(buffer[i]);
					sum = sum + e;
					if (e > maxP)
					{
						maxP = e;
						maxIdx = i;
					}
				}

				float rtp = maxP / sum;
				std::cout << "ID:" << maxIdx << " Cof" << rtp << std::endl;
			}
		}
		//多层输出
		else if (output_shared_layers.size() == 2)
		{
			//[Nx5][N]
			//[Nx5] FP32	= float
			//[N]	I32		= int32_t
			if (outputShape.size() == 2 && outputShape.back() == 5)
			{
				//需要拷贝，不能破坏原有内存图像数据
				frame_ptr.copyTo(debug_frame);

				const InferenceEngine::SizeVector inputShape = inputsInfo.begin()->second->getTensorDesc().getDims();

				size_t max_count = outputShape[0];
				size_t object_size = outputShape[1];

				size_t input_height = inputShape[2];
				size_t input_width = inputShape[3];

				//[Nx5]
				InferenceEngine::Blob::Ptr data0;										
				request->GetBlob(output_shared_layers[0].first.c_str(), data0, 0);
				const float* buffer0 = data0->buffer().as<float*>();

				//[N]
				InferenceEngine::Blob::Ptr data1;
				request->GetBlob(output_shared_layers[1].first.c_str(), data1, 0);
				const int32_t* buffer1 = data1->buffer().as<int32_t*>();

				results.clear();
				for (int i = 0; i < max_count; i++)
				{
					Result rt;
					rt.id = i;
					rt.label = buffer1[i];
					if (rt.label < 0) break;

					int offset = i * object_size;
					rt.confidence = buffer0[offset + 4];
					if (rt.confidence <= confidence_threshold) continue;

					int x = static_cast<int>(buffer0[offset + 0] / input_width * debug_frame.cols);
					int y = static_cast<int>(buffer0[offset + 1] / input_height * debug_frame.rows);
					int width = static_cast<int>(buffer0[offset + 2] / input_width * debug_frame.cols - x);
					int height = static_cast<int>(buffer0[offset + 3] / input_height * debug_frame.rows - y);

					rt.location = ScaleRectangle(cv::Rect(x, y, width, height), 1.2, 1.0, 1.0);

					results.push_back(rt);
				}
			}
		}

		for (auto& network : sub_network)
			network->RequestInfer(frame_ptr, results);

		if (is_debug)UpdateDebugShow();
	}
	
	void ObjectDetection::UpdateDebugShow()
	{
		if (debug_frame.empty()) return;

		DrawObjectBound(debug_frame, results, labels);

		std::stringstream use_time;
		use_time << "Detection Count:" << results.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
		cv::putText(debug_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

		cv::imshow(debug_title.str(), debug_frame);
		cv::waitKey(1);
	}

	void ObjectDetection::AddSubNetwork(ObjectRecognition* object_recognition)
	{
		this->sub_network.push_back(object_recognition);
	};
}
