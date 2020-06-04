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

	void ObjectDetection::SetParameters(double confidence_threshold, const std::vector<std::string> labels)
	{
		this->labels = labels;
		this->confidence_threshold = confidence_threshold;
	}

	/// <summary>
	/// 绘制检测对象的边界
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="results"></param>
	/// <param name="labels"></param>
	void ObjectDetection::DrawObjectBound(cv::Mat frame, const std::vector<ObjectDetection::Result>& results, const std::vector<std::string>& labels)
	{
		size_t length = results.size();
		if (length <= 0) return;

		std::stringstream txt;
		cv::Scalar border_color(0, 255, 0);

		float scale = 0.2f;
		int shift = 0;
		int thickness = 2;
		int lineType = cv::LINE_AA;

		std::string label;
		size_t label_length = labels.size();

		for (const auto& result : results)
		{
			cv::Rect rect = ScaleRectangle(result.location);
			cv::rectangle(frame, rect, border_color, 1);

			cv::Point h_width = cv::Point(rect.width * scale, 0);   //水平宽度
			cv::Point v_height = cv::Point(0, rect.height * scale); //垂直高度

			cv::Point left_top(rect.x, rect.y);
			cv::line(frame, left_top, left_top + h_width, border_color, thickness, lineType, shift);     //-
			cv::line(frame, left_top, left_top + v_height, border_color, thickness, lineType, shift);    //|

			cv::Point left_bottom(rect.x, rect.y + rect.width - 1);
			cv::line(frame, left_bottom, left_bottom + h_width, border_color, thickness, lineType, shift);       //-
			cv::line(frame, left_bottom, left_bottom - v_height, border_color, thickness, lineType, shift);      //|

			cv::Point right_top(rect.x + rect.width - 1, rect.y);
			cv::line(frame, right_top, right_top - h_width, border_color, thickness, lineType, shift);   //-
			cv::line(frame, right_top, right_top + v_height, border_color, thickness, lineType, shift);  //|

			cv::Point right_bottom(rect.x + rect.width - 1, rect.y + rect.height - 1);
			cv::line(frame, right_bottom, right_bottom - h_width, border_color, thickness, lineType, shift);   //-
			cv::line(frame, right_bottom, right_bottom - v_height, border_color, thickness, lineType, shift);  //|

			txt.str("");
			label = (label_length <= 0 || result.label < 0 || result.label >= label_length) ?
				std::to_string(result.label) : labels[result.label];
			txt << std::setprecision(3) << "Label:" << label << "  Conf:" << result.confidence;
			cv::putText(frame, txt.str(), cv::Point(left_top.x, left_top.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
		}
	}

	void ObjectDetection::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
	{
		static InferenceEngine::SizeVector outputShape = outputsInfo.find(shared_output_layers.begin()->first)->second->getTensorDesc().getDims();

		//单层输出
		if (shared_output_layers.size() == 1)
		{
			// 三种方式都可获取到数据
			//const float* buffer = (float*)shared_output_layers.begin()->second;
			//const float* buffer = requestPtr->GetBlob(shared_output_layers.begin()->first)->buffer().as<float*>();
			
			InferenceEngine::Blob::Ptr data;
			request->GetBlob(shared_output_layers.begin()->first.c_str(), data, 0);
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

					rt.location.x = static_cast<int>(buffer[offset + 3] * debug_frame.cols);
					rt.location.y = static_cast<int>(buffer[offset + 4] * debug_frame.rows);
					rt.location.width = static_cast<int>(buffer[offset + 5] * debug_frame.cols - rt.location.x);
					rt.location.height = static_cast<int>(buffer[offset + 6] * debug_frame.rows - rt.location.y);

					results.push_back(rt);
				}
			}
		}
		//多层输出
		else if (shared_output_layers.size() == 2)
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

				//[Nx5]  三种方式都可获取到数据
				//const float* buffer0 = (float*)shared_output_layers[0].second;		//第一种
				//const float* buffer0 = requestPtr->GetBlob(shared_output_layers[0].first)->buffer().as<float*>();	//第二种
				InferenceEngine::Blob::Ptr data0;										//第三种
				request->GetBlob(shared_output_layers[0].first.c_str(), data0, 0);
				const float* buffer0 = data0->buffer().as<float*>();

				//[N]
				//const int32_t* buffer1 = (int32_t*)shared_output_layers[1].second;		
				//const int32_t* buffer1 = requestPtr->GetBlob(shared_output_layers[1].first)->buffer().as<int32_t*>();
				InferenceEngine::Blob::Ptr data1;
				request->GetBlob(shared_output_layers[1].first.c_str(), data1, 0);
				const int32_t* buffer1 = data1->buffer().as<int32_t*>();

				results.clear();
				for (int i = 0; i < max_count; i++)
				{
					Result rt;
					rt.label = buffer1[i];
					if (rt.label < 0) break;

					int offset = i * object_size;
					rt.confidence = buffer0[offset + 4];
					if (rt.confidence <= confidence_threshold) continue;

					rt.location.x = static_cast<int>(buffer0[offset + 0] / input_width * debug_frame.cols);
					rt.location.y = static_cast<int>(buffer0[offset + 1] / input_height * debug_frame.rows);
					rt.location.width = static_cast<int>(buffer0[offset + 2] / input_width * debug_frame.cols - rt.location.x);
					rt.location.height = static_cast<int>(buffer0[offset + 3] / input_height * debug_frame.rows - rt.location.y);

					results.push_back(rt);
				}
			}
		}

		if (is_debug)UpdateDebugShow();
	}
	
	void ObjectDetection::UpdateDebugShow()
	{
		DrawObjectBound(debug_frame, results, labels);

		std::stringstream use_time;
		use_time << "Detection Count:" << results.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
		cv::putText(debug_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

		cv::imshow(debug_title.str(), debug_frame);
		cv::waitKey(1);
	}
}
