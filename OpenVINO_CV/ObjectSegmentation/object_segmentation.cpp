
#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_segmentation.hpp"

#include "samples/common.hpp"
#include <samples/ocv_common.hpp>

namespace space
{
	ObjectSegmentation::ObjectSegmentation(const std::vector<std::string>& output_layer_names, bool is_debug) 
		:OpenModelInferBase(output_layer_names, is_debug){};
	ObjectSegmentation::~ObjectSegmentation()
	{
	};

	void ObjectSegmentation::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
	{
		//单层输出
		//[B,H,W] = [1,H,W]
		//[B,C,H,W] = [1,3,H,W]
		if (shared_output_layers.size() == 1)
		{
			int width, height, channels;
			InferenceEngine::SizeVector outputSizeVector = outputsInfo.begin()->second->getTensorDesc().getDims();

			if (outputSizeVector.size() == 3)
			{
				channels = 0;
				height = outputSizeVector[1];
				width = outputSizeVector[2];
			}
			else if (outputSizeVector.size() == 4)
			{
				channels = outputSizeVector[1];
				height = outputSizeVector[2];
				width = outputSizeVector[3];
			}
			else
			{
				return;
			}

			const float* buffer = (float*)shared_output_layers.begin()->second;

			if (mask_img.empty())
				mask_img.create(height, width, CV_8UC3);

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					//类别 id
					size_t classId = 0;
					if (channels == 0)
					{
						classId = static_cast<size_t>(buffer[h * width + w]);
					}
					else
					{
						float maxProb = -1.0f;
						//在多个维度的数据中，找出最大值概率值对应的通道索引
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

					//现有的颜色表数量不够，添加几个
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

			if (is_debug) UpdateDebugShow();
		}
	}

	void ObjectSegmentation::UpdateDebugShow()
	{
		static bool debug_track = false;
		if (!debug_track)
		{
			cv::namedWindow(debug_title.str());
			int initValue = static_cast<int>(blending * 100);

			cv::createTrackbar("blending", debug_title.str(), &initValue, 100,
				[](int position, void* blendingPtr)
				{
					*static_cast<float*>(blendingPtr) = position * 0.01f;
				},
				&blending);

			debug_track = true;
		}

		//int frame_width = frame_ptr.cols;
		//int frame_height = frame_ptr.rows;

		cv::resize(mask_img, resize_img, frame_ptr.size());// cv::Size(frame_width, frame_height));
		resize_img = frame_ptr * blending + resize_img * (1 - blending);

		std::stringstream use_time;
		use_time << "Infer Use Time:" << GetInferUseTime() << "ms" << "  FPS:" << GetFPS();
		cv::putText(resize_img, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

		cv::imshow(debug_title.str(), resize_img);
		cv::waitKey(1);

	}
}
