
#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_segmentation.hpp"

#include "samples/common.hpp"
#include <samples/ocv_common.hpp>

namespace space
{
	ObjectSegmentation::ObjectSegmentation(bool is_debug) :OpenModelInferBase(is_debug){};
	ObjectSegmentation::ObjectSegmentation(const std::vector<std::string>& output_layer_names, bool is_debug) :OpenModelInferBase(output_layer_names, is_debug){};
	ObjectSegmentation::~ObjectSegmentation()
	{
	};

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

		int frame_width = frame_ptr.cols;
		int frame_height = frame_ptr.rows;

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

			cv::resize(mask_img, resize_img, cv::Size(frame_width, frame_height));
			resize_img = frame_ptr * blending + resize_img * (1 - blending);

			std::stringstream use_time;
			use_time << "Infer Use Time:" << GetInferUseTime() << "ms";
			cv::putText(resize_img, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

			cv::imshow(debug_title.str(), resize_img);
			cv::waitKey(1);

			return;
		}
	}
	
#if backup
	void ObjectSegmentation::start(const cv::Mat& frame)
	{
		if (CV_8UC3 != frame.type())
			throw std::runtime_error("输入图片应支持 CV_8UC3 类型");

		frame_height = frame.size().height;
		frame_width = frame.size().width;

		size_t channels = frame.channels();
		size_t strideH = frame.step.buf[0];
		size_t strideW = frame.step.buf[1];

		bool is_dense = strideW == channels && strideH == channels * frame_width;
		if (!is_dense) THROW_IE_EXCEPTION << "不支持从非稠密 cv::Mat 转换";

		//重新设置输入层 Blob，将输入图像内存数据指针共享给输入层，做实时或异步推断
		//输入为图像原始尺寸，其实是会存在问题的，如果源图尺寸很大，那处理时间会更长
		//InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8, { 1, channels, frame_height, frame_width }, InferenceEngine::Layout::NHWC);
		//requestPtr->SetBlob(input_name, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));
		InferenceEngine::TensorDesc inputTensorDesc = inputsInfo.begin()->second->getTensorDesc();
		InferenceEngine::TensorDesc tDesc(inputTensorDesc.getPrecision(), { 1, channels, frame_height, frame_width }, inputTensorDesc.getLayout());
		requestPtr->SetBlob(input_name, InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data));

		//设置异步推理完成回调
		//可以读取包含推理结果的输出，并将新输入再次用于重复异步请求
		requestPtr->SetCompletionCallback(
			[&]
			{
				t1 = std::chrono::high_resolution_clock::now();
				total_use_time = std::chrono::duration_cast<ms>(t1 - t0).count();				
				//std::cout << "\33[2K\r[ INFO] " << "Total Use Time: " << total_use_time;

				updateDisplay();

				//InferenceEngine::StatusCode code = requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
				//std::cout << "complete... " << code << std::endl;

				requestPtr->StartAsync();
				t0 = std::chrono::high_resolution_clock::now();
			});

		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();

		//InferenceEngine::StatusCode code = requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
		//std::cout << "Code::" << code << std::endl;
	}

	double ObjectSegmentation::getUseTime() const
	{
		return total_use_time;
	}

	void ObjectSegmentation::updateDisplay()
	{
		if (!is_debug) return;
		
		//单层输出
		//[B,H,W] = [1,H,W]
		//[B,C,H,W] = [1,3,H,W]
		if (outputsInfo.size() == 1)
		{
			int width, height, channels;
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

			const float* buffer = (float*)pBuffers[0];

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

			cv::resize(mask_img, resize_img, cv::Size(frame_width, frame_height));
			//resize_img = resize_img * 0.1f;
			
			cv::imshow("Debug Output", resize_img);
			cv::waitKey(1);

			return;
		}
#if 后面在处理
		//多层输出
		if (shared_layer_infos.size() == 2)
		{
			//if (outputSizeVector.size() == 3 && outputSizeVector.back() == 5)
			//{
				//const float* buffer0 = (float*)pBuffers[0];
				//const float* buffer1 = (float*)pBuffers[0];

				//return;
			//}

			return;
		}
		else
		{
			LOG("WARN") << "调试模式下未分析网络输出数据，output layouts: " << outputsInfo.size() << std::endl;
		}
#endif
		return;
	}
#endif
}
