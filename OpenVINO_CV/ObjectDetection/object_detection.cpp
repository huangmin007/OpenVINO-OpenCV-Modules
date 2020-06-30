#include "pch.h"
#include <map>
#include <iostream>
#include "static_functions.hpp"
#include "object_detection.hpp"


namespace space
{
	ObjectDetection::ObjectDetection(bool is_show):is_show(is_show){};
	
	ObjectDetection::~ObjectDetection()
	{
		DisposeSharedMapping(shared_layers);
	};
	
	void ObjectDetection::ConfigNetwork(const Params& params)
	{
		this->params = params;
		this->params.Check();
		this->params.UpdateIEConfig();

		LOG("INFO") << this->params << std::endl;
		
		
		SetNetworkIO();

		//4.
		LOG("INFO") << "正在配置异步推断回调 ... " << std::endl;
		SetInferCallback();	//设置异步回调

		//5.
		if (this->params.is_mapping_blob)
		{
			LOG("INFO") << "正在创建/映射共享内存 ... " << std::endl;
			shared_layers = CreateSharedBlobs(requestPtrs, this->params.output_layers, SHARED_RESERVE_BYTE_SIZE);
		}

		is_configuration = true;
		LOG("INFO") << "网络模型配置完成 ... " << std::endl;
		title << "[Object Detection] Output Size:" << outputsInfo.size() << "  Name:[" << outputsInfo.begin()->first << "]  Shape:" << outputsInfo.begin()->second->getTensorDesc().getDims() << std::endl;
	}


	void ObjectDetection::SetNetworkIO()
	{
		LOG("INFO") << "正在配置网络模型 ... " << std::endl;
		// 1
		LOG("INFO") << "\t读取 IR 文件: " << this->params.model << " [" << this->params.device << "]" << std::endl;
		cnnNetwork = this->params.ie.ReadNetwork(this->params.model);
		cnnNetwork.setBatchSize(this->params.batch_count);

		LOG("INFO") << "\t配置网络输入 ... Reshape:[" << std::boolalpha << this->params.is_reshape << "]" << std::endl;
		inputsInfo = cnnNetwork.getInputsInfo();
		inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
		inputsInfo.begin()->second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
		inputsInfo.begin()->second->setLayout(this->params.is_reshape ? InferenceEngine::Layout::NHWC : InferenceEngine::Layout::NCHW);

		//print inputs info
		for (const auto& input : inputsInfo)
		{
			InferenceEngine::SizeVector inputDims = input.second->getTensorDesc().getDims();
			LOG("INFO") << "\tInput Name:[" << input.first << "]  Shape:" << inputDims << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
		}

		//默认支持一层输入，子类可以更改为多层输入
		//多层输入以输入层名.bin文件指向
		//那bin的数据类型，精度怎么解释呢？
		if (inputsInfo.size() != 1)
		{
			LOG("WARN") << "当前模块只支持一个网络层的输入 ... " << std::endl;
			throw std::logic_error("当前模块只支持一个网络层的输入 ... ");
		}

		LOG("INFO") << "\t配置网络输出  ... " << std::endl;
		outputsInfo = cnnNetwork.getOutputsInfo();
		outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
		if (this->params.output_layers.size() == 0)
		{
			for (const auto& output : outputsInfo)
				this->params.output_layers.push_back(output.first);
		}
		//print outputs info
		for (const auto& output : outputsInfo)
		{
			InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
			std::vector<std::string>::iterator iter = find(this->params.output_layers.begin(), this->params.output_layers.end(), output.first);
			LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << outputDims << "  Precision:[" << output.second->getPrecision() << "]" <<
				(iter != this->params.output_layers.end() ? (this->params.is_mapping_blob ? "  \t[√]共享" : "  \t[×]共享") : "") << std::endl;
		}

		//3.
#pragma region 创建可执行网络
		LOG("INFO") << "正在创建可执行网络 ... " << std::endl;
		if (this->params.ie_config.size() > 0)
			LOG("INFO") << "\t" << this->params.ie_config.begin()->first << "=" << this->params.ie_config.begin()->second << std::endl;
		execNetwork = this->params.ie.LoadNetwork(cnnNetwork, this->params.device, this->params.ie_config);
		LOG("INFO") << "正在创建推断请求对象 ... " << std::endl;
		for (int i = 0; i < this->params.infer_count; i++)
		{
			InferenceEngine::InferRequest::Ptr inferPtr = execNetwork.CreateInferRequestPtr();
			requestPtrs.push_back(inferPtr);
		}
#pragma endregion
	}

	void ObjectDetection::SetInferCallback()
	{
		for (auto& requestPtr : requestPtrs)
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

					//1.解析数据，外部解析 OR 内部解析
					ParsingOutputData(request, code);

					//2.循环再次启动异步推断
					if (this->params.auto_request_infer)
					{
						if (!this->params.is_reshape)
						{
							InferenceEngine::Blob::Ptr inputBlob;
							request->GetBlob(inputsInfo.begin()->first.c_str(), inputBlob, 0);
							MatU8ToBlob<uint8_t>(frame_ptr, inputBlob);
						};

						//控制帧频 在 30 帧内
						if (WaitKeyDelay - infer_use_time > 0)
							Sleep(WaitKeyDelay - infer_use_time);

						request->SetBatch(this->params.batch_count, 0);
						request->StartAsync(0);
						t0 = std::chrono::high_resolution_clock::now();
					}

					//3.更新显示结果
					if (is_show) UpdateDebugShow();
				});
		}

		timer.start(1000, [&]
			{
				std::lock_guard<std::shared_mutex> lock(_fps_mutex);
				fps = fps_count;
				fps_count = 0;
			});
	}


	void ObjectDetection::RequestInfer(const cv::Mat& frame)
	{
		if (!is_configuration)
			throw std::logic_error("未配置网络对象，不可操作 ...");
		if (frame.empty() || frame.type() != CV_8UC3)
			throw std::logic_error("输入图像帧不能为空帧对象，CV_8UC3 类型 ... ");

		frame_ptr = frame;
		InferenceEngine::StatusCode code;
		InferenceEngine::InferRequest::Ptr requestPtr;

		//查找没有启动的推断对象
		for (auto& request : requestPtrs)
		{
			code = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
			if (this->params.auto_request_infer)
			{
				if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED)
				{
					requestPtr = request;
					LOG("INFO") << "找到一个 InferRequestPtr 对象，Input Address:" << requestPtr->GetBlob(inputsInfo.begin()->first)->buffer().as<void*>() << std::endl;

					break;
				}
			}
			else
			{
				if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED
					|| code == InferenceEngine::StatusCode::OK
					)
				{
					requestPtr = request;
					break;
				}
			}
		}
		if (requestPtr == nullptr) return;

		InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
		//LOG("INFO") << "找到一个 InferRequestPtr 对象，Input Address:" << (void*)inputBlob->buffer().as<uint8_t*>() << std::endl;

		if (this->params.is_reshape)
		{
			MrapMat2Blob(requestPtr, inputsInfo.begin()->first, frame);
		}
		else
		{
			MatU8ToBlob<uint8_t>(frame, inputBlob);
		}

		//if(this->params.batch_count > 1)
		//	requestPtr->SetBatch(this->params.batch_count);

		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();
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
		PasingO2DNx5(request);
		PasingO1D1x1xNx7(request);
	}

	void ObjectDetection::PasingO1D1x1xNx7(InferenceEngine::IInferRequest::Ptr request)
	{
		if (outputsInfo.size() != 1) return;
		static InferenceEngine::SizeVector outputShape = outputsInfo.begin()->second->getTensorDesc().getDims();
		if (outputShape.size() != 4 || outputShape[0] != 1 || outputShape[1] != 1 || outputShape[3] != 7) return;

		InferenceEngine::Blob::Ptr data;
		request->GetBlob(outputsInfo.begin()->first.c_str(), data, 0);
		const float* buffer = data->buffer().as<float*>();

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

			if (rt.confidence <= this->params.confidence) continue;

			int x = static_cast<int>(buffer[offset + 3] * frame_ptr.cols);
			int y = static_cast<int>(buffer[offset + 4] * frame_ptr.rows);
			int width = static_cast<int>(buffer[offset + 5] * frame_ptr.cols - x);
			int height = static_cast<int>(buffer[offset + 6] * frame_ptr.rows - y);

			rt.location = ScaleRectangle(cv::Rect(x, y, width, height), scaling, 1.0, 1.0);

			results.push_back(rt);
		}
	}
	void ObjectDetection::PasingO2DNx5(InferenceEngine::IInferRequest::Ptr request)
	{
		if (outputsInfo.size() != 2) return;
		static InferenceEngine::SizeVector outputShape = outputsInfo.begin()->second->getTensorDesc().getDims();
		if (outputShape.size() != 2 || outputShape.back() != 5) return;

		const InferenceEngine::SizeVector inputShape = inputsInfo.begin()->second->getTensorDesc().getDims();

		size_t max_count = outputShape[0];
		size_t object_size = outputShape[1];

		size_t input_height = inputShape[2];
		size_t input_width = inputShape[3];

		//[Nx5]
		InferenceEngine::Blob::Ptr data0;
		request->GetBlob(std::get<0>(shared_layers[0]).c_str(), data0, 0);
		const float* buffer0 = data0->buffer().as<float*>();

		//[N]
		InferenceEngine::Blob::Ptr data1;
		request->GetBlob(std::get<0>(shared_layers[1]).c_str(), data1, 0);
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
			if (rt.confidence <= this->params.confidence) continue;

			int x = static_cast<int>(buffer0[offset + 0] / input_width * frame_ptr.cols);
			int y = static_cast<int>(buffer0[offset + 1] / input_height * frame_ptr.rows);
			int width = static_cast<int>(buffer0[offset + 2] / input_width * frame_ptr.cols - x);
			int height = static_cast<int>(buffer0[offset + 3] / input_height * frame_ptr.rows - y);

			rt.location = ScaleRectangle(cv::Rect(x, y, width, height), scaling, 1.0, 1.0);

			results.push_back(rt);
		}
	}
	
	void ObjectDetection::UpdateDebugShow()
	{
		//需要拷贝，不能破坏原有内存图像数据
		frame_ptr.copyTo(show_frame);

		static bool debug_track = false;
		if (!debug_track)
		{
			cv::namedWindow(title.str());
			int initValue = static_cast<int>(scaling * 100);

			cv::createTrackbar("Scaling", title.str(), &initValue, 150,
				[](int position, void* blendingPtr)
				{
					*static_cast<float*>(blendingPtr) = position * 0.01f;
				},
				&scaling);

			debug_track = true;
		}

		DrawObjectBound(show_frame, results, this->params.labels);

		std::stringstream use_time;
		use_time << "Detection Count:" << results.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
		cv::putText(show_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

		cv::imshow(title.str(), show_frame);
		cv::waitKey(1);
	}

}
