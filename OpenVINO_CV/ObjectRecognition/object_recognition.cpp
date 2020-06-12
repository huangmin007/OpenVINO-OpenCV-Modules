#include "object_recognition.hpp"
#include <static_functions.hpp>
#include <samples/ocv_common.hpp>
namespace space
{
	ObjectRecognition::ObjectRecognition(const std::vector<std::string>& input_shared, const std::vector<std::string>& output_layers, bool is_debug)
		:OpenModelInferBase(output_layers, is_debug),
		input_shared(input_shared)
	{
	}
	ObjectRecognition::~ObjectRecognition()
	{
		for (auto& shared : input_shared_layers)
		{
			UnmapViewOfFile(shared.second);
			shared.second = NULL;
		}
		for (auto& handle : input_shared_handle)
		{
			CloseHandle(handle);
			handle = NULL;
		}

		input_shared_handle.clear();
		input_shared_layers.clear();
	}

	void ObjectRecognition::RequestInfer(const cv::Mat& frame)
	{
		frame_ptr = frame;

		std::vector<ObjectResult> object_results;
		if (input_shared_layers.size() == 1)
		{
			const float* buffer = (float*)input_shared_layers.begin()->second + SHARED_RESERVE_BYTE_SIZE;

			size_t max_count = 200;
			size_t object_size = 7;
			for (int i = 0; i < max_count; i++)
			{
				int offset = i * object_size;
				float batch_id = buffer[offset];
				if (batch_id < 0) break;

				ObjectResult rt;
				rt.id = static_cast<int>(batch_id);
				rt.label = static_cast<int>(buffer[offset + 1]);
				rt.confidence = std::min(std::max(0.0f, buffer[offset + 2]), 1.0f);

				if (rt.confidence <= 0.5f) continue;

				int x = static_cast<int>(buffer[offset + 3] * frame_ptr.cols);
				int y = static_cast<int>(buffer[offset + 4] * frame_ptr.rows);
				int width = static_cast<int>(buffer[offset + 5] * frame_ptr.cols - x);
				int height = static_cast<int>(buffer[offset + 6] * frame_ptr.rows - y);

				rt.location = ScaleRectangle(cv::Rect(x, y, width, height), 1.2, 1.0, 1.0);

				object_results.push_back(rt);
			}
		}

		if (object_results.size() == 0) return;

		InferenceEngine::StatusCode code;
		InferenceEngine::InferRequest::Ptr requestPtr;
		//查找没有启动的推断对象
		for (auto& request : requestPtrs)
		{
			code = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
			if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED || code == InferenceEngine::StatusCode::OK)
			{
				requestPtr = request;
				break;
			}
		}
		if (requestPtr == nullptr) return;

		frame.copyTo(prev_frame);
		this->results = object_results;

		InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);
		//const size_t batch_size = inputBlob->getTensorDesc().getDims()[0];

		auto count = std::min(results.size(), 4ULL);

		debug_frame = frame_ptr.clone();
		std::vector<ObjectResult> results = this->results;

		for (int i = 0; i < count; i++)
		{
			auto clipRect = results[i].location & cv::Rect(0, 0, frame.cols, frame.rows);

			cv::Mat roi = prev_frame(clipRect);
			matU8ToBlob<uint8_t>(roi, inputBlob, i);
		}
		requestPtr->SetBatch(count + 1);
		requestPtr->StartAsync();
		//requestPtr->Infer();
		t0 = std::chrono::high_resolution_clock::now();
	}

	void ObjectRecognition::SetInferCallback()
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

					ParsingOutputData(request, code);

					if (is_debug) UpdateDebugShow();
				});
		}
		timer.start(1000, [&]
			{
				std::lock_guard<std::shared_mutex> lock(_fps_mutex);
				fps = fps_count;
				fps_count = 0;
			});
	}
	void ObjectRecognition::SetMemoryShared()
	{
		OpenModelInferBase::SetMemoryShared();

		for (const auto& layer : input_shared)
		{
			HANDLE handle;
			LPVOID buffer;
			if (GetOnlyReadMapFile(handle, buffer, layer.c_str()))
			{
				input_shared_handle.push_back(handle);
				input_shared_layers.push_back({ layer, buffer });
			}
			else
			{
				LOG("ERROR") << "获取共享内存 " << layer << " 错误 ... " << std::endl;
				throw std::logic_error("获取共享内存 " + layer + " 错误 ... ");
			}
		}
	}

	void ObjectRecognition::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
	{
		debug_frame = frame_ptr.clone();
		std::vector<ObjectResult> results = this->results;

		InferenceEngine::Blob::Ptr data;
		request->GetBlob(outputsInfo.begin()->first.c_str(), data, 0);
		InferenceEngine::SizeVector dims = data->getTensorDesc().getDims();

		if (outputsInfo.size() == 1 && (dims.size() == 2 && dims[1] == 70) || (dims.size() == 4 && dims[1] == 10))
		{
			const float* buffer = data->buffer().as<float*>();
			int length = dims[1];

			for (int i = 0; i < results.size(); i++)
			{
				auto begin = i * length;
				auto end = begin + length / 2;

				results[i].id = i;
				cv::Rect rect = results[i].location;

				for (auto index = begin; index < end; index++)
				{
					float fx = buffer[index * 2];
					float fy = buffer[index * 2 + 1];

					int tx = rect.x + (rect.width * fx);
					int ty = rect.y + (rect.height * fy);

					cv::circle(debug_frame, cv::Point(tx, ty), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
				}

				DrawObjectBound(debug_frame, rect);
			}

			std::stringstream use_time;
			use_time << "Recognition Count:" << results.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
			cv::putText(debug_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
		}
	}

	void ObjectRecognition::UpdateDebugShow()
	{
		cv::imshow("Object Recognition", debug_frame);
		cv::waitKey(1);
	}
}