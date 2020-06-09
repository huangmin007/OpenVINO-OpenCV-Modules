#include "object_detection.hpp"
//#include "object_recognition.hpp"
#include <static_functions.hpp>

namespace space
{
	ObjectRecognition::ObjectRecognition(const std::vector<std::string>& output_layers_name, bool is_debug)
		:OpenModelInferBase(output_layers_name, is_debug) {};

	ObjectRecognition::~ObjectRecognition() {}
	
	
	void ObjectRecognition::RequestInfer(const cv::Mat& frame, const std::vector<ObjectDetection::Result> results)
	{
		if (!is_configuration)
			throw std::logic_error("未配置网络对象，不可操作 ...");
		if (frame.empty() || frame.type() != CV_8UC3)
			throw std::logic_error("输入图像帧不能为空帧对象，CV_8UC3 类型 ... ");
		if (is_reshape)
			throw std::invalid_argument("对象识别不支持重塑网络输入层 ... ");

		if (results.size() == 0) return;

		frame_ptr = frame;
		this->results = results;

		InferenceEngine::StatusCode code;
		InferenceEngine::InferRequest::Ptr requestPtr;

		//查找没有启动的推断对象
		for (auto& request : requestPtrs)
		{
			code = request->Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
			if (code == InferenceEngine::StatusCode::INFER_NOT_STARTED
				//|| code == InferenceEngine::StatusCode::OK
				)
			{
				requestPtr = request;
				break;
			}
		}
		if (requestPtr == nullptr) return;
		//LOG("INFO") << "找到一个 InferRequestPtr 对象，Input Address:" << (void*)requestPtr->GetBlob(inputsInfo.begin()->first)->buffer().as<uint8_t*>() << std::endl;

		frame.copyTo(prev_frame);

		auto count = std::min(results.size(), batch_size);
		InferenceEngine::Blob::Ptr inputBlob = requestPtr->GetBlob(inputsInfo.begin()->first);

		for (int i = 0; i < count; i++)
		{
			if (results[i].location.empty()) continue;
			auto clipRect = results[i].location & cv::Rect(0, 0, frame.cols, frame.rows);

			cv::Mat roi = frame(clipRect);
			MatU8ToBlob<uint8_t>(roi, inputBlob, i);
		}

		requestPtr->SetBatch(count + 1);
		requestPtr->StartAsync();
		t0 = std::chrono::high_resolution_clock::now();
	};

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

					//1.解析数据
					ParsingOutputData(request, code);

					//2.更新显示结果
					if (is_debug) UpdateDebugShow();

					//3.next infer request
					prev_results = results;
					frame_ptr.copyTo(prev_frame);

					auto count = std::min(results.size(), batch_size);
					InferenceEngine::Blob::Ptr inputBlob;
					request->GetBlob(inputsInfo.begin()->first.c_str(), inputBlob, 0);

					for (int i = 0; i < count; i++)
					{
						if (results[i].location.empty()) continue;
						auto clipRect = results[i].location & cv::Rect(0, 0, prev_frame.cols, prev_frame.rows);

						cv::Mat roi = prev_frame(clipRect);
						MatU8ToBlob<uint8_t>(roi, inputBlob, i);
					}

					request->SetBatch(count + 1, 0);
					request->StartAsync(0);
					t0 = std::chrono::high_resolution_clock::now();
				});
		};

		timer.start(1000, [&]
			{
				{
					std::lock_guard<std::shared_mutex> lock(_fps_mutex);
					fps = fps_count;
					fps_count = 0;
				};
			});
	}

	static void BuildCameraMatrix(cv::Mat& cameraMatrix, int cx, int cy, float focalLength)
	{
		if (!cameraMatrix.empty()) return;
		cameraMatrix = cv::Mat::zeros(3, 3, CV_32F);
		cameraMatrix.at<float>(0) = focalLength;
		cameraMatrix.at<float>(2) = static_cast<float>(cx);
		cameraMatrix.at<float>(4) = focalLength;
		cameraMatrix.at<float>(5) = static_cast<float>(cy);
		cameraMatrix.at<float>(8) = 1;
	}

	static void DrawHeadPose(cv::Mat& frame, cv::Point3f cpoint, float yaw, float pitch, float roll) 
	{
		//double yaw = headPose.angle_y;
		//double pitch = headPose.angle_p;
		//double roll = headPose.angle_r;

		pitch *= CV_PI / 180.0;
		yaw *= CV_PI / 180.0;
		roll *= CV_PI / 180.0;

		cv::Matx33f Rx(1, 0, 0,
			0, static_cast<float>(cos(pitch)), static_cast<float>(-sin(pitch)),
			0, static_cast<float>(sin(pitch)), static_cast<float>(cos(pitch)));

		cv::Matx33f Ry(static_cast<float>(cos(yaw)), 0, static_cast<float>(-sin(yaw)),
			0, 1, 0,
			static_cast<float>(sin(yaw)), 0, static_cast<float>(cos(yaw)));

		cv::Matx33f Rz(static_cast<float>(cos(roll)), static_cast<float>(-sin(roll)), 0,
			static_cast<float>(sin(roll)), static_cast<float>(cos(roll)), 0,
			0, 0, 1);


		auto r = cv::Mat(Rz * Ry * Rx);
		cv::Mat cameraMatrix;
		BuildCameraMatrix(cameraMatrix, frame.cols / 2, frame.rows / 2, 950.0);

		cv::Mat xAxis(3, 1, CV_32F), yAxis(3, 1, CV_32F), zAxis(3, 1, CV_32F), zAxis1(3, 1, CV_32F);

		float scale = 50.0f;
		int axisThickness = 2;
		cv::Scalar xAxisColor(0, 0, 255);
		cv::Scalar yAxisColor(0, 255, 0);
		cv::Scalar zAxisColor(255, 0, 0);

		xAxis.at<float>(0) = 1 * scale;
		xAxis.at<float>(1) = 0;
		xAxis.at<float>(2) = 0;

		yAxis.at<float>(0) = 0;
		yAxis.at<float>(1) = -1 * scale;
		yAxis.at<float>(2) = 0;

		zAxis.at<float>(0) = 0;
		zAxis.at<float>(1) = 0;
		zAxis.at<float>(2) = -1 * scale;

		zAxis1.at<float>(0) = 0;
		zAxis1.at<float>(1) = 0;
		zAxis1.at<float>(2) = 1 * scale;

		cv::Mat o(3, 1, CV_32F, cv::Scalar(0));
		o.at<float>(2) = cameraMatrix.at<float>(0);

		xAxis = r * xAxis + o;
		yAxis = r * yAxis + o;
		zAxis = r * zAxis + o;
		zAxis1 = r * zAxis1 + o;

		cv::Point p1, p2;

		p2.x = static_cast<int>((xAxis.at<float>(0) / xAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
		p2.y = static_cast<int>((xAxis.at<float>(1) / xAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
		cv::line(frame, cv::Point(static_cast<int>(cpoint.x), static_cast<int>(cpoint.y)), p2, xAxisColor, axisThickness);

		p2.x = static_cast<int>((yAxis.at<float>(0) / yAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
		p2.y = static_cast<int>((yAxis.at<float>(1) / yAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
		cv::line(frame, cv::Point(static_cast<int>(cpoint.x), static_cast<int>(cpoint.y)), p2, yAxisColor, axisThickness);

		p1.x = static_cast<int>((zAxis1.at<float>(0) / zAxis1.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
		p1.y = static_cast<int>((zAxis1.at<float>(1) / zAxis1.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);

		p2.x = static_cast<int>((zAxis.at<float>(0) / zAxis.at<float>(2) * cameraMatrix.at<float>(0)) + cpoint.x);
		p2.y = static_cast<int>((zAxis.at<float>(1) / zAxis.at<float>(2) * cameraMatrix.at<float>(4)) + cpoint.y);
		cv::line(frame, p1, p2, zAxisColor, axisThickness);
		cv::circle(frame, p2, 3, zAxisColor, axisThickness);
	}


	void ObjectRecognition::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
	{
		std::vector<ObjectDetection::Result> results = this->prev_results;

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
				//std::vector<float> marks;	

				results[i].id = i;
				cv::Rect rect = results[i].location;
				//auto rect = results[i].location & cv::Rect(0, 0, prev_frame.cols, prev_frame.rows);

				for (auto index = begin; index < end; index ++)
				{
					float fx = buffer[index * 2];
					float fy = buffer[index * 2 + 1];

					//marks.push_back(fx);
					//marks.push_back(fy);

					int tx = rect.x + (rect.width * fx);
					int ty = rect.y + (rect.height * fy);

					cv::circle(prev_frame, cv::Point(tx, ty), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
				}

				//DrawObjectBound(prev_frame, rect);
			}

			ObjectDetection::DrawObjectBound(prev_frame, results, {});

			std::stringstream use_time;
			use_time << "Recognition Count:" << results.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
			cv::putText(prev_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

			cv::imshow("Object Recognition [Face Landmarks]", prev_frame);
			cv::waitKey(1);
		}
		else if (outputsInfo.size() == 2)
		{

		}
		else if (outputsInfo.size() == 3 && dims.size() == 2 && dims[1] == 1)
		{
			InferenceEngine::Blob::Ptr angleR, angleP, angleY;
			request->GetBlob(output_shared_layers[0].first.c_str(), angleR, 0);
			request->GetBlob(output_shared_layers[1].first.c_str(), angleP, 0);
			request->GetBlob(output_shared_layers[2].first.c_str(), angleY, 0);

			for (int i = 0; i < results.size(); i++)
			{
				float roll = angleR->buffer().as<float*>()[i];		//roll
				float pitch = angleP->buffer().as<float*>()[i];		//pitch
				float yaw = angleY->buffer().as<float*>()[i];		//yaw

				results[i].id = i;
				cv::Rect rect = results[i].location;
				cv::Point3f center(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5, 0.0f);
			
				DrawObjectBound(prev_frame, rect);
				DrawHeadPose(prev_frame, center, yaw, pitch, roll);
			}

			ObjectDetection::DrawObjectBound(prev_frame, results, {});

			std::stringstream use_time;
			use_time << "Recognition Count:" << results.size() << "  Infer Use Time:" << GetInferUseTime() << "ms  FPS:" << GetFPS();
			cv::putText(prev_frame, use_time.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

			cv::imshow("Object Recognition [Head Pose]", prev_frame);
			cv::waitKey(1);
		}
		else
		{

		}

		
	}

	void ObjectRecognition::UpdateDebugShow()
	{
	}
}