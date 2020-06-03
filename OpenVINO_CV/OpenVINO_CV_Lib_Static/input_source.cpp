#include "pch.h"
#include "input_source.hpp"
#include "static_functions.hpp"

namespace space
{
	InputSource::InputSource():isopen(false){}
	InputSource::InputSource(const std::string output_shared_name) : output_shared_name(output_shared_name), isopen(false) {}
	
	InputSource::~InputSource()
	{
		if (isopen)	release();
	}

	bool InputSource::isOpened() const { return isopen; }
	InputType InputSource::getType() const { return type; }

	double InputSource::get(int propId) const
	{
		if (!isopen)
		{
			LOG("WARN") << "输入源未打开，无法读取参数 ... " << std::endl;
			return -1.0;
		}

		switch (type)
		{
		case InputType::VIDEO:
		case InputType::CAMERA:
			switch (propId)
			{
			case SourceProperties::FRAME_ID: return 0;
			default:	return capture.get(propId);
			}

		case InputType::SHARED:
			switch (propId)
			{

			case cv::CAP_PROP_FPS:					return vsd_data->fps;
			case cv::CAP_PROP_FORMAT:				return vsd_data->format;
			case cv::CAP_PROP_FRAME_WIDTH:			return vsd_data->width;
			case cv::CAP_PROP_FRAME_HEIGHT:			return vsd_data->height;
			case SourceProperties::FRAME_ID:		return vsd_data->fid;
			default: return -1.0;
			}
		}

		return -1.0;
	}

	bool InputSource::set(int propId, double value)
	{
		if (!isopen)
		{
			LOG("WARN") << "输入源未打开，无法设置参数 ... " << std::endl;
			return false;
		}

		switch (type)
		{
		case InputType::VIDEO:
		case InputType::CAMERA:
			return capture.set(propId, value);
		}

		return false;
	}

	bool InputSource::open(const std::string args)
	{
		if (args.empty())
		{
			throw std::invalid_argument("open() 参数不能为空");
			return false;
		}
		if (isopen)
		{
			LOG("WARN") << "输入源已打开 ... " << std::endl;
			return false;
		}
		LOG("INFO") << "Parsing Input Source: [" << args << "]" << std::endl;

		std::vector<std::string> input = SplitString(args, ':');
		size_t length = input.size();
		if (length <= 0)
		{
			throw std::logic_error("输入源参数错误：" + args);
			return false;
		}

		if (input[0] == "cam" || input[0] == "camera")
		{
			type = InputType::CAMERA;

			//camera
			int index = 0, width = 640, height = 480;
			//camera[:index]
			if (length >= 2)    index = std::stoi(input[1]);
			//camera[:index[:size]]
			if (length >= 3)    ParseArgForSize(input[2], width, height);

			isopen = capture.open(index);
			capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
			capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);

			LOG("INFO") << "Input Source Camera: [Index:" << index << ", Size:" << capture.get(cv::CAP_PROP_FRAME_WIDTH)
				<< "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << "]" << std::endl;
		}
		else if (input[0] == "url" || input[0] == "video")
		{
			type = InputType::VIDEO;
			//Video 没有默认参数
			if (length == 1)
			{
				throw std::invalid_argument("未指定 VIDEO 源路径：" + args);
				return false;
			}

			video_url = input[1];
			//video(:url)
			isopen = capture.open(video_url);
			//video(:url[:size])
			if (length >= 3)
			{
				int width = 640, height = 480;
				if (ParseArgForSize(input[2], width, height))
				{
					capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
					capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
				}
			}

			LOG("INFO") << "Input Source Video: [URL:" << input[1] << ", Size:" << capture.get(cv::CAP_PROP_FRAME_WIDTH)
				<< "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << "]" << std::endl;
		}
		else if (input[0] == "share" || input[0] == "shared")
		{
			type = InputType::SHARED;

			//shared
			std::string shared_name = SHARED_INPUT_SOURCE_NAME;
			//shared[:name]
			if (length >= 2) shared_name = input[1];

			if (!GetOnlyReadMapFile(input_map_file, input_buffer, shared_name.c_str()))
			{
				input_map_file = NULL, input_buffer = NULL;
				LOG("WARN") << "内存共享文件打开错误：" << shared_name << std::endl;
				return false;
			}

			isopen = true;
			//读取帧头部数据
			vsd_data = (VideoSourceData*)input_buffer;
			//std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)&vsd_data);
			LOG("INFO") << "Input Source Shared: [Name:" << shared_name << "]" << std::endl;

			return isopen;
		}
		else
		{
			LOG("ERROR") << "不支持的输入源类型 参数：" << args << std::endl;
			throw std::invalid_argument("不支持的输入源类型 参数：" + args);
			return false;
		}

		//创建 Video|Camera 共享源
		if (!output_shared_name.empty() && (type == InputType::CAMERA || type == InputType::VIDEO))
		{
			int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
			int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

			if (!CreateOnlyWriteMapFile(output_map_file, output_buffer, width * height * 4 + VSD_SIZE, output_shared_name.c_str()))
			{
				if (output_buffer != NULL) UnmapViewOfFile(output_buffer);
				if (output_map_file != NULL) CloseHandle(output_map_file);

				output_buffer = NULL;
				output_map_file = NULL;
			}
			else
			{
				vsd_data = (VideoSourceData*)output_buffer;
				vsd_data->fid = 0;
				vsd_data->width = width;
				vsd_data->height = height;
			}
		}
		else
		{
			vsd_data = new VideoSourceData();
			vsd_data->fid = 0;
		}

		return true;
	}

	bool InputSource::read(cv::Mat& frame)
	{
		if (!isopen)
		{
			LOG("WARN") << "输入源未打开，无法读取帧数据 ... " << std::endl;
			return false;
		}

		bool isReadOK = false;
		switch (type)
		{
		case InputType::VIDEO:
			if (frame.empty() && output_buffer != NULL)
			{
				frame.create(vsd_data->height, vsd_data->width, CV_8UC3);

				vsd_data = (VideoSourceData*)output_buffer;
				frame.data = (uint8_t*)output_buffer + VSD_SIZE;
			}

			isReadOK = capture.read(frame);

			//重复播放
			if (!isReadOK)
			{
				vsd_data->fid = 0;
				capture.open(video_url);
				isReadOK = capture.read(frame);
			}
			
			if (isReadOK)
			{
				vsd_data->fid += 1;
				vsd_data->copy(frame);
			}

			return isReadOK;

		case InputType::CAMERA:
			if (frame.empty() && output_buffer != NULL)
			{
				frame.create(vsd_data->height, vsd_data->width, CV_8UC3);
				vsd_data = (VideoSourceData*)output_buffer;
				frame.data = (uint8_t*)output_buffer + VSD_SIZE;
			}

			isReadOK = capture.read(frame);

			if (isReadOK)
			{
				vsd_data->fid += 1;
				vsd_data->copy(frame);
			}
			return isReadOK;

		case InputType::SHARED:
#if USE_COPY_INPUT_METHOD
			std::copy((uint8_t*)input_buffer, (uint8_t*)input_buffer + VSD_SIZE, (uint8_t*)&vsd_data);
			if (frame.empty()) frame.create(vsd_data->height, vsd_data->width, vsd_data->type);

			//同一帧数据，返回false
			if (lastFrameID == vsd_data->fid) return false;

			lastFrameID = vsd_data->fid;
			std::copy((uint8_t*)input_buffer + VSD_SIZE, (uint8_t*)input_buffer + VSD_SIZE + vsd_data->length, frame.data);
#else
			vsd_data = (VideoSourceData*)input_buffer;
			if (frame.empty()) frame.create(vsd_data->height, vsd_data->width, vsd_data->type);

			//同一帧数据，返回false
			if (lastFrameID == vsd_data->fid) return false;

			lastFrameID = vsd_data->fid;
			frame.data = (uint8_t*)input_buffer + VSD_SIZE;
#endif
			return true;
		}

		return false;
	}
	bool InputSource::read(cv::Mat& frame, size_t& frame_id)
	{
		if (!isopen)
		{
			LOG("WARN") << "输入源未打开，无法读取帧数据 ... " << std::endl;
			return false;
		}

		//注意顺序是先读帧，才能获取帧id
		bool result = read(frame);
		frame_id = vsd_data->fid;

		return result;
	}

	bool InputSource::release()
	{
		if (!isopen)
		{
			LOG("WARN") << "输入源未打开，无需操作释放 ... " << std::endl;
			return false;
		}

		switch (type)
		{
		case InputType::VIDEO:
		case InputType::CAMERA:
			isopen = false;
			capture.release();
			if (output_buffer != NULL)	UnmapViewOfFile(output_buffer);
			if (output_map_file != NULL)	CloseHandle(output_map_file);

			output_buffer = NULL;
			output_map_file = NULL;

			return true;

		case InputType::SHARED:
			isopen = false;

			if (input_buffer != NULL)	UnmapViewOfFile(input_buffer);	//撤消地址空间内的视图
			if (input_map_file != NULL)	CloseHandle(input_map_file);		//关闭共享文件句柄

			input_buffer = NULL;
			input_map_file = NULL;
			return true;
		}

		return false;
	}
}