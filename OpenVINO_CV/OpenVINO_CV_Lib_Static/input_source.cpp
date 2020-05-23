#include "pch.h"
#include "input_source.hpp"
#include "static_functions.hpp"


InputSource::InputSource()
{
	isopen = false;	
}
InputSource::~InputSource()
{
	release();
}

bool InputSource::isOpened() const{	return isopen;}
InputType InputSource::getType() const{	return type;}

double InputSource::get(int propId) const
{
	if (!isopen)
	{
		LOG_WARN << "输入源未打开，无法读取参数 ... " << std::endl;
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
		case cv::CAP_PROP_FPS:					return vsd_data.fps;
		case cv::CAP_PROP_FORMAT:				return vsd_data.format;
		case cv::CAP_PROP_FRAME_WIDTH:			return vsd_data.width;
		case cv::CAP_PROP_FRAME_HEIGHT:			return vsd_data.height;
		case SourceProperties::FRAME_ID:		return vsd_data.fid;
		default: return -1.0;
		}
	}

	
	return -1.0;
}

bool InputSource::set(int propId, double value)
{
	if (!isopen)
	{
		LOG_WARN << "输入源未打开，无法设置参数 ... " << std::endl;
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
		LOG_WARN << "输入源已打开 ... " << std::endl;
		return false;
	}

	lastFrameID = 0;
	vsd_data.fid = 0;
	LOG("INFO") << "InputSource Open: [" << args << "]" << std::endl;

	size_t start = 0;
	size_t end = args.find(Delimiter);
	std::string head = args.substr(start, end);

	if (head == "cam" || head == "camera")
	{
		type = InputType::CAMERA;
		//相机 默认索引 为 0
		int index = (end == std::string::npos) ? 0 : std::stoi(args.substr(end + 1).c_str());
		
		isopen = capture.open(index);
		LOG("INFO") << "InputSource Camera: [Index:" << index << "]" << std::endl;

		return isopen;
	}
	else if (head == "url" || head == "video")
	{
		type = InputType::VIDEO;
		//Video 没有默认参数
		if (end == std::string::npos)
			throw std::invalid_argument("未指定 VIDEO 源路径：" + args);

		std::string url = args.substr(end + 1);

		isopen = capture.open(url);
		LOG("INFO") << "InputSource Video: [URL:" << url << "]" << std::endl;

		return isopen;
	}
	else if (head == "share" || head == "shared")
	{
		type = InputType::SHARED;
		//内存共享源名称 有默认值 为 "source.bin"
		shared_name = (end == std::string::npos) ? SHARED_INPUT_SOURCE_NAME : args.substr(end + 1);

		//===== 这里要不要设计一个持续等待过程？？？？
		if (!GetOnlyReadMapFile(pMapFile, pBuffer, shared_name.c_str()))
		{
			pMapFile = NULL, pBuffer = NULL;
			LOG_WARN << "内存共享文件打开错误：" << shared_name << std::endl;
			return false;
		}

		isopen = true;
		//读取帧头部数据
		std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)&vsd_data);
		LOG("INFO") << "InputSource Shared: [Name:" << shared_name << "]" << std::endl;

		return isopen;
	}
	else if (head == "sock" || head == "socket")
	{
		isopen = false;
		type = InputType::SOCKET;

		LOG_WARN << "暂时未完成对 SOCKET 的支持：" << args << std::endl;
		throw std::invalid_argument("暂时未完成对 SOCKET 的支持：" + args);

		return isopen;
	}
	else
	{
		LOG_ERROR << "不支持的输入源类型 参数：" << args << std::endl;
		throw std::invalid_argument("不支持的输入源类型 参数：" + args);
	}

	return false;
}

bool InputSource::read(cv::Mat& frame)
{
	if (!isopen)
	{
		LOG_WARN << "输入源未打开，无法读取帧数据 ... " << std::endl;
		return false;
	}

	switch (type)
	{
		case InputType::VIDEO:
		case InputType::CAMERA:
			vsd_data.fid++;
			return capture.read(frame);

		case InputType::SHARED:
			std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)&vsd_data);

			if (frame.empty())
				frame.create(vsd_data.height, vsd_data.width, vsd_data.type);
			
			if(lastFrameID != vsd_data.fid)
				std::copy((uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)pBuffer + VSD_SIZE + vsd_data.length, frame.data);
			
			lastFrameID = vsd_data.fid;
			return true;

		case InputType::SOCKET:
			return false;
			break;
	}

	return false;
}
bool InputSource::read(cv::Mat &frame, size_t &frame_id)
{
	if (!isopen)
	{
		LOG_WARN << "输入源未打开，无法读取帧数据 ... " << std::endl;
		return false;
	}

	frame_id = vsd_data.fid;
	return read(frame);
}

bool InputSource::release()
{
	if (!isopen)
	{
		LOG_WARN << "输入源未打开，无需操作释放 ... " << std::endl;
		return false;
	}

	switch (type)
	{
	case InputType::VIDEO:
	case InputType::CAMERA:
		isopen = false;
		capture.release();
		return true;

	case InputType::SHARED:
		isopen = false;
		if (pBuffer != NULL)	UnmapViewOfFile(pBuffer);	//撤消地址空间内的视图
		if (pMapFile != NULL)	CloseHandle(pMapFile);		//关闭共享文件句柄
		break;

	case InputType::SOCKET:
		isopen = false;
		break;
	}

	return false;
}
