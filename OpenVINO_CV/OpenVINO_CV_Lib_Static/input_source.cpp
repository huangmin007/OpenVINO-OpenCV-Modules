#include "pch.h"
#include "input_source.hpp"
#include "static_functions.hpp"


InputSource::InputSource()
{
	isopen = false;	
}
InputSource::~InputSource()
{
	if(isopen)	release();
}

bool InputSource::isOpened() const{	return isopen;}
InputType InputSource::getType() const{	return type;}

double InputSource::get(int propId) const
{
	if (!isopen)
	{
		LOG_WARN << "����Դδ�򿪣��޷���ȡ���� ... " << std::endl;
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
		LOG_WARN << "����Դδ�򿪣��޷����ò��� ... " << std::endl;
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
		throw std::invalid_argument("open() ��������Ϊ��");
		return false;
	}
	if (isopen)
	{
		LOG_WARN << "����Դ�Ѵ� ... " << std::endl;
		return false;
	}
	LOG("INFO") << "Parsing Input Source: [" << args << "]" << std::endl;

	std::vector<std::string> input = SplitString(args, ':');
	size_t length = input.size();
	if (length <= 0) throw std::logic_error("����Դ��������" + args);

	lastFrameID = 0;
	vsd_data.fid = 0;

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
		return isopen;
	}
	else if (input[0] == "url" || input[0] == "video")
	{
		type = InputType::VIDEO;
		//Video û��Ĭ�ϲ���
		if (length == 1)
		{
			throw std::invalid_argument("δָ�� VIDEO Դ·����" + args);
			return false;
		}

		//video(:url)
		capture.open(input[1]);
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

		return isopen;
	}
	else if (input[0] == "share" || input[0] == "shared")
	{
		type = InputType::SHARED;

		//shared
		std::string shared_name = SHARED_INPUT_SOURCE_NAME;
		//shared[:name]
		if (length >= 2) shared_name = input[1];

		if (!GetOnlyReadMapFile(pMapFile, pBuffer, shared_name.c_str()))
		{
			pMapFile = NULL, pBuffer = NULL;
			LOG_WARN << "�ڴ湲���ļ��򿪴���" << shared_name << std::endl;
			return false;
		}

		isopen = true;
		//��ȡ֡ͷ������
		std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)&vsd_data);
		LOG("INFO") << "Input Source Shared: [Name:" << shared_name << "]" << std::endl;

		return isopen;
	}
	else if (input[0] == "sock" || input[0] == "socket")
	{
		isopen = false;

		LOG_WARN << "��ʱδ��ɶ� SOCKET ��֧�֣�" << args << std::endl;
		throw std::invalid_argument("��ʱδ��ɶ� SOCKET ��֧�֣�" + args);

		return isopen;
	}
	else
	{
		LOG_ERROR << "��֧�ֵ�����Դ���� ������" << args << std::endl;
		throw std::invalid_argument("��֧�ֵ�����Դ���� ������" + args);
	}

	return false;
}

bool InputSource::read(cv::Mat& frame)
{
	if (!isopen)
	{
		LOG_WARN << "����Դδ�򿪣��޷���ȡ֡���� ... " << std::endl;
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
			
			//ͬһ֡���ݣ�����false
			if (lastFrameID == vsd_data.fid) return false;

			lastFrameID = vsd_data.fid;
			std::copy((uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)pBuffer + VSD_SIZE + vsd_data.length, frame.data);
			return true;
	}

	return false;
}
bool InputSource::read(cv::Mat &frame, size_t &frame_id)
{
	if (!isopen)
	{
		LOG_WARN << "����Դδ�򿪣��޷���ȡ֡���� ... " << std::endl;
		return false;
	}

	frame_id = vsd_data.fid;
	return read(frame);
}

bool InputSource::release()
{
	if (!isopen)
	{
		LOG_WARN << "����Դδ�򿪣���������ͷ� ... " << std::endl;
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
		if (pBuffer != NULL)	UnmapViewOfFile(pBuffer);	//������ַ�ռ��ڵ���ͼ
		if (pMapFile != NULL)	CloseHandle(pMapFile);		//�رչ����ļ����
		break;
	}

	return false;
}
