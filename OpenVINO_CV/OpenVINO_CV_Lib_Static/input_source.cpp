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

	lastFrameID = 0;
	vsd_data.fid = 0;
	LOG("INFO") << "InputSource Open: [" << args << "]" << std::endl;

	size_t start = 0;
	size_t end = args.find(Delimiter);
	std::string head = args.substr(start, end);

	if (head == "cam" || head == "camera")
	{
		type = InputType::CAMERA;
		//��� Ĭ������ Ϊ 0
		int index = (end == std::string::npos) ? 0 : std::stoi(args.substr(end + 1).c_str());
		
		isopen = capture.open(index);
		LOG("INFO") << "InputSource Camera: [Index:" << index << "]" << std::endl;

		return isopen;
	}
	else if (head == "url" || head == "video")
	{
		type = InputType::VIDEO;
		//Video û��Ĭ�ϲ���
		if (end == std::string::npos)
			throw std::invalid_argument("δָ�� VIDEO Դ·����" + args);

		std::string url = args.substr(end + 1);

		isopen = capture.open(url);
		LOG("INFO") << "InputSource Video: [URL:" << url << "]" << std::endl;

		return isopen;
	}
	else if (head == "share" || head == "shared")
	{
		type = InputType::SHARED;
		//�ڴ湲��Դ���� ��Ĭ��ֵ Ϊ "source.bin"
		shared_name = (end == std::string::npos) ? SHARED_INPUT_SOURCE_NAME : args.substr(end + 1);

		//===== ����Ҫ��Ҫ���һ�������ȴ����̣�������
		if (!GetOnlyReadMapFile(pMapFile, pBuffer, shared_name.c_str()))
		{
			pMapFile = NULL, pBuffer = NULL;
			LOG_WARN << "�ڴ湲���ļ��򿪴���" << shared_name << std::endl;
			return false;
		}

		isopen = true;
		//��ȡ֡ͷ������
		std::copy((uint8_t*)pBuffer, (uint8_t*)pBuffer + VSD_SIZE, (uint8_t*)&vsd_data);
		LOG("INFO") << "InputSource Shared: [Name:" << shared_name << "]" << std::endl;

		return isopen;
	}
	else if (head == "sock" || head == "socket")
	{
		isopen = false;
		type = InputType::SOCKET;

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

	case InputType::SOCKET:
		isopen = false;
		break;
	}

	return false;
}
