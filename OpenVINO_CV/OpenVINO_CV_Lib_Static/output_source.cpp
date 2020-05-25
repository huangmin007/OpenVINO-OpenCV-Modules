#include "pch.h"
#include "input_source.hpp"
#include "output_source.hpp"
#include "static_functions.hpp"


OutputSource::OutputSource()
{

}

OutputSource::~OutputSource()
{
	if (isopen)	release();
}

bool OutputSource::isOpened() const	{	return isopen;	}
OutputType OutputSource::getType() const{	return type;	}

double OutputSource::get(int propId) const
{
	if (!isopen)
	{
		LOG_WARN << "输出源未打开，无法读取参数 ... " << std::endl;
		return -1.0;
	}
	return -1.0;
}

bool OutputSource::set(int propId, double value)
{
	if (!isopen)
	{
		LOG_WARN << "输出源未打开，无法设置参数 ... " << std::endl;
		return false;
	}

	switch (propId)
	{
	case SourceProperties::FRAME_ID:	osd_data.fid = value;	return true;
	case SourceProperties::MODULE_ID:	osd_data.mid = value;	return true;
	case SourceProperties::OUTPUT_FORMAT:	osd_data.format = (OutputFormat)value;	return true;
	case cv::CAP_PROP_FPS:				return true;
	case cv::CAP_PROP_FRAME_WIDTH:		return true;
	case cv::CAP_PROP_FRAME_HEIGHT:		return true;
	}

	return false;
}


bool OutputSource::open(const std::string args)
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
	LOG("INFO") << "Parsing Output Source: [" << args << "]" << std::endl;

	std::vector<std::string> output = SplitString(args, ':');
	size_t length = output.size();
	if (length <= 0) throw std::logic_error("输入源参数错误：" + args);

	//size_t start = 0;
	//size_t end = args.find(Delimiter);
	//std::string head = args.substr(start, end);

	if (output[0] == "share" || output[0] == "shared")
	{
		type = OutputType::SHARED;

		//shared
		sed_name = SHARED_OUTPUT_SOURCE_NAME;
		sed_size = SHARED_OUTPUT_SOURCE_SIZE;
		//shared[:name]
		if (length >= 2)		sed_name = output[1];
		//shared[:name[:size]]
		if (length >= 3)		sed_size = std::stoi(output[2]);

		if (!CreateOnlyWriteMapFile(pMapFile, pBuffer, sed_size, sed_name.c_str()))
		{
			pMapFile = NULL, pBuffer = NULL;
			LOG_WARN << "内存共享文件创建错误：Name:" << sed_name << " Size:"<< sed_size << std::endl;
			return false;
		}

		isopen = true;
		LOG("INFO") << "Output Source Shared: [Name:" << sed_name << "    Size:" << sed_size << " Bytes]" << std::endl;
		return isopen;
	}
	else if (output[0] == "console")
	{
		isopen = true;
		type = OutputType::CONSOLE;

		//console
		console_head = CONSOLE_OUTPUT_SOURCE_HEAD;
		//console[:head]
		if (length >= 2)	console_head = output[1];

		LOG("INFO") << "OutputSource Console: [Head:" << console_head << "]" << std::endl;

		return isopen;
	}
	else
	{

	}

	return true;
}


bool OutputSource::write(const uint8_t* frame, const size_t length)
{
	if (!isopen)
	{
		LOG_WARN << "输出源未打开，无法写入数据 ... " << std::endl;
		return false;
	}

	osd_data.length = length;

	switch (type)
	{
	case OutputType::SHARED:
		//head
		std::copy((uint8_t*)&osd_data, (uint8_t*)&osd_data + OSD_SIZE, (uint8_t*)pBuffer);
		//data
		std::copy(frame, frame + length, (uint8_t*)pBuffer + OSD_SIZE);
		return true;
		
	case OutputType::CONSOLE:
		std::cout << "[" << console_head << "]" << std::endl;
		std::cout << frame << std::endl;
		break;
	}

	return false;
}

bool OutputSource::write(const uint8_t* frame, const size_t length, const size_t frame_id)
{
	if (!isopen)
	{
		LOG_WARN << "输出源未打开，无法写入数据 ... " << std::endl;
		return false;
	}

	osd_data.fid = frame_id;
	return write(frame, length);
}


bool OutputSource::release()
{
	if (!isopen)
	{
		LOG_WARN << "输出源未打开，无需操作释放 ... " << std::endl;
		return false;
	}

	switch (type)
	{
		case OutputType::SHARED:
			isopen = false;
			if (pBuffer != NULL)	UnmapViewOfFile(pBuffer);	//撤消地址空间内的视图
			if (pMapFile != NULL)	CloseHandle(pMapFile);		//关闭共享文件句柄
			return true;

		case OutputType::CONSOLE:
			isopen = false;
			return true;
	}

	return false;
}