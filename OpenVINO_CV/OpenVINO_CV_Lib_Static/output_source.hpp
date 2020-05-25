#pragma once

#include <iostream>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include "io_data_format.hpp"

//默认的内存共享输出源名称
#define	SHARED_OUTPUT_SOURCE_NAME  "o_source.bin"
//默认的内存共享输出源大小
#define	SHARED_OUTPUT_SOURCE_SIZE  1024 * 1024 * 2
//默认控制台输出头标记信息
#define CONSOLE_OUTPUT_SOURCE_HEAD	"Console"


/// <summary>
/// 输出源支持的类型
/// </summary>
enum class OutputType :uint8_t
{
	VIDEO,			//输出视频
	SHARED,			//内存共享类型
	CONSOLE,		//控制台输出类型
};


/// <summary>
/// 输出源类，解析参数格式：(shared|console|socket)[:value[:value[:...]]]
/// </summary>
class OutputSource
{
public:
	/// <summary>
	/// 输出源构造函数，解析参数格式：(shared|console|socket)[:value[:value[:...]]]
	/// </summary>
	/// <returns></returns>
	OutputSource();
	~OutputSource();

	/// <summary>
	/// 打开输出源通道
	/// </summary>
	/// <param name="args">解析参数格式：(shared|console|socket)[:value[:value[:...]]]</param>
	/// <returns></returns>
	bool open(const std::string args);

	
	/// <summary>
	/// 输出源是否已经打开
	/// </summary>
	/// <returns></returns>
	bool isOpened() const;

	
	/// <summary>
	/// 获取输出源参数
	/// </summary>
	/// <param name="propId"></param>
	/// <returns></returns>
	double get(int propId) const;


	/// <summary>
	/// 设置输出源参数
	/// </summary>
	/// <param name="propId"></param>
	/// <returns></returns>
	bool set(int propId, double value);


	/// <summary>
	/// 从输出源中写入数据
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="length"></param>
	/// <returns></returns>
	bool write(const uint8_t* frame, const size_t length);


	/// <summary>
	/// 从输出源中写入数据
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="data"></param>
	/// <returns></returns>
	//bool write(const uint8_t *frame, const OutputSourceData *data);


	/// <summary>
	/// 从输出源中写入数据
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="length"></param>
	/// <param name="frame_id"></param>
	/// <returns></returns>
	bool write(const uint8_t* frame, const size_t length, const size_t frame_id);


	/// <summary>
	/// 获取输出源类型
	/// </summary>
	/// <returns></returns>
	OutputType getType() const;


	/// <summary>
	/// 释放并关闭输出源
	/// </summary>
	/// <returns></returns>
	bool release();

protected:

private:
	OutputSourceData osd_data;		//输出共享数据源头数据信息
	HANDLE	pMapFile = NULL;		//用于共享内存源的文件句柄
	PVOID	pBuffer = NULL;			//用于共享内存数据指针映射
	std::string sed_name;			//写入共享内存数据的名称
	//写入共享内存数据的大小，默认为 2M
	uint32_t sed_size = SHARED_OUTPUT_SOURCE_SIZE;		

	OutputType type;			//输出源类型
	bool isopen = false;		//输出源是否已经打开
	std::string console_head;	//控制台输出头信息
	
};
