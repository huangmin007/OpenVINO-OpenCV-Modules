#pragma once

#include <iostream>
#include <Windows.h>
#include "io_data_format.hpp"
#include <opencv2/opencv.hpp>

//默认的共享输入源名称
#define	SHARED_INPUT_SOURCE_NAME  "source.bin"


/// <summary>
/// 源属性ID，用于获取或设置参数(与 cv::VideoCaptureProperties 同时使用，或者是扩展它)
/// </summary>
enum SourceProperties : uint16_t
{
	FRAME_ID = 0xFF00,			//帧ID
	MODULE_ID = 0xFF01,			//模块ID
	OUTPUT_FORMAT = 0xFF02,		//输出格式
};


/// <summary>
///	输入源支持的类型
/// </summary>
enum class InputType:uint8_t
{
	VIDEO,		//视频源，默认为循环播放
	CAMERA,		//相机源
	SHARED,		//共享内存源
};


/// <summary>
/// 输入源类，解析参数格式：(camera|video|shared|socket)[:value[:value[:...]]]
/// </summary>
class InputSource
{
public:
	/// <summary>
	/// 输入源对象，解析参数格式：(camera|video|shared|socket)[:value[:value[:...]]]
	/// </summary>
	/// <returns></returns>
	InputSource();
	~InputSource();


	/// <summary>
	/// 打开输入源
	/// </summary>
	/// <param name="args">输入源参数，参考格式：(camera|video|shared|socket)[:value[:value[:...]]]</param>
	/// <returns></returns>
	bool open(const std::string args);


	/// <summary>
	/// 输入源是否已经打开
	/// </summary>
	/// <returns></returns>
	bool isOpened() const;

	/// <summary>
	/// 读取输入源帧数据
	/// </summary>
	/// <param name="frame">返回帧对象</param>
	/// <returns>读取成功返回 true，否则返回 false</returns>
	bool read(cv::Mat& frame);

	/// <summary>
	/// 读取输入源帧数据
	/// </summary>
	/// <param name="frame">返回帧对象</param>
	/// <param name="frame_id">返回帧id</param>
	/// <returns>读取成功返回 true，否则返回 false</returns>
	bool read(cv::Mat &frame, size_t &frame_id);


	/// <summary>
	/// 获取当前输入源类型
	/// </summary>
	/// <returns></returns>
	InputType getType() const;

	/// <summary>
	/// 获取输入源参数
	/// </summary>
	/// <param name="propId"></param>
	/// <returns></returns>
	double get(int propId) const;

	/// <summary>
	/// 设置输入源参数
	/// </summary>
	/// <param name="propId"></param>
	/// <param name="value"></param>
	/// <returns></returns>
	bool set(int propId, double value);


	/// <summary>
	/// 释放并关闭输入源
	/// </summary>
	/// <returns></returns>
	bool release();

protected:

private:
	size_t lastFrameID = 0;

	VideoSourceData vsd_data;		//输入共享视频源头数据信息
	HANDLE	pMapFile = NULL;		//用于共享内存源的文件句柄
	PVOID	pBuffer = NULL;			//用于共享内存数据指针映射
	std::string shared_name;		//读取共享内存数据的名称

	InputType type;					//输入源类型
	bool isopen = false;			//输入源是否已经打开
	cv::VideoCapture capture;		//用于视频源和相机源

};

