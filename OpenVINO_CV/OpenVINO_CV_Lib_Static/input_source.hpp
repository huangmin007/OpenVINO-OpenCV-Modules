#pragma once

#include <iostream>
#include <Windows.h>
#include "io_data_format.hpp"
#include <opencv2/opencv.hpp>

//ʹ���ڴ濽�������������ڴ�ָ��ָ�򷽷�
#define USE_COPY_INPUT_METHOD	  false

//Ĭ�ϵĹ�������Դ����
#define	SHARED_INPUT_SOURCE_NAME  "source.bin"

namespace space
{
	/// <summary>
	/// Դ����ID�����ڻ�ȡ�����ò���(�� cv::VideoCaptureProperties ͬʱʹ�ã���������չ��)
	/// </summary>
	enum SourceProperties : uint16_t
	{
		FRAME_ID = 0xFF00,			//֡ID
		MODULE_ID = 0xFF01,			//ģ��ID
		OUTPUT_FORMAT = 0xFF02,		//�����ʽ
	};


	/// <summary>
	///	����Դ֧�ֵ�����
	/// </summary>
	enum class InputType :uint8_t
	{
		VIDEO,		//��ƵԴ��Ĭ��Ϊѭ������
		CAMERA,		//���Դ
		SHARED,		//�����ڴ�Դ
	};


	/// <summary>
	/// ����Դ�࣬����������ʽ��(camera|video|shared|socket)[:value[:value[:...]]]
	/// </summary>
	class InputSource
	{
	public:
		/// <summary>
		/// ����Դ���󣬽���������ʽ��(camera|video|shared|socket)[:value[:value[:...]]]
		/// </summary>
		/// <returns></returns>
		InputSource();
		
		/// <summary>
		/// �Ƿ��� Video|Camera Դ��Դ����
		/// </summary>
		/// <param name="shared"></param>
		/// <returns></returns>
		InputSource(const std::string shared);
		~InputSource();


		/// <summary>
		/// ������Դ
		/// </summary>
		/// <param name="args">����Դ�������ο���ʽ��(camera|video|shared|socket)[:value[:value[:...]]]</param>
		/// <returns></returns>
		bool open(const std::string args);


		/// <summary>
		/// ����Դ�Ƿ��Ѿ���
		/// </summary>
		/// <returns></returns>
		bool isOpened() const;

		/// <summary>
		/// ��ȡ����Դ֡����
		/// </summary>
		/// <param name="frame">����֡����</param>
		/// <returns>��ȡ�ɹ����� true�����򷵻� false</returns>
		bool read(cv::Mat& frame);

		/// <summary>
		/// ��ȡ����Դ֡����
		/// </summary>
		/// <param name="frame">����֡����</param>
		/// <param name="frame_id">����֡id</param>
		/// <returns>��ȡ�ɹ����� true�����򷵻� false</returns>
		bool read(cv::Mat& frame, size_t& frame_id);

		/// <summary>
		/// ��ȡ��ǰ����Դ����
		/// </summary>
		/// <returns></returns>
		InputType getType() const;

		/// <summary>
		/// ��ȡ����Դ����
		/// </summary>
		/// <param name="propId"></param>
		/// <returns></returns>
		double get(int propId) const;

		/// <summary>
		/// ��������Դ����
		/// </summary>
		/// <param name="propId"></param>
		/// <param name="value"></param>
		/// <returns></returns>
		bool set(int propId, double value);

		/// <summary>
		/// �ͷŲ��ر�����Դ
		/// </summary>
		/// <returns></returns>
		bool release();

	protected:

	private:
		size_t lastFrameID = 0xFFFFFFFF;
		VideoSourceData* vsd_data;		//���빲����ƵԴͷ������Ϣ

		HANDLE	input_map_file = NULL;		//���ڹ����ڴ�Դ���ļ����
		LPVOID	input_buffer = NULL;		//���ڹ����ڴ�����ָ��ӳ��

		InputType type;					//����Դ����
		bool isopen = false;			//����Դ�Ƿ��Ѿ���
		cv::VideoCapture capture;		//������ƵԴ�����Դ

		HANDLE output_map_file;
		LPVOID output_buffer;
		std::string output_shared_name;

		std::string video_url;

	};
};

