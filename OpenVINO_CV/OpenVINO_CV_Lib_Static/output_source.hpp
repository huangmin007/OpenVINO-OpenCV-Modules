#pragma once

#include <iostream>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include "io_data_format.hpp"

//Ĭ�ϵ��ڴ湲�����Դ����
#define	SHARED_OUTPUT_SOURCE_NAME  "o_source.bin"
//Ĭ�ϵ��ڴ湲�����Դ��С
#define	SHARED_OUTPUT_SOURCE_SIZE  1024 * 1024 * 2
//Ĭ�Ͽ���̨���ͷ�����Ϣ
#define CONSOLE_OUTPUT_SOURCE_HEAD	"Console"


/// <summary>
/// ���Դ֧�ֵ�����
/// </summary>
enum class OutputType :uint8_t
{
	VIDEO,			//�����Ƶ
	SHARED,			//�ڴ湲������
	CONSOLE,		//����̨�������
};


/// <summary>
/// ���Դ�࣬����������ʽ��(shared|console|socket)[:value[:value[:...]]]
/// </summary>
class OutputSource
{
public:
	/// <summary>
	/// ���Դ���캯��������������ʽ��(shared|console|socket)[:value[:value[:...]]]
	/// </summary>
	/// <returns></returns>
	OutputSource();
	~OutputSource();

	/// <summary>
	/// �����Դͨ��
	/// </summary>
	/// <param name="args">����������ʽ��(shared|console|socket)[:value[:value[:...]]]</param>
	/// <returns></returns>
	bool open(const std::string args);

	
	/// <summary>
	/// ���Դ�Ƿ��Ѿ���
	/// </summary>
	/// <returns></returns>
	bool isOpened() const;

	
	/// <summary>
	/// ��ȡ���Դ����
	/// </summary>
	/// <param name="propId"></param>
	/// <returns></returns>
	double get(int propId) const;


	/// <summary>
	/// �������Դ����
	/// </summary>
	/// <param name="propId"></param>
	/// <returns></returns>
	bool set(int propId, double value);


	/// <summary>
	/// �����Դ��д������
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="length"></param>
	/// <returns></returns>
	bool write(const uint8_t* frame, const size_t length);


	/// <summary>
	/// �����Դ��д������
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="data"></param>
	/// <returns></returns>
	//bool write(const uint8_t *frame, const OutputSourceData *data);


	/// <summary>
	/// �����Դ��д������
	/// </summary>
	/// <param name="frame"></param>
	/// <param name="length"></param>
	/// <param name="frame_id"></param>
	/// <returns></returns>
	bool write(const uint8_t* frame, const size_t length, const size_t frame_id);


	/// <summary>
	/// ��ȡ���Դ����
	/// </summary>
	/// <returns></returns>
	OutputType getType() const;


	/// <summary>
	/// �ͷŲ��ر����Դ
	/// </summary>
	/// <returns></returns>
	bool release();

protected:

private:
	OutputSourceData osd_data;		//�����������Դͷ������Ϣ
	HANDLE	pMapFile = NULL;		//���ڹ����ڴ�Դ���ļ����
	PVOID	pBuffer = NULL;			//���ڹ����ڴ�����ָ��ӳ��
	std::string sed_name;			//д�빲���ڴ����ݵ�����
	//д�빲���ڴ����ݵĴ�С��Ĭ��Ϊ 2M
	uint32_t sed_size = SHARED_OUTPUT_SOURCE_SIZE;		

	OutputType type;			//���Դ����
	bool isopen = false;		//���Դ�Ƿ��Ѿ���
	std::string console_head;	//����̨���ͷ��Ϣ
	
};
