#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

//�� 1 �ֽڶ���
#pragma pack(push, 1)


// [�ڴ湲����������Դ] ��Ƶ�����Դ���ݸ�ʽ
// pack(1) ���Ϊ 32 �ֽڴ�С������ͷ��Ϣ���鲻Ҫ���� 32 �ֽ�
struct VideoSourceData
{
    uint32_t    fid = 0;        //Frame ID�����ݿ�ÿ����һ�Σ���ֵ�����һ�Σ������ڼ��㲻���ظ�����֡

    /// Camera ����
    uint16_t    width;          //��Ƶ�����Դ��
    uint16_t    height;         //��Ƶ�����Դ��
    uint8_t     channels;       //��Ƶ�����Դͨ������RGB��3ͨ����ARGB��4ͨ�����Ҷ�ͼ��2ͨ���͵�ͨ��(1ͨ��)
    uint32_t    length;         //�ļ����ݴ�С
    uint8_t     type;           //Mat type
    uint8_t     format;         //��Ƶ�����Դ��ʽ��0 ��ʾδԭʼ���ݸ�ʽ��>0 ��ʾѹ�����ͼ�����ݣ�����ͼ���ѹ����δ����

    /// Video ����
    uint8_t     fps = 0;   //����
    uint32_t    reserve1 = 0;   //����
    uint32_t    reserve2 = 0;   //����
    uint32_t    reserve3 = 0;   //����
    uint32_t    reserve4 = 0;   //����

    //uint8_t     *data;          //raw data

    //copy Mat info to VideoSourceData
    void copy(const cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();

        this->length = this->width * this->height * this->channels * src.elemSize1();
    };

    //copy Mat info to VideoSourceData
    VideoSourceData& operator << (const cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();
        this->length = this->width * this->height * this->channels * src.elemSize1();

        return *this;
    };
};

// ���ݸ�ʽ
enum class OutputFormat :uint8_t
{
    RAW,        //ԭʼ�ֽ����ݣ��ɽ�ԭʼ�ֽ�תΪָ���Ľṹ����
    STRING,     //�ַ����ͣ�δ�����ַ���ʽ������һ������
    JSON,       //�ַ����ͣ�����Ϊ JSON ��ʽ
    XML,        //�ַ����ͣ�����Ϊ XML ��ʽ
};

// [�ڴ湲������] �����������ݽṹ
// pack(1) ���Ϊ 32 �ֽڴ�С������ͷ��Ϣ���鲻Ҫ���� 32 �ֽ�
struct OutputSourceData
{
    uint32_t    fid = 0;        //֡ id
    uint8_t     mid;            //����Ӧ����ָģ��ID�����ĸ�ģ�����������
    uint32_t    length;         //���ݵ���Ч����
    OutputFormat     format;         //���ݸ�ʽ��see OutputFormat

    uint16_t    reserve0 = 0;
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;
    uint32_t    reserve5 = 0;

    //uint8_t     data[];         //raw data����ͬ�Ľ�����ͣ����ݵĽ��Ͳ�һ����Ӧ�����ݵ���Ч���ȶ�̬
};

//VideoSourceData �ṹ���ݴ�С
static const uint16_t VSD_SIZE = sizeof(VideoSourceData);

//OutputSourceData �ṹ���ݴ�С
static const uint16_t OSD_SIZE = sizeof(OutputSourceData);

#pragma pack(pop)