#pragma once

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

//��дԴʼ�����ڴ湲���ļ�ͷ����Ϣ
struct SourceMapFileInfo
{
    uint32_t fid;       //���ݿ�ÿ����һ�Σ���ֵ�����һ�Σ������ڼ��㲻���ظ�����֡

    /// OpenCV Mat OR Camera ����
    uint32_t width;     //��ƵԴ��
    uint32_t height;    //��ƵԴ��
    uint32_t channels;   //��ƵԴͨ������RGB��3ͨ����ARGB��4ͨ�����Ҷ�ͼ��2ͨ���͵�ͨ��(1ͨ��)
    uint32_t type;       //Mat type
    uint32_t format;     //��ƵԴ��ʽ��0 ��ʾδԭʼ���ݸ�ʽ��>0 ��ʾѹ�����ͼ�����ݣ�����ͼ���ѹ����δ����

    /// ---- Video ����
    //double duration;

    /// ����Ϊ�����ֶ� 4 x 4 = 16
    uint32_t reserve0 = 0;
    uint32_t reserve1 = 0;
    uint32_t reserve2 = 0;
    uint32_t reserve3 = 0;

    SourceMapFileInfo& operator << (cv::Mat& src)
    {
        this->width = src.cols;
        this->height = src.rows;
        this->type = src.type();
        this->channels = src.channels();

        return *this;
    };
};