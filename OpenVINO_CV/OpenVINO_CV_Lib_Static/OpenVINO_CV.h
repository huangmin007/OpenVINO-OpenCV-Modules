#pragma once

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

//读写源始数据内存共享文件头部信息
struct SourceMapFileInfo
{
    uint32_t fid;       //数据块每更新一次，该值会递增一次，可用于计算不做重复分析帧

    /// OpenCV Mat OR Camera 参数
    uint32_t width;     //视频源宽
    uint32_t height;    //视频源高
    uint32_t channels;   //视频源通道数，RGB是3通道，ARGB是4通道，灰度图分2通道和单通道(1通道)
    uint32_t type;       //Mat type
    uint32_t format;     //视频源格式，0 表示未原始数据格式，>0 表示压缩后的图像数据，各种图像的压缩暂未定义

    /// ---- Video 参数
    //double duration;

    /// 以下为保留字段 4 x 4 = 16
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