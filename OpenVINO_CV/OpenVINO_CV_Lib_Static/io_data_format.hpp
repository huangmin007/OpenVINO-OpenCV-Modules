#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

//以 1 字节对齐
#pragma pack(push, 1)


// [内存共享输入数据源] 视频或相机源数据格式
// pack(1) 设计为 32 字节大小，数据头信息建议不要超出 32 字节
struct VideoSourceData
{
    uint32_t    fid = 0;        //Frame ID，数据块每更新一次，该值会递增一次，可用于计算不做重复分析帧

    /// Camera 参数
    uint16_t    width;          //视频或相机源宽
    uint16_t    height;         //视频或相机源高
    uint8_t     channels;       //视频或相机源通道数，RGB是3通道，ARGB是4通道，灰度图分2通道和单通道(1通道)
    uint32_t    length;         //文件数据大小
    uint8_t     type;           //Mat type
    uint8_t     format;         //视频或相机源格式，0 表示未原始数据格式，>0 表示压缩后的图像数据，各种图像的压缩暂未定义

    /// Video 参数
    uint8_t     fps = 0;   //保留
    uint32_t    reserve1 = 0;   //保留
    uint32_t    reserve2 = 0;   //保留
    uint32_t    reserve3 = 0;   //保留
    uint32_t    reserve4 = 0;   //保留

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

// 数据格式
enum class OutputFormat :uint8_t
{
    RAW,        //原始字节数据，可将原始字节转为指定的结构数据
    STRING,     //字符类型，未定义字符格式，但有一定规则
    JSON,       //字符类型，定义为 JSON 格式
    XML,        //字符类型，定义为 XML 格式
};

// [内存共享数据] 推理结果的数据结构
// pack(1) 设计为 32 字节大小，数据头信息建议不要超出 32 字节
struct OutputSourceData
{
    uint32_t    fid = 0;        //帧 id
    uint8_t     mid;            //这里应该是指模块ID，由哪个模块输出的数据
    uint32_t    length;         //数据的有效长度
    OutputFormat     format;         //数据格式，see OutputFormat

    uint16_t    reserve0 = 0;
    uint32_t    reserve1 = 0;
    uint32_t    reserve2 = 0;
    uint32_t    reserve3 = 0;
    uint32_t    reserve4 = 0;
    uint32_t    reserve5 = 0;

    //uint8_t     data[];         //raw data，不同的结果类型，数据的解释不一样，应该数据的有效长度动态
};

//VideoSourceData 结构数据大小
static const uint16_t VSD_SIZE = sizeof(VideoSourceData);

//OutputSourceData 结构数据大小
static const uint16_t OSD_SIZE = sizeof(OutputSourceData);

#pragma pack(pop)