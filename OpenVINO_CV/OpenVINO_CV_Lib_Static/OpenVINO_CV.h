#pragma once

#include <iostream>
#include <chrono>

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

//读写源始数据内存共享文件头部信息
struct SourceMapFileInfo
{
    int width;
    int height;
    int channls;
    //编码类型
    int type;
    int reserve0 = 0;
    int reserve1 = 0;
    int reserve2 = 0;
    int reserve3 = 0;
};