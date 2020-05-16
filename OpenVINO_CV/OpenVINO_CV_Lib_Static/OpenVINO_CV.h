#pragma once

#include <iostream>
#include <chrono>

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

//��дԴʼ�����ڴ湲���ļ�ͷ����Ϣ
struct SourceMapFileInfo
{
    int width;
    int height;
    int channls;
    //��������
    int type;
    int reserve0 = 0;
    int reserve1 = 0;
    int reserve2 = 0;
    int reserve3 = 0;
};