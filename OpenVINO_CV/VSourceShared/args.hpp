#pragma once

#include <iostream>
#include <gflags/gflags.h>

#define     VERSION     "v0.0.1"

static const char h_message[]	= "参数信息";
DEFINE_bool(h, false, h_message);

static const char i_message[]		= "输入视频源或是相机，示例：-i url:<path> 或 -i cam:<index>，默认：cam:0";
DEFINE_string(i, "cam:0", i_message);


static const char s_message[]		= "设置视频源或是相机的输出尺寸，默认:640,480";
DEFINE_string(s, "640,480", s_message);


static const char fn_message[]		= "内存共享文件名称，默认为 source.bin";
DEFINE_string(fn, "source.bin", fn_message);


static const char fs_message[] = "分配给共享内存的空间大小，默认为：1920 x 1080 x 4 = 829400 Bytes";
DEFINE_uint32(fs, 1920 * 1080 * 4, fs_message);

static const char show_message[] = "是否显示视频窗口，用于调试";
DEFINE_bool(show, false, show_message);


std::string GetUsageMessage()
{
    std::stringstream info("Video Source Shared");

    info << "[OPTION]" << std::endl;
    info << "    -h     " << h_message << std::endl;
    info << "    -i     " << i_message << std::endl;
    info << "    -s     " << s_message << std::endl;
    info << "    -fn    " << fn_message << std::endl;
    info << "    -fs    " << fs_message << std::endl;
    info << "    -show  " << show_message << std::endl;
    
    return info.str();
};

bool ParseAndCheckCommandLine(int argc, char* argv[]) 
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, false);
    if (FLAGS_h) 
    {
        std::cout << GetUsageMessage() << std::endl;
        return false;
    }

    //if (FLAGS_i.empty()) {
    //    throw std::logic_error("参数 -i 不能这空");
    //    return false;
    //}

    return true;
}