#pragma once

#include <iostream>
#include <gflags/gflags.h>
#include <inference_engine.hpp>


static const char h_message[] = "参数信息";
DEFINE_bool(h, false, h_message);


static const char m_message[] = "人体姿势估计模型（.xml）文件的路径。";
DEFINE_string(m, "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml", m_message);
static const char d_message[] = "[可选]指定用于 人体姿势评估 的目标设备";
DEFINE_string(d, "GPU", d_message);
static const char performance_counter_message[] = "[可选]启用每层性能报告";
DEFINE_bool(pc, false, performance_counter_message);


static const char um_message[] = "[可选]最初显示的监视器列表";
DEFINE_string(um, "", um_message);


static const char rfn_message[] = "用于读取AI分析的数据内存共享名称，默认为 source.bin";
DEFINE_string(rfn, "source.bin", rfn_message);
static const char wfn_message[] = "用于存储AI分析结果数据的内存共享名称， 默认为source.raw";
DEFINE_string(wfn, "source.raw", wfn_message);
static const char wfs_message[] = "用于存储AI分析结果数据的内存大小，默认为 1024 * 1024 Bytes";
DEFINE_uint32(wfs, 1024 * 1024, "用于存储AI分析结果数据的内存大小，默认为 1024 * 1024 Bytes");


//static const char s_message[] = "使用比例";
//DEFINE_double(s, 0.25, s_message);

static const char output_message[] = "[可选]输出推断结果作为原始值。";
DEFINE_bool(output, false, output_message);

static const char show_message[] = "是否显示视频窗口，用于调试";
DEFINE_bool(show, false, show_message);




std::string GetUsageMessage()
{
    std::stringstream info("2d_human_pose_estimation");

    info << "[OPTION]" << std::endl;
    info << "    -h     " << h_message << std::endl;
    info << "    -m     " << m_message << std::endl;
    info << "    -d     " << d_message << std::endl;
    info << "    -rfn   " << rfn_message << std::endl;
    info << "    -wfn   " << wfn_message << std::endl;
    info << "    -wfs   " << wfs_message << std::endl;
    info << "    -show  " << show_message << std::endl;

    return info.str();
};

//获取引擎的可计算设备列表
static std::string GetAvailableDevices()
{
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();

    std::stringstream info;
    info << "可用目标设备:";
    for (const auto& device : devices)
        info << "  " << device;

    return info.str();
}

bool ParseAndCheckCommandLine(int argc, char* argv[])
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, false);
    if (FLAGS_h)
    {
        std::cout << GetUsageMessage() << std::endl;
        std::cout << GetAvailableDevices() << std::endl;
        return false;
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("参数 -m 不能这空");
        return false;
    }

    return true;
}