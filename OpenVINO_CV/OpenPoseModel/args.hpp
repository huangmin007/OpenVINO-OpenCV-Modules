#pragma once

#include <iostream>
#include <gflags/gflags.h>
#include <inference_engine.hpp>


static const char h_message[] = "������Ϣ";
DEFINE_bool(h, false, h_message);


static const char m_message[] = "�������ƹ���ģ�ͣ�.xml���ļ���·����";
DEFINE_string(m, "models/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml", m_message);
static const char d_message[] = "[��ѡ]ָ������ ������������ ��Ŀ���豸";
DEFINE_string(d, "GPU", d_message);
static const char performance_counter_message[] = "[��ѡ]����ÿ�����ܱ���";
DEFINE_bool(pc, false, performance_counter_message);


static const char um_message[] = "[��ѡ]�����ʾ�ļ������б�";
DEFINE_string(um, "", um_message);


static const char rfn_message[] = "���ڶ�ȡAI�����������ڴ湲�����ƣ�Ĭ��Ϊ source.bin";
DEFINE_string(rfn, "source.bin", rfn_message);
static const char wfn_message[] = "���ڴ洢AI����������ݵ��ڴ湲�����ƣ� Ĭ��Ϊsource.raw";
DEFINE_string(wfn, "source.raw", wfn_message);
static const char wfs_message[] = "���ڴ洢AI����������ݵ��ڴ��С��Ĭ��Ϊ 1024 * 1024 Bytes";
DEFINE_uint32(wfs, 1024 * 1024, "���ڴ洢AI����������ݵ��ڴ��С��Ĭ��Ϊ 1024 * 1024 Bytes");


//static const char s_message[] = "ʹ�ñ���";
//DEFINE_double(s, 0.25, s_message);

static const char output_message[] = "[��ѡ]����ƶϽ����Ϊԭʼֵ��";
DEFINE_bool(output, false, output_message);

static const char show_message[] = "�Ƿ���ʾ��Ƶ���ڣ����ڵ���";
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

//��ȡ����Ŀɼ����豸�б�
static std::string GetAvailableDevices()
{
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();

    std::stringstream info;
    info << "����Ŀ���豸:";
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
        throw std::logic_error("���� -m �������");
        return false;
    }

    return true;
}