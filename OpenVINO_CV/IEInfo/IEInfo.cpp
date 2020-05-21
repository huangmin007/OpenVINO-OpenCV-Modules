// IEInfo.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iomanip>
#include <inference_engine.hpp>
#include <ie_plugin_config.hpp>

#define WIDTH 16
#define LOG(type)    (std::cout << "[" << std::setw(5) << std::right << type << "] ")


int main()
{
    //-------------------------------- 版本信息 --------------------------------
    const InferenceEngine::Version* version = InferenceEngine::GetInferenceEngineVersion();

    std::cout.setf(std::ios::left);
    std::cout << "[InferenceEngine]" << std::endl;
    std::cout << std::setw(WIDTH) << "Version:"  << version << std::endl;
    std::cout << std::setw(WIDTH) << "Major:" << version->apiVersion.major << std::endl;
    std::cout << std::setw(WIDTH) << "Minor:" << version->apiVersion.minor << std::endl;
    std::cout << std::setw(WIDTH) << "BuildNumber:" << version->buildNumber << std::endl;
    std::cout << std::setw(WIDTH) << "Description:" << version->description << std::endl;
    std::cout << std::endl;

    //-------------------------------- 支持的硬件设备 --------------------------------
    InferenceEngine::Core core;

    //GetAvailableDevices()
    std::vector<std::string> devices = core.GetAvailableDevices();
    std::cout << std::setw(WIDTH) << "Support Devices:";
    for (const auto& device : devices)
        std::cout << " " << device;
    std::cout << std::endl;
    
    //GetConfig()
    //bool dumpDotFile = core.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>();

    //GetMetric()
    for (int i = 0; i < devices.size(); i++)
    {
        std::string deviceName = core.GetMetric(devices[i], METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
        std::cout << "\t\t" << devices[i] << "::" << deviceName << std::endl;
    }
    std::cout << std::endl;

    system("pause");
    return EXIT_SUCCESS;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
