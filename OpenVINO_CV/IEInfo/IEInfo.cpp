// IEInfo.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iomanip>
#include <inference_engine.hpp>
#include <ie_plugin_config.hpp>

#include "cmdline.h"
#include "static_functions.hpp"


int main(int argc, char** argv)
{
    try
    {
        cmdline::parser args;
        args.add<std::string>("model", 'm', "输入模型名称", false, "");
        args.parse(argc, argv);

        //-------------------------------- 版本信息 --------------------------------
        const InferenceEngine::Version* version = InferenceEngine::GetInferenceEngineVersion();

        LOG("INFO") << "[Inference Engine]" << std::endl;
        LOG("INFO") << "Major:" << version->apiVersion.major << std::endl;
        LOG("INFO") << "Minor:" << version->apiVersion.minor << std::endl;
        LOG("INFO") << "Version:" << version << std::endl;
        LOG("INFO") << "BuildNumber:" << version->buildNumber << std::endl;
        LOG("INFO") << "Description:" << version->description << std::endl;
        std::cout << std::endl;

        InferenceEngine::Core ie;
        //-------------------------------- 网络模型信息 --------------------------------
        std::map<std::string, std::string> model = space::ParseArgsForModel(args.get<std::string>("model"));
        if (args.exist("model") && !model["model"].empty() && !model["path"].empty())
        {
            LOG("INFO") << "[Network Model Infomation] " << model["model"] << std::endl;
            InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork(model["path"]);

            LOG("INFO") << "Newtork Name:" << cnnNetwork.getName() << std::endl;
            //------------Input-----------
            LOG("INFO") << "Input Layer" << std::endl;
            InferenceEngine::InputsDataMap inputsInfo = cnnNetwork.getInputsInfo();
            for (const auto& input : inputsInfo)
            {
                InferenceEngine::SizeVector inputDims = input.second->getTensorDesc().getDims();

                std::stringstream shape; shape << "[";
                for (int i = 0; i < inputDims.size(); i++)
                    shape << inputDims[i] << (i != inputDims.size() - 1 ? "x" : "]");

                LOG("INFO") << "\tOutput Name:[" << input.first << "]  Shape:" << shape.str() << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
            }

            //------------Output-----------
            LOG("INFO") << "Output Layer" << std::endl;
            InferenceEngine::OutputsDataMap outputsInfo = cnnNetwork.getOutputsInfo();
            for (const auto& output : outputsInfo)
            {
                InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
                std::stringstream shape; shape << "[";
                for (int i = 0; i < outputDims.size(); i++)
                    shape << outputDims[i] << (i != outputDims.size() - 1 ? "x" : "]");

                LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << shape.str() << "  Precision:[" << output.second->getPrecision() << "]" << std::endl;
            }

            std::cout << std::endl;
        }

        //-------------------------------- 支持的硬件设备信息 --------------------------------
        LOG("INFO") << "[Support Target Devices]";
        std::vector<std::string> devices = ie.GetAvailableDevices();
        for (const auto& device : devices)
        {
            std::string deviceFullName = ie.GetMetric(device, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            LOG("INFO") << "[" << device << "] " << deviceFullName << std::endl;
        }
        std::cout << std::endl;
    }
    catch (const std::exception& ex)
    {
        LOG("ERROR") << ex.what() << std::endl;
    }
    catch (...)
    {
        LOG("ERROR") << "未知错误/异常 ... " << std::endl;
    }

    system("pause");
    std::cout << std::endl;

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
