#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <Windows.h>
#include <winioctl.h>
#include <opencv2\opencv.hpp>
#include <inference_engine.hpp>


#define LOG(type)       (std::cout << "[" << std::setw(5) << std::right << type << "] ")

/// <summary>
/// ms
/// </summary>
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;


/// <summary>
/// OpenCV cv::waitKey 默认参考延时
/// </summary>
static const int WaitKeyDelay = 33;

/// <summary>
/// 控制台线程运行状态，会监听 SetConsoleCtrlHandler 改变
/// </summary>
static bool IsRunning = true;
//extern bool IsRunning;

namespace space
{
	void InferenceEngineInfomation();
	BOOL CloseHandlerRoutine(DWORD CtrlType);
	DWORD GetLastErrorFormatMessage(LPVOID& message);
	bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name);
	bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name);

	bool ParseArgForSize(const std::string& size, int& width, int& height);
	const std::map<std::string, std::string> ParseArgsForModel(const std::string& args);
	const std::vector<std::string> SplitString(const std::string& src, const char delimiter);

	float CosineDistance(const std::vector<float>& fv1, const std::vector<float>& fv2);

	template <typename T>
	void MatU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);


    /// <summary>
    /// 输出 InferenceEngine 引擎版本，及可计算设备列表
    /// </summary>
    inline void InferenceEngineInfomation(std::string modelInfo = "")
    {
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
        std::map<std::string, std::string> model = ParseArgsForModel(modelInfo);
        if (!model["model"].empty() && !model["path"].empty())
        {
            LOG("INFO") << "[Network Model Infomation] " << std::endl;
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

        //-------------------------------- 支持的硬件设备 --------------------------------
        LOG("INFO") << "[Support Target Devices]";
        std::vector<std::string> devices = ie.GetAvailableDevices();
        for (const auto& device : devices)
        {
            std::string deviceFullName = ie.GetMetric(device, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            LOG("INFO") << "[" << device << "] " << deviceFullName << std::endl;
        }
        std::cout << std::endl;

        system("pause");
    }

#if test
    /// <summary>
    /// [参考用的]创建通用参数解析；已经添加了 help/info/input/output/model/async/show 参数
    /// </summary>
    /// <param name="args">参数解析对象</param>
    /// <param name="program_name">程序名称</param>
    /// <param name="default_model">默认的模型配置参数</param>
    /// <returns></returns>
    static void CreateGeneralCmdLine(cmdline::parser& args, const std::string& program_name, const std::string& default_model)
    {
        args.add("help", 'h', "参数说明");
        args.add("info", 0, "Inference Engine Infomation");

        args.add<std::string>("input", 'i', "输入源参数，格式：(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
        args.add<std::string>("output", 'o', "输出源参数，格式：(shared|console|socket)[:value[:value[:...]]]", false, "shared:o_source.bin");// "shared:o_source.bin");
        args.add<std::string>("model", 'm', "用于 AI识别检测 的 网络模型名称/文件(.xml)和目标设备，格式：(AI模型名称)[:精度[:硬件]]，"
            "示例：face-detection-adas-0001:FP16:CPU 或 face-detection-adas-0001:FP16:HETERO:CPU,GPU", default_model.empty(), default_model);

        args.add<bool>("async", 0, "是否异步分析识别", false, true);

#ifdef _DEBUG
        args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, true);
#else
        args.add<bool>("show", 0, "是否显示视频窗口，用于调试", false, false);
#endif

        args.set_program_name(program_name);
    }
#endif

    /// <summary>
    /// 控制台程序操作关闭或退出处理，WinAPI 函数 SetConsoleCtrlHandler 调用
    /// </summary>
    /// <param name="CtrlType"></param>
    /// <returns></returns>
    inline BOOL CloseHandlerRoutine(DWORD CtrlType)
    {
        std::cout << "SetConsoleCtrlHandler:::" << CtrlType << std::endl;
        switch (CtrlType)
        {
        case CTRL_C_EVENT:          //当用户按下了CTRL+C,或者由GenerateConsoleCtrlEvent API发出
        case CTRL_BREAK_EVENT:      //用户按下CTRL+BREAK, 或者由 GenerateConsoleCtrlEvent API 发出
            if (!IsRunning) exit(0);
            else IsRunning = false;
            break;

        case CTRL_CLOSE_EVENT:      //当试图关闭控制台程序，系统发送关闭消息
        case CTRL_LOGOFF_EVENT:     //用户退出时，但是不能决定是哪个用户
        case CTRL_SHUTDOWN_EVENT:   //当系统被关闭时
            exit(0);
            break;
        }

        return TRUE;
    }


    /// <summary>
    /// 获取 Window API 函数执行状态
    /// </summary>
    /// <param name="message">输出信息文本</param>
    /// <returns></returns>
    inline DWORD GetLastErrorFormatMessage(LPVOID& message)
    {
        DWORD id = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, id, 0, (LPTSTR)&message, 0, NULL);

        return id;
    };


    /// <summary>
    /// 创建只写的 内存共享文件句柄 及 内存空间映射视图
    /// </summary>
    /// <param name="handle">内存文件句柄</param>
    /// <param name="buffer">内存映射到当前程序的数据指针</param>
    /// <param name="size">分配的内存空间大小，字节为单</param>
    /// <param name="name">内存共享文件的名称，其它进程读写使用该名称</param>
    /// <returns>创建成功 返回 true</returns>
    inline bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name)
    {
        ///创建共享内存
        handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCSTR)name);
        LPVOID message = NULL;
        DWORD ErrorID = GetLastErrorFormatMessage(message);     //GetLastError 检查 CreateFileMapping 状态
        LOG("INFO") << "CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

        if (ErrorID != 0 || handle == NULL) return false;

        ///映射到当前进程的地址空间视图
        buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);  //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
        ErrorID = GetLastErrorFormatMessage(message);                   //GetLastError 检查 MapViewOfFile 状态
        LOG("INFO") << "MapViewOfFile     [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0 || buffer == NULL)
        {
            CloseHandle(handle);
            return false;
        }

        return true;
    }


    /// <summary>
    /// 获取只读的 内存共享文件句柄 及 内存空间映射视图
    /// </summary>
    /// <param name="handle">内存文件句柄</param>
    /// <param name="buffer">内存映射到当前程序的数据指针</param>
    /// <param name="name">需要读取的内存共享文件的名称</param>
    /// <returns>获取成功 返回 true</returns>
    inline bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name)
    {
        ///创建共享内存
        handle = OpenFileMapping(FILE_MAP_READ, FALSE, (LPCSTR)name);   //PAGE_READONLY
        //GetLastError 检查 OpenFileMapping 状态
        LPVOID message = NULL;
        DWORD ErrorID = GetLastErrorFormatMessage(message);
        LOG("INFO") << "OpenFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

        if (ErrorID != 0 || handle == NULL) return false;

        ///映射到当前进程的地址空间视图
        buffer = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
        //GetLastError 检查 MapViewOfFile 状态
        ErrorID = GetLastErrorFormatMessage(message);
        LOG("INFO") << "MapViewOfFile   [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0 || buffer == NULL)
        {
            CloseHandle(handle);
            return false;
        }

        return true;
    }



    /// <summary>
    /// 解析 Size 参数，格式：width(x|,)height
    /// </summary>
    /// <param name="size"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <returns></returns>
    inline bool ParseArgForSize(const std::string& size, int& width, int& height)
    {
        if (size.empty())return false;

        int index = size.find('x') != std::string::npos ? size.find('x') : size.find(',');
        if (index == std::string::npos) return false;

        width = std::stoi(size.substr(0, index));
        height = std::stoi(size.substr(index + 1));

        return true;
    }

    /// <summary>
    /// 解析输入模型参数
    /// <para>输入格式为：(AI模型名称)[:精度[:硬件]]，示例：face-detection-adas-0001:FP32:CPU 或 face-detection-adas-0001:FP16:GPU </para>
    /// </summary>
    /// <param name="args"></param>
    /// <returns>返回具有 ["model","fp","device","path"] 属性的 std::map 数据</returns>
    inline const std::map<std::string, std::string> ParseArgsForModel(const std::string& args)
    {
        std::map<std::string, std::string> argMap
        {
            {"path", ""},
            {"model", ""},
            {"fp", "FP16"},
            {"device", "CPU"},
            {"full", ""}
        };

        std::vector<std::string> model = SplitString(args, ':');
        size_t length = model.size();
        if (length <= 0) return argMap;

        //model
        if (length >= 1)    argMap["model"] = model[0];
        //model[:FP]
        if (length >= 2)    argMap["fp"] = model[1];
        //model[:FP[:device]]
        if (length >= 3)    argMap["device"] = model[2];
        //model[:fp[:device(HETERO:CPU,GPU)]]
        if (length >= 4)     argMap["device"] = model[2] + ":" + model[3];

        argMap["full"] = argMap["model"] + ":" + argMap["fp"] + ":" + argMap["device"];
        argMap["path"] = "models\\" + argMap["model"] + "\\" + argMap["fp"] + "\\" + argMap["model"] + ".xml";

        return argMap;
    }

    /// <summary>
    /// 将字符分割成数组
    /// </summary>
    /// <param name="src">源始字符串</param>
    /// <param name="delimiter">定界符</param>
    /// <returns></returns>
    inline const std::vector<std::string> SplitString(const std::string& src, const char delimiter = ':')
    {
        std::vector<std::string> vec;
        size_t start = 0, end = src.find(delimiter);

        while (end != std::string::npos)
        {
            vec.push_back(src.substr(start, end - start));

            start = end + 1;
            end = src.find(delimiter, start);
        }

        if (start != src.length())
            vec.push_back(src.substr(start));

        return vec;
    }

    /// <summary>
    /// 优化后的
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="orig_image"></param>
    /// <param name="blob"></param>
    /// <param name="batchIndex"></param>
    template <typename T>
    inline void MatU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex)
    {
        InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
        const size_t width = blobSize[3];
        const size_t height = blobSize[2];
        const size_t channels = blobSize[1];

        if (static_cast<size_t>(orig_image.channels()) != channels)
            THROW_IE_EXCEPTION << "网络输入和图像的通道数必须匹配";

        T* blob_data = blob->buffer().as<T*>();

        cv::Mat resized_image(orig_image);
        if (static_cast<int>(width) != orig_image.size().width || static_cast<int>(height) != orig_image.size().height) 
        {
            cv::resize(orig_image, resized_image, cv::Size(width, height));
        }

        int batchOffset = batchIndex * width * height * channels;

        if (channels == 3)
        {
            int offset = 0;
            for (int h = 0; h < height; h++)
            {
                T* row_pixels = resized_image.data + h * resized_image.step;
                for (int w = 0; w < width; w++)
                {
                    offset = h * width + w;
                    blob_data[batchOffset + offset] = row_pixels[0];
                    blob_data[batchOffset + offset + (width * height * 1)] = row_pixels[1];
                    blob_data[batchOffset + offset + (width * height * 2)] = row_pixels[2];

                    row_pixels += 3;
                }
            }
        }
        else if (channels == 1) 
        {
            for (size_t h = 0; h < height; h++)
            {
                for (size_t w = 0; w < width; w++) 
                {
                    blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
                }
            }
        }
        else 
        {
            THROW_IE_EXCEPTION << "通道数量不受支持 ... ";
        }
    }


    /// <summary>
    /// 通过新的Blob指针包装存储在传递的 cv::Mat 对象内部的数据，没有发生内存分配。该 Blob 仅指向已经存在的 cv::Mat 数据
    /// </summary>
    /// <param name="mat">给定具有图像数据的 cv::Mat 对象</param>
    /// <returns>返回 Blob 指针</returns>
    inline InferenceEngine::Blob::Ptr WrapMat2Blob(const cv::Mat& frame)
    {
        size_t channels = frame.channels();
        size_t height = frame.size().height;
        size_t width = frame.size().width;

        size_t strideH = frame.step.buf[0];
        size_t strideW = frame.step.buf[1];

        bool is_dense = strideW == channels && strideH == channels * width;
        if (!is_dense)
            THROW_IE_EXCEPTION << "不支持从非稠密 cv::Mat 转换";

        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
            { 1, channels, height, width }, InferenceEngine::Layout::NHWC);

        return InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data);
    }

    /// <summary>
    /// 余弦相似比较。 计算余弦相似, 0 ~ 1 距离，距离越小越相似，0表示夹角为0°，1表示夹角为90°
    /// </summary>
    /// <param name="fv1"></param>
    /// <param name="fv2"></param>
    /// <returns></returns>
    inline float CosineDistance(const std::vector<float>& fv1, const std::vector<float>& fv2)
    {
        float dot = 0;
        float sum2 = 0;
        float sum3 = 0;
        for (int i = 0; i < fv1.size(); i++)
        {
            dot += fv1[i] * fv2[i];
            sum2 += pow(fv1[i], 2);
            sum3 += pow(fv2[i], 2);
        }
        float norm = sqrt(sum2) * sqrt(sum3);
        float similarity = dot / norm;
        float dist = acos(similarity) / CV_PI;

        return dist;
    }


    inline std::size_t getTensorWidth(const InferenceEngine::TensorDesc& desc)
    {
        const auto& layout = desc.getLayout();
        const auto& dims = desc.getDims();
        const auto& size = dims.size();
        if ((size >= 2) &&
            (layout == InferenceEngine::Layout::NCHW ||
                layout == InferenceEngine::Layout::NHWC ||
                layout == InferenceEngine::Layout::NCDHW ||
                layout == InferenceEngine::Layout::NDHWC ||
                layout == InferenceEngine::Layout::OIHW ||
                layout == InferenceEngine::Layout::CHW ||
                layout == InferenceEngine::Layout::HW))
        {
            // 无论布局如何，尺寸均以固定顺序存储
            return dims.back();
        }
        else {
            THROW_IE_EXCEPTION << "张量没有宽度尺寸";
        }
        return 0;
    }

    inline std::size_t getTensorHeight(const InferenceEngine::TensorDesc& desc) 
    {
        const auto& layout = desc.getLayout();
        const auto& dims = desc.getDims();
        const auto& size = dims.size();
        if ((size >= 2) &&
            (layout == InferenceEngine::Layout::NCHW ||
                layout == InferenceEngine::Layout::NHWC ||
                layout == InferenceEngine::Layout::NCDHW ||
                layout == InferenceEngine::Layout::NDHWC ||
                layout == InferenceEngine::Layout::OIHW ||
                layout == InferenceEngine::Layout::CHW ||
                layout == InferenceEngine::Layout::HW)) {
            // Regardless of layout, dimensions are stored in fixed order
            return dims.at(size - 2);
        }
        else {
            THROW_IE_EXCEPTION << "张量没有高度尺寸";
        }
        return 0;
    }

    inline std::size_t getTensorChannels(const InferenceEngine::TensorDesc& desc) 
    {
        const auto& layout = desc.getLayout();
        if (layout == InferenceEngine::Layout::NCHW ||
            layout == InferenceEngine::Layout::NHWC ||
            layout == InferenceEngine::Layout::NCDHW ||
            layout == InferenceEngine::Layout::NDHWC ||
            layout == InferenceEngine::Layout::C ||
            layout == InferenceEngine::Layout::CHW ||
            layout == InferenceEngine::Layout::NC ||
            layout == InferenceEngine::Layout::CN) {
            // Regardless of layout, dimensions are stored in fixed order
            const auto& dims = desc.getDims();
            switch (desc.getLayoutByDims(dims)) {
            case InferenceEngine::Layout::C:     return dims.at(0);
            case InferenceEngine::Layout::NC:    return dims.at(1);
            case InferenceEngine::Layout::CHW:   return dims.at(0);
            case InferenceEngine::Layout::NCHW:  return dims.at(1);
            case InferenceEngine::Layout::NCDHW: return dims.at(1);
            case InferenceEngine::Layout::SCALAR:   // [[fallthrough]]
            case InferenceEngine::Layout::BLOCKED:  // [[fallthrough]]
            default:
                THROW_IE_EXCEPTION << "张量没有通道尺寸";
            }
        }
        else {
            THROW_IE_EXCEPTION << "张量没有通道尺寸";
        }
        return 0;
    }

    inline std::size_t getTensorBatch(const InferenceEngine::TensorDesc& desc)
    {
        const auto& layout = desc.getLayout();
        if (layout == InferenceEngine::Layout::NCHW ||
            layout == InferenceEngine::Layout::NHWC ||
            layout == InferenceEngine::Layout::NCDHW ||
            layout == InferenceEngine::Layout::NDHWC ||
            layout == InferenceEngine::Layout::NC ||
            layout == InferenceEngine::Layout::CN) {
            // Regardless of layout, dimensions are stored in fixed order
            const auto& dims = desc.getDims();
            switch (desc.getLayoutByDims(dims)) {
            case InferenceEngine::Layout::NC:    return dims.at(0);
            case InferenceEngine::Layout::NCHW:  return dims.at(0);
            case InferenceEngine::Layout::NCDHW: return dims.at(0);
            case InferenceEngine::Layout::CHW:      // [[fallthrough]]
            case InferenceEngine::Layout::C:        // [[fallthrough]]
            case InferenceEngine::Layout::SCALAR:   // [[fallthrough]]
            case InferenceEngine::Layout::BLOCKED:  // [[fallthrough]]
            default:
                THROW_IE_EXCEPTION << "张量没有批次尺寸";
            }
        }
        else {
            THROW_IE_EXCEPTION << "张量没有批次尺寸";
        }
        return 0;
    }

    /// <summary>
    /// 缩放 cv::Rect 对象
    /// </summary>
    /// <param name="rect"></param>
    /// <param name="scale">放大比例</param>
    /// <param name="dx_offset">按 x 方向偏移系数</param>
    /// <param name="dy_offset">按 y 方向偏移系数</param>
    /// <returns></returns>
    inline cv::Rect ScaleRectangle(const cv::Rect& rect, float scale = 1.2, float dx_offset = 1.0, float dy_offset = 1.0)
    {
        int width = rect.width;
        int height = rect.height;

        int center_x = rect.x + width / 2;
        int center_y = rect.y + height / 2;

        int max_of_size = std::max(width, height);

        int new_width = static_cast<int>(scale * max_of_size);
        int new_height = static_cast<int>(scale * max_of_size);

        cv::Rect new_rect;
        new_rect.x = center_x - static_cast<int>(std::floor(dx_offset * new_width / 2));
        new_rect.y = center_y - static_cast<int>(std::floor(dy_offset * new_height / 2));

        new_rect.width = new_width;
        new_rect.height = new_height;

        return new_rect;
    }

}