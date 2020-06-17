#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
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
    void MemorySharedBlob(InferenceEngine::InferRequest::Ptr& requestPtr, const std::pair<std::string, LPVOID>& shared, size_t offset = 0);

	template <typename T>
	void MatU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);

   
    template <typename T>
    std::ostream& operator << (std::ostream& stream, const std::vector<T>& v)
    {
        stream << "[";
        for (int i = 0; i < v.size(); i++)
            stream << v[i] << (i != v.size() - 1 ? "," : "]");

        return stream;
    }

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
                LOG("INFO") << "\tOutput Name:[" << input.first << "]  Shape:" << inputDims << "  Precision:[" << input.second->getPrecision() << "]" << std::endl;
            }

            //------------Output-----------
            LOG("INFO") << "Output Layer" << std::endl;
            InferenceEngine::OutputsDataMap outputsInfo = cnnNetwork.getOutputsInfo();
            for (const auto& output : outputsInfo)
            {
                InferenceEngine::SizeVector outputDims = output.second->getTensorDesc().getDims();
                LOG("INFO") << "\tOutput Name:[" << output.first << "]  Shape:" << outputDims << "  Precision:[" << output.second->getPrecision() << "]" << std::endl;
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
        LOG("INFO") << "CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0) return false;

        ///映射到当前进程的地址空间视图
        buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);  //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
        ErrorID = GetLastErrorFormatMessage(message);                   //GetLastError 检查 MapViewOfFile 状态
        LOG("INFO") << "MapViewOfFile     [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0) return false;

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

        if (ErrorID != 0) return false;

        ///映射到当前进程的地址空间视图
        buffer = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
        //GetLastError 检查 MapViewOfFile 状态
        ErrorID = GetLastErrorFormatMessage(message);
        LOG("INFO") << "MapViewOfFile   [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0) return false;

        return true;
    }

    using OutputSharedLayer = std::tuple<std::string, HANDLE, LPVOID>;
    using OutputSharedLayers = std::vector<OutputSharedLayer>;

    inline const OutputSharedLayer CreateSharedBlob(
        InferenceEngine::InferRequest::Ptr& request, const std::string name, 
        size_t reserve_byte_size = 32)
    {
        if(request == nullptr || name.empty())  throw std::invalid_argument("参数不能为空");

        InferenceEngine::Blob::Ptr blob = request->GetBlob(name);
        if (blob == nullptr) throw std::invalid_argument("未获取到 Blob 数据");

        size_t buffer_size = blob->byteSize() + reserve_byte_size;
        InferenceEngine::TensorDesc tensor = blob->getTensorDesc();
        const InferenceEngine::Precision precision = tensor.getPrecision();

        LPVOID buffer; HANDLE handle;
        if (CreateOnlyWriteMapFile(handle, buffer, buffer_size, name.c_str()))
        {
            switch (precision)
            {
            case InferenceEngine::Precision::U64:	//uint64_t
                request->SetBlob(name, InferenceEngine::make_shared_blob<uint64_t>(tensor, (uint64_t*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::I64:	//int64_t
                request->SetBlob(name, InferenceEngine::make_shared_blob<int64_t>(tensor, (int64_t*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::FP32:	//float
                request->SetBlob(name, InferenceEngine::make_shared_blob<float>(tensor, (float*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::I32:	//int32_t
                request->SetBlob(name, InferenceEngine::make_shared_blob<int32_t>(tensor, (int32_t*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::U16:	//uint16_t
                request->SetBlob(name, InferenceEngine::make_shared_blob<uint16_t>(tensor, (uint16_t*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::I16:	//int16_t
            case InferenceEngine::Precision::Q78:	//int16_t, uint16_t
            case InferenceEngine::Precision::FP16:	//int16_t, uint16_t	
                request->SetBlob(name, InferenceEngine::make_shared_blob<int16_t>(tensor, (int16_t*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::U8:	//uint8_t
            case InferenceEngine::Precision::BOOL:	//uint8_t
                request->SetBlob(name, InferenceEngine::make_shared_blob<uint8_t>(tensor, (uint8_t*)buffer + reserve_byte_size));
                break;
            case InferenceEngine::Precision::I8:	//int8_t
            case InferenceEngine::Precision::BIN:	//int8_t, uint8_t
                request->SetBlob(name, InferenceEngine::make_shared_blob<int8_t>(tensor, (int8_t*)buffer + reserve_byte_size));
                break;
            default:
                LOG("WARN") << "共享映射数据，未处理的输出精度：" << precision << std::endl;
                throw std::invalid_argument("共享映射数据，未处理的输出精度：" + precision);
            }
        }
        else
        {
            LOG("ERROR") << "共享内存块 " << name << "创建失败 ... " << std::endl;
            throw std::logic_error("共享内存块创建失败 ... ");
        }
        
        return { name, handle, buffer};
    }

    inline const OutputSharedLayers CreateSharedMapping(
        const std::vector<InferenceEngine::InferRequest::Ptr>& requestPtrs, 
        const std::vector<std::pair<std::string, std::size_t>>& shared_layers_info, 
        bool is_mapping_blob = true, size_t reserve_byte_size = 32)
    {
        OutputSharedLayers shared_layers;

        for (const auto& shared : shared_layers_info)
        {
            LPVOID buffer;
            HANDLE handle;


            if (CreateOnlyWriteMapFile(handle, buffer, shared.second + reserve_byte_size, shared.first.c_str()))
            {
                shared_layers.push_back({shared.first, handle, buffer });

                //将输出层映射到内存共享
                //重新设置指定输出层 Blob, 将指定输出层数据指向为共享指针，实现该层数据共享
                //多个推断对象，输出映射到同一个共享位置？？
                if (is_mapping_blob)
                    for (auto requestPtr : requestPtrs)
                        MemorySharedBlob(requestPtr, { shared.first , buffer }, reserve_byte_size);

                LOG("INGO") << "共享内存 " << shared.first << " 映射：" << std::boolalpha << is_mapping_blob << std::endl;
            }
            else
            {
                LOG("ERROR") << "共享内存块创建失败 ... " << std::endl;
                throw std::logic_error("共享内存块创建失败 ... ");
            }
        }

        return shared_layers;
    }


    inline void DisposeSharedMapping(OutputSharedLayers &shared_layers)
    {
        for (auto& shared : shared_layers)
        {
            UnmapViewOfFile(std::get<2>(shared));   //buffer
            CloseHandle(std::get<1>(shared));       //handle

            std::get<1>(shared) = NULL;
            std::get<2>(shared) = NULL;
        }

        shared_layers.clear();
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
            {"path", ""},           //模型文件路径
            {"model", ""},          //模型名称
            {"fp", "FP32"},         //模型文件精度
            {"device", "CPU"},      //模型加载到使用的硬件
            {"labelpath", ""},      //模型标签文件路径
            {"full", ""}            //原始参数
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
    /// 读取标签文件
    /// </summary>
    /// <param name="labels_file"></param>
    /// <returns></returns>
    inline const std::vector<std::string> ReadLabels(const std::string& labels_file)
    {
        std::vector<std::string> labels;        
        std::fstream file(labels_file, std::fstream::in);

        if (!file.is_open())
        {
            LOG("WARN") << "标签文件不存在 " << labels_file << " 文件 ...." << std::endl;
        }
        else
        {
            std::string line;
            while (std::getline(file, line))
            {
                labels.push_back(line);
            }
            file.close();
        }

        return labels;
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
    
    /// <summary>
    /// 绘制检测对象的边界
    /// </summary>
    /// <param name="frame"></param>
    /// <param name="rect"></param>
    static void DrawObjectBound(cv::Mat frame, const cv::Rect& src_rect)
    {
        auto rect = src_rect & cv::Rect(0, 0, frame.cols, frame.rows);

        float scale = 0.2f;
        int shift = 0;
        int thickness = 2;
        int lineType = cv::LINE_AA;
        cv::Scalar border_color(0, 255, 0);

        cv::rectangle(frame, rect, border_color, 1);
        cv::Point h_width = cv::Point(rect.width * scale, 0);   //水平宽度
        cv::Point v_height = cv::Point(0, rect.height * scale); //垂直高度

        cv::Point left_top(rect.x, rect.y);
        cv::line(frame, left_top, left_top + h_width, border_color, thickness, lineType, shift);     //-
        cv::line(frame, left_top, left_top + v_height, border_color, thickness, lineType, shift);    //|

        cv::Point left_bottom(rect.x, rect.y + rect.height - 1);
        cv::line(frame, left_bottom, left_bottom + h_width, border_color, thickness, lineType, shift);       //-
        cv::line(frame, left_bottom, left_bottom - v_height, border_color, thickness, lineType, shift);      //|

        cv::Point right_top(rect.x + rect.width - 1, rect.y);
        cv::line(frame, right_top, right_top - h_width, border_color, thickness, lineType, shift);   //-
        cv::line(frame, right_top, right_top + v_height, border_color, thickness, lineType, shift);  //|

        cv::Point right_bottom(rect.x + rect.width - 1, rect.y + rect.height - 1);
        cv::line(frame, right_bottom, right_bottom - h_width, border_color, thickness, lineType, shift);   //-
        cv::line(frame, right_bottom, right_bottom - v_height, border_color, thickness, lineType, shift);  //|
    }

    /// <summary>
    /// 获取精度所对应的数据类型字节大小
    /// </summary>
    /// <param name="precision"></param>
    /// <returns></returns>
    inline size_t GetPrecisionOfSize(const InferenceEngine::Precision& precision)
    {
        switch (precision)
        {
        case InferenceEngine::Precision::U64:	//uint64_t
        case InferenceEngine::Precision::I64:	//int64_t
            return sizeof(uint64_t);

        case InferenceEngine::Precision::FP32:	//float
            return sizeof(float);

        case InferenceEngine::Precision::I32:	//int32_t
            return sizeof(int32_t);

        case InferenceEngine::Precision::U16:	//uint16_t
        case InferenceEngine::Precision::I16:	//int16_t
        case InferenceEngine::Precision::Q78:	//int16_t, uint16_t
        case InferenceEngine::Precision::FP16:	//int16_t, uint16_t	
            return sizeof(uint16_t);

        case InferenceEngine::Precision::U8:	//uint8_t
        case InferenceEngine::Precision::BOOL:	//uint8_t
        case InferenceEngine::Precision::I8:	//int8_t
        case InferenceEngine::Precision::BIN:	//int8_t, uint8_t
            return sizeof(uint8_t);
        }

        return sizeof(uint8_t);
    }

    /// <summary>
    /// 内存共享 Blob (网络层)
    /// </summary>
    /// <param name="requestPtr"></param>
    /// <param name="shared">{name, void*}</param>
    inline void MemorySharedBlob(InferenceEngine::InferRequest::Ptr& requestPtr, const std::pair<std::string, LPVOID>& shared, size_t offset)
    {
        InferenceEngine::TensorDesc tensor = requestPtr->GetBlob(shared.first)->getTensorDesc();
        const InferenceEngine::Precision precision  = tensor.getPrecision();

        switch (precision)
        {
        case InferenceEngine::Precision::U64:	//uint64_t
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<uint64_t>(tensor, (uint64_t*)shared.second + offset));
            break;
        case InferenceEngine::Precision::I64:	//int64_t
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int64_t>(tensor, (int64_t*)shared.second + offset));
            break;
        case InferenceEngine::Precision::FP32:	//float
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<float>(tensor, (float*)shared.second + offset));
            break;
        case InferenceEngine::Precision::I32:	//int32_t
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int32_t>(tensor, (int32_t*)shared.second + offset));
            break;
        case InferenceEngine::Precision::U16:	//uint16_t
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<uint16_t>(tensor, (uint16_t*)shared.second + offset));
            break;
        case InferenceEngine::Precision::I16:	//int16_t
        case InferenceEngine::Precision::Q78:	//int16_t, uint16_t
        case InferenceEngine::Precision::FP16:	//int16_t, uint16_t	
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int16_t>(tensor, (int16_t*)shared.second + offset));
            break;
        case InferenceEngine::Precision::U8:	//uint8_t
        case InferenceEngine::Precision::BOOL:	//uint8_t
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<uint8_t>(tensor, (uint8_t*)shared.second + offset));
            break;
        case InferenceEngine::Precision::I8:	//int8_t
        case InferenceEngine::Precision::BIN:	//int8_t, uint8_t
            requestPtr->SetBlob(shared.first, InferenceEngine::make_shared_blob<int8_t>(tensor, (int8_t*)shared.second + offset));
            break;
        default:
            LOG("WARN") << "共享映射数据，未处理的输出精度：" << precision << std::endl;
            throw std::invalid_argument("共享映射数据，未处理的输出精度：" + precision);
        }
    }


    /// <summary>
    /// 输出参数信息
    /// </summary>
    /// <param name="value"></param>
    static void PrintParameterValue(const InferenceEngine::Parameter& value)
    {
        if (value.is<bool>())
        {
            std::cout << std::boolalpha << value.as<bool>() << std::noboolalpha << std::endl;
        }
        else if (value.is<int>())
        {
            std::cout << value.as<int>() << std::endl;
        }
        else if (value.is<unsigned int>())
        {
            std::cout << value.as<unsigned int>() << std::endl;
        }
        else if (value.is<float>())
        {
            std::cout << value.as<float>() << std::endl;
        }
        else if (value.is<std::string>())
        {
            std::string stringValue = value.as<std::string>();
            std::cout << (stringValue.empty() ? "\"\"" : stringValue) << std::endl;
        }
        else if (value.is<std::vector<std::string> >())
        {
            std::cout << value.as<std::vector<std::string> >() << std::endl;
        }
        else if (value.is<std::vector<int> >())
        {
            std::cout << value.as<std::vector<int> >() << std::endl;
        }
        else if (value.is<std::vector<float> >())
        {
            std::cout << value.as<std::vector<float> >() << std::endl;
        }
        else if (value.is<std::vector<unsigned int> >())
        {
            std::cout << value.as<std::vector<unsigned int> >() << std::endl;
        }
        else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int>>())
        {
            auto values = value.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
            std::cout << "{ ";
            std::cout << std::get<0>(values) << ", ";
            std::cout << std::get<1>(values) << ", ";
            std::cout << std::get<2>(values);
            std::cout << " }";
            std::cout << std::endl;
        }
        else if (value.is<std::tuple<unsigned int, unsigned int> >())
        {
            auto values = value.as<std::tuple<unsigned int, unsigned int> >();
            std::cout << "{ ";
            std::cout << std::get<0>(values) << ", ";
            std::cout << std::get<1>(values);
            std::cout << " }";
            std::cout << std::endl;
        }
        else
        {
            std::cout << "UNSUPPORTED TYPE" << std::endl;
        }
    }

    
}