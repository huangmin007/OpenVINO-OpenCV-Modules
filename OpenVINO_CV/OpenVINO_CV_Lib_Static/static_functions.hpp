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
/// OpenCV cv::waitKey Ĭ�ϲο���ʱ
/// </summary>
static const int WaitKeyDelay = 33;

/// <summary>
/// ����̨�߳�����״̬������� SetConsoleCtrlHandler �ı�
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
    /// ��� InferenceEngine ����汾�����ɼ����豸�б�
    /// </summary>
    inline void InferenceEngineInfomation(std::string modelInfo = "")
    {
        //-------------------------------- �汾��Ϣ --------------------------------
        const InferenceEngine::Version* version = InferenceEngine::GetInferenceEngineVersion();

        LOG("INFO") << "[Inference Engine]" << std::endl;
        LOG("INFO") << "Major:" << version->apiVersion.major << std::endl;
        LOG("INFO") << "Minor:" << version->apiVersion.minor << std::endl;
        LOG("INFO") << "Version:" << version << std::endl;
        LOG("INFO") << "BuildNumber:" << version->buildNumber << std::endl;
        LOG("INFO") << "Description:" << version->description << std::endl;
        std::cout << std::endl;

        InferenceEngine::Core ie;
        //-------------------------------- ����ģ����Ϣ --------------------------------
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

        //-------------------------------- ֧�ֵ�Ӳ���豸 --------------------------------
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
    /// [�ο��õ�]����ͨ�ò����������Ѿ������ help/info/input/output/model/async/show ����
    /// </summary>
    /// <param name="args">������������</param>
    /// <param name="program_name">��������</param>
    /// <param name="default_model">Ĭ�ϵ�ģ�����ò���</param>
    /// <returns></returns>
    static void CreateGeneralCmdLine(cmdline::parser& args, const std::string& program_name, const std::string& default_model)
    {
        args.add("help", 'h', "����˵��");
        args.add("info", 0, "Inference Engine Infomation");

        args.add<std::string>("input", 'i', "����Դ��������ʽ��(video|camera|shared|socket)[:value[:value[:...]]]", false, "cam:0");
        args.add<std::string>("output", 'o', "���Դ��������ʽ��(shared|console|socket)[:value[:value[:...]]]", false, "shared:o_source.bin");// "shared:o_source.bin");
        args.add<std::string>("model", 'm', "���� AIʶ���� �� ����ģ������/�ļ�(.xml)��Ŀ���豸����ʽ��(AIģ������)[:����[:Ӳ��]]��"
            "ʾ����face-detection-adas-0001:FP16:CPU �� face-detection-adas-0001:FP16:HETERO:CPU,GPU", default_model.empty(), default_model);

        args.add<bool>("async", 0, "�Ƿ��첽����ʶ��", false, true);

#ifdef _DEBUG
        args.add<bool>("show", 0, "�Ƿ���ʾ��Ƶ���ڣ����ڵ���", false, true);
#else
        args.add<bool>("show", 0, "�Ƿ���ʾ��Ƶ���ڣ����ڵ���", false, false);
#endif

        args.set_program_name(program_name);
    }
#endif

    /// <summary>
    /// ����̨��������رջ��˳�����WinAPI ���� SetConsoleCtrlHandler ����
    /// </summary>
    /// <param name="CtrlType"></param>
    /// <returns></returns>
    inline BOOL CloseHandlerRoutine(DWORD CtrlType)
    {
        std::cout << "SetConsoleCtrlHandler:::" << CtrlType << std::endl;
        switch (CtrlType)
        {
        case CTRL_C_EVENT:          //���û�������CTRL+C,������GenerateConsoleCtrlEvent API����
        case CTRL_BREAK_EVENT:      //�û�����CTRL+BREAK, ������ GenerateConsoleCtrlEvent API ����
            if (!IsRunning) exit(0);
            else IsRunning = false;
            break;

        case CTRL_CLOSE_EVENT:      //����ͼ�رտ���̨����ϵͳ���͹ر���Ϣ
        case CTRL_LOGOFF_EVENT:     //�û��˳�ʱ�����ǲ��ܾ������ĸ��û�
        case CTRL_SHUTDOWN_EVENT:   //��ϵͳ���ر�ʱ
            exit(0);
            break;
        }

        return TRUE;
    }


    /// <summary>
    /// ��ȡ Window API ����ִ��״̬
    /// </summary>
    /// <param name="message">�����Ϣ�ı�</param>
    /// <returns></returns>
    inline DWORD GetLastErrorFormatMessage(LPVOID& message)
    {
        DWORD id = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, id, 0, (LPTSTR)&message, 0, NULL);

        return id;
    };


    /// <summary>
    /// ����ֻд�� �ڴ湲���ļ���� �� �ڴ�ռ�ӳ����ͼ
    /// </summary>
    /// <param name="handle">�ڴ��ļ����</param>
    /// <param name="buffer">�ڴ�ӳ�䵽��ǰ���������ָ��</param>
    /// <param name="size">������ڴ�ռ��С���ֽ�Ϊ��</param>
    /// <param name="name">�ڴ湲���ļ������ƣ��������̶�дʹ�ø�����</param>
    /// <returns>�����ɹ� ���� true</returns>
    inline bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name)
    {
        ///���������ڴ�
        handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size, (LPCSTR)name);
        LPVOID message = NULL;
        DWORD ErrorID = GetLastErrorFormatMessage(message);     //GetLastError ��� CreateFileMapping ״̬
        LOG("INFO") << "CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

        if (ErrorID != 0 || handle == NULL) return false;

        ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
        buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);  //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
        ErrorID = GetLastErrorFormatMessage(message);                   //GetLastError ��� MapViewOfFile ״̬
        LOG("INFO") << "MapViewOfFile     [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0 || buffer == NULL)
        {
            CloseHandle(handle);
            return false;
        }

        return true;
    }


    /// <summary>
    /// ��ȡֻ���� �ڴ湲���ļ���� �� �ڴ�ռ�ӳ����ͼ
    /// </summary>
    /// <param name="handle">�ڴ��ļ����</param>
    /// <param name="buffer">�ڴ�ӳ�䵽��ǰ���������ָ��</param>
    /// <param name="name">��Ҫ��ȡ���ڴ湲���ļ�������</param>
    /// <returns>��ȡ�ɹ� ���� true</returns>
    inline bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name)
    {
        ///���������ڴ�
        handle = OpenFileMapping(FILE_MAP_READ, FALSE, (LPCSTR)name);   //PAGE_READONLY
        //GetLastError ��� OpenFileMapping ״̬
        LPVOID message = NULL;
        DWORD ErrorID = GetLastErrorFormatMessage(message);
        LOG("INFO") << "OpenFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message: " << (char*)message;

        if (ErrorID != 0 || handle == NULL) return false;

        ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
        buffer = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
        //GetLastError ��� MapViewOfFile ״̬
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
    /// ���� Size ��������ʽ��width(x|,)height
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
    /// ��������ģ�Ͳ���
    /// <para>�����ʽΪ��(AIģ������)[:����[:Ӳ��]]��ʾ����face-detection-adas-0001:FP32:CPU �� face-detection-adas-0001:FP16:GPU </para>
    /// </summary>
    /// <param name="args"></param>
    /// <returns>���ؾ��� ["model","fp","device","path"] ���Ե� std::map ����</returns>
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
    /// ���ַ��ָ������
    /// </summary>
    /// <param name="src">Դʼ�ַ���</param>
    /// <param name="delimiter">�����</param>
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
    /// �Ż����
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
            THROW_IE_EXCEPTION << "���������ͼ���ͨ��������ƥ��";

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
            THROW_IE_EXCEPTION << "ͨ����������֧�� ... ";
        }
    }


    /// <summary>
    /// ͨ���µ�Blobָ���װ�洢�ڴ��ݵ� cv::Mat �����ڲ������ݣ�û�з����ڴ���䡣�� Blob ��ָ���Ѿ����ڵ� cv::Mat ����
    /// </summary>
    /// <param name="mat">��������ͼ�����ݵ� cv::Mat ����</param>
    /// <returns>���� Blob ָ��</returns>
    inline InferenceEngine::Blob::Ptr WrapMat2Blob(const cv::Mat& frame)
    {
        size_t channels = frame.channels();
        size_t height = frame.size().height;
        size_t width = frame.size().width;

        size_t strideH = frame.step.buf[0];
        size_t strideW = frame.step.buf[1];

        bool is_dense = strideW == channels && strideH == channels * width;
        if (!is_dense)
            THROW_IE_EXCEPTION << "��֧�ִӷǳ��� cv::Mat ת��";

        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8,
            { 1, channels, height, width }, InferenceEngine::Layout::NHWC);

        return InferenceEngine::make_shared_blob<uint8_t>(tDesc, frame.data);
    }

    /// <summary>
    /// �������ƱȽϡ� ������������, 0 ~ 1 ���룬����ԽСԽ���ƣ�0��ʾ�н�Ϊ0�㣬1��ʾ�н�Ϊ90��
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
            // ���۲�����Σ��ߴ���Թ̶�˳��洢
            return dims.back();
        }
        else {
            THROW_IE_EXCEPTION << "����û�п�ȳߴ�";
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
            THROW_IE_EXCEPTION << "����û�и߶ȳߴ�";
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
                THROW_IE_EXCEPTION << "����û��ͨ���ߴ�";
            }
        }
        else {
            THROW_IE_EXCEPTION << "����û��ͨ���ߴ�";
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
                THROW_IE_EXCEPTION << "����û�����γߴ�";
            }
        }
        else {
            THROW_IE_EXCEPTION << "����û�����γߴ�";
        }
        return 0;
    }

    /// <summary>
    /// ���� cv::Rect ����
    /// </summary>
    /// <param name="rect"></param>
    /// <param name="scale">�Ŵ����</param>
    /// <param name="dx_offset">�� x ����ƫ��ϵ��</param>
    /// <param name="dy_offset">�� y ����ƫ��ϵ��</param>
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