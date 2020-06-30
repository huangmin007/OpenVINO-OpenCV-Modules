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

//using namespace std;


/// <summary>
/// OpenCV cv::waitKey Ĭ�ϲο���ʱ
/// </summary>
static const int WaitKeyDelay = 33;

/// <summary>
/// ����̨�߳�����״̬������� SetConsoleCtrlHandler �ı�
/// </summary>
static bool IsRunning = true;

namespace space
{
    //�����ڴ汣���ֽ���
    #define SHARED_RESERVE_BYTE_SIZE 32

    //��־��ʽ
    #define LOG(type)       (std::cout << "[" << std::setw(5) << std::right << type << "] ")

    //ms
    using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
    using hc = std::chrono::high_resolution_clock;

    //����㹲�����
    using OutputSharedLayer = std::tuple<std::string, HANDLE, LPVOID>;
    //����㹲�����
    using OutputSharedLayers = std::vector<std::tuple<std::string, HANDLE, LPVOID>>;

    //�ƶ���ɻص�����
    using TCompletionCallback = InferenceEngine::IInferRequest::CompletionCallback;
    //�ƶ���ɻص�����
    using FCompletionCallback = std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>;
    //�ƶ���ɻص�����
    using FPCompletionCallback = std::function<void(InferenceEngine::IInferRequest::Ptr, InferenceEngine::StatusCode)>;


#if 0
	/// <summary>
	/// ����̨��������رջ��˳�����WinAPI ���� SetConsoleCtrlHandler ����
	/// </summary>
	/// <param name="CtrlType"></param>
	/// <returns></returns>
	BOOL CloseHandlerRoutine(DWORD CtrlType);
    /// <summary>
    /// ��ȡ Window API ����ִ��״̬
    /// </summary>
    /// <param name="message">�����Ϣ�ı�</param>
    /// <returns></returns>
	DWORD GetLastErrorFormatMessage(LPVOID& message);


    
    /// <summary>
    /// ��ȡֻ���� �ڴ湲���ļ���� �� �ڴ�ռ�ӳ����ͼ
    /// </summary>
    /// <param name="handle">�ڴ��ļ����</param>
    /// <param name="buffer">�ڴ�ӳ�䵽��ǰ���������ָ��</param>
    /// <param name="name">��Ҫ��ȡ���ڴ湲���ļ�������</param>
    /// <returns>��ȡ�ɹ� ���� true</returns>
	bool GetOnlyReadMapFile(HANDLE& handle, LPVOID& buffer, const char* name);
    /// <summary>
    /// ����ֻд�� �ڴ湲���ļ���� �� �ڴ�ռ�ӳ����ͼ
    /// </summary>
    /// <param name="handle">�ڴ��ļ����</param>
    /// <param name="buffer">�ڴ�ӳ�䵽��ǰ���������ָ��</param>
    /// <param name="size">������ڴ�ռ��С���ֽ�Ϊ��</param>
    /// <param name="name">�ڴ湲���ļ������ƣ��������̶�дʹ�ø�����</param>
    /// <returns>�����ɹ� ���� true</returns>
	bool CreateOnlyWriteMapFile(HANDLE& handle, LPVOID& buffer, uint32_t size, const char* name);
    /// <summary>
    /// �������� Blob ���ڴ�
    /// </summary>
    /// <param name="request">�ƶ��������</param>
    /// <param name="name">name of blob </param>
    /// <param name="reserve_byte_size">�ڹ�������ǰ�����ֽ���</param>
    /// <returns></returns>
    const OutputSharedLayer CreateSharedBlob(InferenceEngine::InferRequest::Ptr& request, const std::string name, size_t reserve_byte_size = SHARED_RESERVE_BYTE_SIZE);
    /// <summary>
    /// ������ CreateSharedBlob �������ڴ湲�����
    /// </summary>
    /// <param name="shared_layers"></param>
    void DisposeSharedMapping(OutputSharedLayers& shared_layers);



    /// <summary>
    /// ���ַ��ָ������
    /// </summary>
    /// <param name="src">Դʼ�ַ���</param>
    /// <param name="delimiter">�����</param>
    /// <returns></returns>
    const std::vector<std::string> SplitString(const std::string& src, const char delimiter = ':');
	/// <summary>
	/// ���� Size ��������ʽ��width(x)height �� width(,)height 
	/// </summary>
	/// <param name="size"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <returns></returns>
	bool ParseArgForSize(const std::string& size, int& width, int& height);
    /// <summary>
    /// ��������ģ�Ͳ���
    /// <para>�����ʽΪ��(AIģ������)[:����[:Ӳ��]]��ʾ����face-detection-adas-0001:FP32:CPU �� face-detection-adas-0001:FP16:GPU </para>
    /// </summary>
    /// <param name="args"></param>
    /// <returns>���ؾ��� ["model","fp","device","path"] ���Ե� std::map ����</returns>
	const std::map<std::string, std::string> ParseArgsForModel(const std::string& args);
    /// <summary>
    /// ��ȡ��ǩ�ļ�
    /// </summary>
    /// <param name="labels_file"></param>
    /// <returns></returns>
    const std::vector<std::string> ReadLabels(const std::string& labels_file);


    /// <summary>
    /// �Ż���ģ�Mat U8 to Blob
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="orig_image"></param>
    /// <param name="blob"></param>
    /// <param name="batchIndex"></param>
    template <typename T>
    void MatU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);
    /// <summary>
    /// ͨ���µ�Blobָ���װ�洢�ڴ��ݵ� cv::Mat �����ڲ������ݣ�û�з����ڴ���䡣�� Blob ��ָ���Ѿ����ڵ� cv::Mat ����
    /// </summary>
    /// <param name="mat">��������ͼ�����ݵ� cv::Mat ����</param>
    /// <returns>���� Blob ָ��</returns>
    InferenceEngine::Blob::Ptr WrapMat2Blob(const cv::Mat& frame);
    /// <summary>
    /// ͨ���µ�Blobָ���װ�洢�ڴ��ݵ� cv::Mat �����ڲ������ݣ�û�з����ڴ���䡣�� Blob ��ָ���Ѿ����ڵ� cv::Mat ����
    /// </summary>
    /// <param name="request"></param>
    /// <param name="input_name"></param>
    /// <param name="frame"></param>
    void MrapMat2Blob(InferenceEngine::InferRequest::Ptr request, const std::string& input_name, const cv::Mat& frame);
	/// <summary>
	/// �������ƱȽϡ� ������������, 0 ~ 1 ���룬����ԽСԽ���ƣ�0��ʾ�н�Ϊ0�㣬1��ʾ�н�Ϊ90��
	/// </summary>
	/// <param name="fv1"></param>
	/// <param name="fv2"></param>
	/// <returns></returns>
	float CosineDistance(const std::vector<float>& fv1, const std::vector<float>& fv2);
    /// <summary>
    /// ���� cv::Rect ����
    /// </summary>
    /// <param name="rect"></param>
    /// <param name="scale">�Ŵ����</param>
    /// <param name="dx_offset">�� x ����ƫ��ϵ��</param>
    /// <param name="dy_offset">�� y ����ƫ��ϵ��</param>
    /// <returns></returns>
    cv::Rect ScaleRectangle(const cv::Rect& rect, float scale = 1.2, float dx_offset = 1.0, float dy_offset = 1.0);
    /// <summary>
    /// ���Ƽ�����ı߽�
    /// </summary>
    /// <param name="frame"></param>
    /// <param name="rect"></param>
    void DrawObjectBound(cv::Mat frame, const cv::Rect& src_rect);
    
    /// <summary>
    /// ��ȡ��������Ӧ�����������ֽڴ�С
    /// </summary>
    /// <param name="precision"></param>
    /// <returns></returns>
    size_t GetPrecisionOfSize(const InferenceEngine::Precision& precision);


    /// <summary>
    /// ��� InferenceEngine ����汾�����ɼ����豸�б�
    /// </summary>
    void InferenceEngineInformation(std::string modelInfo = "");
    /// <summary>
    /// ���������Ϣ
    /// </summary>
    /// <param name="value"></param>
    void PrintParameterValue(const InferenceEngine::Parameter& value);
#endif

    template <typename T>
    inline std::ostream& operator << (std::ostream& stream, const std::vector<T>& v)
    {
        stream << "[";
        for (int i = 0; i < v.size(); i++)
            stream << v[i] << (i != v.size() - 1 ? "," : "]");

        return stream;
    }    
    inline std::ostream& operator << (std::ostream& stream, const InferenceEngine::InputsDataMap& inputsInfo)
    {
        for (const auto& input : inputsInfo)
        {
            stream << "Input Name:[" << input.first << "] " <<
                "Shape:" << input.second->getTensorDesc().getDims() << " " <<
                "Layout:[" << input.second->getTensorDesc().getLayout() << "] " <<
                "Precision:[" << input.second->getPrecision() << "] " <<
                "ColorFormat:[" << input.second->getPreProcess().getColorFormat() << "] " <<
                "ResizeAlgorithm:[" << input.second->getPreProcess().getResizeAlgorithm() << "] " <<
                "\r\n";
        }

        return stream;
    }    
    inline std::ostream& operator << (std::ostream& stream, const InferenceEngine::OutputsDataMap& outputsInfo)
    {
        for (const auto& output : outputsInfo)
        {
            stream << "Output Name:[" << output.first << "] " <<
                "Shape:" << output.second->getTensorDesc().getDims() << " " <<
                "Layout:[" << output.second->getTensorDesc().getLayout() << "] " <<
                "Precision:[" << output.second->getPrecision() << "] " <<
                "\r\n";
        }

        return stream;
    }
    
    /// <summary>
    /// ����̨��������رջ��˳�����WinAPI ���� SetConsoleCtrlHandler ����
    /// </summary>
    /// <param name="CtrlType"></param>
    /// <returns></returns>
    static BOOL CloseHandlerRoutine(DWORD CtrlType)
    {
        switch (CtrlType)
        {
        case CTRL_C_EVENT:          //���û�������CTRL+C,������GenerateConsoleCtrlEvent API����
        case CTRL_BREAK_EVENT:      //�û�����CTRL+BREAK, ������ GenerateConsoleCtrlEvent API ����
            if (!IsRunning)
                exit(0);
            else 
                IsRunning = false;
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


#pragma region �����ڴ���غ���
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

        if (ErrorID != 0) return false;

        ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
        buffer = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, 0);
        //GetLastError ��� MapViewOfFile ״̬
        ErrorID = GetLastErrorFormatMessage(message);
        LOG("INFO") << "MapViewOfFile   [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0) return false;

        return true;
    }

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
        LOG("INFO") << "CreateFileMapping [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0) return false;

        ///ӳ�䵽��ǰ���̵ĵ�ַ�ռ���ͼ
        buffer = MapViewOfFile(handle, FILE_WRITE_ACCESS, 0, 0, size);  //FILE_WRITE_ACCESS,FILE_MAP_ALL_ACCESS,,FILE_MAP_WRITE
        ErrorID = GetLastErrorFormatMessage(message);                   //GetLastError ��� MapViewOfFile ״̬
        LOG("INFO") << "MapViewOfFile     [" << name << "]  GetLastError:" << ErrorID << "  Message:" << (char*)message;

        if (ErrorID != 0) return false;

        return true;
    }

    /// <summary>
    /// �������� Blob �㣬��ָ���� Blob ����Ϊ����㣨���������𣿣���
    /// </summary>
    /// <param name="request">�ƶ��������</param>
    /// <param name="name">name of blob </param>
    /// <param name="reserve_byte_size">�ڹ�������ǰ�����ֽ���</param>
    /// <returns></returns>
    static const OutputSharedLayers CreateSharedBlobs(
        const std::vector<InferenceEngine::InferRequest::Ptr>& requestPtrs, 
        const std::vector<std::string> names, 
        size_t reserve_byte_size = SHARED_RESERVE_BYTE_SIZE)
    {
        if (requestPtrs.size() == 0 || names.size() == 0 || names[0].empty())  throw std::invalid_argument("��������Ϊ��");
        
        OutputSharedLayers shared_layers;

        for (const auto& name : names)
        {
            InferenceEngine::Blob::Ptr blob = requestPtrs[0]->GetBlob(name);
            if (blob == nullptr) throw std::invalid_argument("δ��ȡ�� Blob ����");

            size_t buffer_size = blob->byteSize() + reserve_byte_size;
            InferenceEngine::TensorDesc tensor = blob->getTensorDesc();
            const InferenceEngine::Precision precision = tensor.getPrecision();

            LPVOID buffer; HANDLE handle;
            if (CreateOnlyWriteMapFile(handle, buffer, buffer_size, name.c_str()))
            {
                for (auto& request : requestPtrs)
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
                        LOG("WARN") << "����ӳ�����ݣ�δ�����������ȣ�" << precision << std::endl;
                        throw std::invalid_argument("����ӳ�����ݣ�δ�����������ȣ�" + precision);
                    }
                }

                shared_layers.push_back({ name, handle, buffer });
            }
            else
            {
                LOG("ERROR") << "�����ڴ�� " << name << " ����ʧ�� ... " << std::endl;
                throw std::logic_error("�����ڴ�鴴��ʧ�� ... ");
            }
        }

        return shared_layers;
    }

    /// <summary>
    /// ������ CreateSharedBlobs �������ڴ湲�����
    /// </summary>
    /// <param name="shared_layers"></param>
    static void DisposeSharedMapping(OutputSharedLayers& shared_layers)
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
#pragma endregion


#pragma region �ַ��ָ��������
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
    /// ���� Size ��������ʽ��width(x)height �� width(,)height 
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
            {"path", ""},           //ģ���ļ�·��
            {"model", ""},          //ģ������
            {"fp", "FP32"},         //ģ���ļ�����
            {"device", "CPU"},      //ģ�ͼ��ص�ʹ�õ�Ӳ��
            {"labelpath", ""},      //ģ�ͱ�ǩ�ļ�·��
            {"full", ""}            //ԭʼ����
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

        argMap["label"] = "models\\" + argMap["model"] + "\\labels.txt";
        argMap["path"] = "models\\" + argMap["model"] + "\\" + argMap["fp"] + "\\" + argMap["model"] + ".xml";

        return argMap;
    }

    /// <summary>
    /// ��ȡ��ǩ�ļ�
    /// </summary>
    /// <param name="labels_file"></param>
    /// <returns></returns>
    inline const std::vector<std::string> ReadLabels(const std::string& labels_file)
    {
        std::vector<std::string> labels;
        std::fstream file(labels_file, std::fstream::in);

        if (!file.is_open())
        {
            LOG("WARN") << "��ǩ�ļ������� " << labels_file << " �ļ� ...." << std::endl;
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
#pragma endregion


#pragma region Mat to Blob
    /// <summary>
    /// �Ż���ģ�Mat U8 to Blob
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="orig_image"></param>
    /// <param name="blob"></param>
    /// <param name="batchIndex"></param>
    template <typename T>
    static inline void MatU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0)
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
    static inline InferenceEngine::Blob::Ptr WrapMat2Blob(const cv::Mat& frame)
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
    /// 
    /// </summary>
    /// <param name="request"></param>
    /// <param name="input_name"></param>
    /// <param name="frame"></param>
    static inline void WrapMat2Blob(InferenceEngine::InferRequest::Ptr request, const std::string& input_name, const cv::Mat& frame)
    {
        size_t width = frame.cols;
        size_t height = frame.rows;
        size_t channels = frame.channels();

        size_t strideH = frame.step.buf[0];
        size_t strideW = frame.step.buf[1];

        bool is_dense = strideW == channels && strideH == channels * width;
        if (!is_dense) throw std::logic_error("�����ͼ��֡��֧��ת�� ... ");

        InferenceEngine::Blob::Ptr inputBlob = request->GetBlob(input_name);

        //������������� Blob��������ͼ���ڴ�����ָ�빲�������㣬��ʵʱ���첽�ƶ�
        //����Ϊͼ��ԭʼ�ߴ磬��ʵ�ǻ��������ģ����Դͼ�ߴ�ܴ��Ǵ���ʱ������
        InferenceEngine::TensorDesc tensor = inputBlob->getTensorDesc();
        InferenceEngine::TensorDesc n_tensor(tensor.getPrecision(), { 1, channels, height, width }, tensor.getLayout());
        request->SetBlob(input_name, InferenceEngine::make_shared_blob<uint8_t>(n_tensor, frame.data));
    }
#pragma endregion


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

    /// <summary>
    /// ���� cv::Rect ����
    /// </summary>
    /// <param name="rect"></param>
    /// <param name="scale">�Ŵ����</param>
    /// <param name="dx_offset">�� x ����ƫ��ϵ��</param>
    /// <param name="dy_offset">�� y ����ƫ��ϵ��</param>
    /// <returns></returns>
    static inline cv::Rect ScaleRectangle(const cv::Rect& rect, float scale = 1.2, float dx_offset = 1.0, float dy_offset = 1.0)
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
    /// ��ȡ��������Ӧ�����������ֽڴ�С
    /// </summary>
    /// <param name="precision"></param>
    /// <returns></returns>
    static size_t GetPrecisionOfSize(const InferenceEngine::Precision& precision)
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
    /// ���Ƽ�����ı߽�
    /// </summary>
    /// <param name="frame"></param>
    /// <param name="rect"></param>
    static inline void DrawObjectBound(cv::Mat frame, const cv::Rect& src_rect)
    {
        auto rect = src_rect & cv::Rect(0, 0, frame.cols, frame.rows);

        float scale = 0.2f;
        int shift = 0;
        int thickness = 2;
        int lineType = cv::LINE_AA;
        cv::Scalar border_color(0, 255, 0);

        cv::rectangle(frame, rect, border_color, 1);
        cv::Point h_width = cv::Point(rect.width * scale, 0);   //ˮƽ���
        cv::Point v_height = cv::Point(0, rect.height * scale); //��ֱ�߶�

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



#pragma region Print Or Debug Information
    /// <summary>
    /// ��� InferenceEngine ����汾�����ɼ����豸�б�
    /// </summary>
    static inline void InferenceEngineInformation(const std::string& modelInfo)
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
                LOG("INFO") << "\tInput Name:[" << input.first << "] " <<
                    "Shape:" << input.second->getTensorDesc().getDims() << " " <<
                    "Layout:[" << input.second->getTensorDesc().getLayout() << "] " <<
                    "Precision:[" << input.second->getPrecision() << "] " <<
                    "ColorFormat:[" << input.second->getPreProcess().getColorFormat() << "] " <<
                    "ResizeAlgorithm:[" << input.second->getPreProcess().getResizeAlgorithm() << "] " <<
                    std::endl;
            }

            //------------Output-----------
            LOG("INFO") << "Output Layer" << std::endl;
            InferenceEngine::OutputsDataMap outputsInfo = cnnNetwork.getOutputsInfo();
            for (const auto& output : outputsInfo)
            {
                LOG("INFO") << "\tOutput Name:[" << output.first << "] " <<
                    "Shape:" << output.second->getTensorDesc().getDims() << " " <<
                    "Layout:[" << output.second->getTensorDesc().getLayout() << "] " <<
                    "Precision:[" << output.second->getPrecision() << "] " <<
                    std::endl;
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

    /// <summary>
    /// ���������Ϣ
    /// </summary>
    /// <param name="value"></param>
    static inline void PrintParameterValue(const InferenceEngine::Parameter& value)
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
#pragma endregion

}