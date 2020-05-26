#include "person_detection.hpp"
//#include "samples/ocv_common.hpp"


PersonDetection::PersonDetection(const std::string& model, const std::string& device, bool isAsync)
	:model(model),device(device),isAsync(isAsync)
{

}
PersonDetection::~PersonDetection()
{
}

InferenceEngine::ExecutableNetwork* PersonDetection::operator ->()
{
	return &execNetwork;
}

void PersonDetection::request()
{
    if (!enquedFrames) return;  //队列帧不等于空，返回
    if (requestPtr == nullptr) return;

    enquedFrames = 0;
    results_fetched = false;

    if (isAsync)
        requestPtr->StartAsync();
    else 
        requestPtr->Infer();
}

void PersonDetection::wait()
{
    if (!requestPtr || !isAsync) return;

    //RESULT_READY， 一直等待，直到有推理结果出来。
    //STATUS_ONLY， 立即返回请求状态，它不会阻塞或中断当前线程。
    requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) 
{
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];

    if (static_cast<size_t>(orig_image.channels()) != channels) 
        THROW_IE_EXCEPTION << "The number of channels for net input and image must match";
    
    T* blob_data = blob->buffer().as<T*>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
        static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    if (channels == 1) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + h * width + w] = resized_image.at<uchar>(h, w);
            }
        }
    }
    else if (channels == 3) 
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        /*
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    blob_data[batchOffset + c * width * height + h * width + w] = resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t1 - t0).count();
        std::cout << "duration:" << t << std::endl;
        */
        
        for (int h = 0; h < height; h++)
        {
            //cv::Vec3b* row_pixels = resized_image.ptr<cv::Vec3b>(h);
            T* row_pixels = resized_image.data + h * resized_image.step;
            for (int w = 0; w < width; w++)
            {
                //cv::Vec3b bgr = row_pixels[w];
                //blob_data[batchOffset + h * width + w] = row_pixels[w][0];// row_pixels[0];
                //blob_data[batchOffset + (width * height * 1) + h * width + w] = row_pixels[w][1];// row_pixels[1];
                //blob_data[batchOffset + (width * height * 2) + h * width + w] = row_pixels[w][2];// row_pixels[2];

                blob_data[batchOffset + h * width + w] = row_pixels[0];
                blob_data[batchOffset + (width * height * 1) + h * width + w] = row_pixels[1];
                blob_data[batchOffset + (width * height * 2) + h * width + w] = row_pixels[2];

                row_pixels += 3;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t1 - t0).count();
        std::cout << "duration:" << t << std::endl;
        
    }
    else {
        THROW_IE_EXCEPTION << "Unsupported number of channels";
    }
}

void PersonDetection::enqueue(const cv::Mat& frame)
{
    if (!requestPtr)
        requestPtr = execNetwork.CreateInferRequestPtr();    //创建推断请求

    input_frame_width = frame.cols;
    input_frame_height = frame.rows;

    InferenceEngine::Blob::Ptr inputBlobPtr = requestPtr->GetBlob(network_input_name);

    matU8ToBlob<uint8_t>(frame, inputBlobPtr);
    enquedFrames = 1;
}

/// <summary>
/// 读取网络，配置输入/输出
/// </summary>
/// <param name="ie"></param>
/// <returns></returns>
InferenceEngine::CNNNetwork PersonDetection::read(InferenceEngine::Core &ie)
{
    std::cout << "Path:" << model << std::endl;
    //读取 IR xml和bin文件
    cnnNetwork = ie.ReadNetwork(model);
    //batch size
    cnnNetwork.setBatchSize(1);

    //使用模型 person_detection_retail_0013
    //不检查了，，，，
    //不检查模型文件的输入与输出是否吻合了，，，
    //网络信息格式 <LayoutName, LayoutInfo>

    //---------- 网络层的输入信息 ---------
    //  [BxCxHxW] name:"input",shape: [1x3x320x544] 
    InferenceEngine::InputsDataMap inputsInfo(cnnNetwork.getInputsInfo());
    //更改用户提供的输入数据的精度为 U8
    inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
    inputsInfo.begin()->second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    inputsInfo.begin()->second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::BGR);

    network_input_name = inputsInfo.begin()->first;
    network_input_width = inputsInfo.begin()->second->getTensorDesc().getDims()[3];
    network_input_height = inputsInfo.begin()->second->getTensorDesc().getDims()[2];
    std::cout << "Inputs::" << network_input_name 
        << "," << network_input_width 
        << "," << network_input_height << std::endl;

    //---------- 网络层的输出信息，不同的网络可能有多层输出 ---------
    //[1, 1, N, 7]
    InferenceEngine::OutputsDataMap outputsInfo(cnnNetwork.getOutputsInfo());
    outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
    outputsInfo.begin()->second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(outputsInfo.begin()->second->getDims()));

    network_output_name = outputsInfo.begin()->first;
    network_output_max_count = outputsInfo.begin()->second->getTensorDesc().getDims()[2];
    network_output_object_size = outputsInfo.begin()->second->getTensorDesc().getDims()[3];

    std::cout << "Outputs::" << network_output_name
        << ",MaxCout:" << network_output_max_count
        << ", objectSize:" << network_output_object_size << std::endl;

    execNetwork = ie.LoadNetwork(cnnNetwork, device);
    //execNetwork = ie.LoadNetwork(cnnNetwork, device, { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } });

    return cnnNetwork;
}

bool PersonDetection::getResults(std::vector<PersonDetection::Result> &results)
{
    if (results_fetched) return false;
    results_fetched = true;

    results.clear();
    const float* detections = requestPtr->GetBlob(network_output_name)->buffer().as<float*>();
    
    for (int i = 0; i < network_output_max_count && network_output_object_size == 7; i++)
    {
        float image_id = detections[i * network_output_object_size + 0];
        if (image_id < 0) break;

        Result rt;
        rt.label = detections[i * network_output_object_size + 1];
        rt.confidence = std::min(std::max(0.0f, detections[i * network_output_object_size + 2]), 1.0f);

        if (rt.confidence <= 0.5) continue;

        rt.location.x = static_cast<int>(detections[i * network_output_object_size + 3] * input_frame_width);
        rt.location.y = static_cast<int>(detections[i * network_output_object_size + 4] * input_frame_height);
        rt.location.width = static_cast<int>(detections[i * network_output_object_size + 5] * input_frame_width - rt.location.x);
        rt.location.height = static_cast<int>(detections[i * network_output_object_size + 6] * input_frame_height - rt.location.y);

        results.push_back(rt);
    }

    return true;

}