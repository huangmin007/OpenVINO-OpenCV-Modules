#include "pose_detection.hpp"
#include "static_functions.hpp"
//#include "samples/ocv_common.hpp"
#include <opencv2/opencv.hpp>



PoseDetection::PoseDetection(const std::string& model, const std::string& device, bool isAsync)
    :model(model), device(device), isAsync(isAsync)
{

}
PoseDetection::~PoseDetection()
{
}

void PoseDetection::request()
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

void PoseDetection::wait()
{
    if (!requestPtr || !isAsync) return;

    //RESULT_READY， 一直等待，直到有推理结果出来。
    //STATUS_ONLY， 立即返回请求状态，它不会阻塞或中断当前线程。
    requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

void PoseDetection::enqueue(const cv::Mat& frame)
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
InferenceEngine::CNNNetwork PoseDetection::read(InferenceEngine::Core& ie)
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
    //inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
    //inputsInfo.begin()->second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    //inputsInfo.begin()->second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::BGR);

    network_input_name = inputsInfo.begin()->first;
    network_input_width = inputsInfo.begin()->second->getTensorDesc().getDims()[3];
    network_input_height = inputsInfo.begin()->second->getTensorDesc().getDims()[2];
    std::cout << "Inputs::" << network_input_name
        << "," << network_input_width
        << "," << network_input_height << std::endl;

    //---------- 网络层的输出信息，不同的网络可能有多层输出 ---------
    InferenceEngine::OutputsDataMap outputsInfo(cnnNetwork.getOutputsInfo());
    //outputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::FP32);
    //outputsInfo.begin()->second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(outputsInfo.begin()->second->getDims()));

    network_output_name = outputsInfo.begin()->first;
    network_output_max_count = outputsInfo.begin()->second->getTensorDesc().getDims()[1];
    //network_output_object_size = outputsInfo.begin()->second->getTensorDesc().getDims()[2];

    H = outputsInfo.begin()->second->getTensorDesc().getDims()[2];
    W = outputsInfo.begin()->second->getTensorDesc().getDims()[3];

    std::cout << "Outputs::" << network_output_name
        << ",B:" << outputsInfo.begin()->second->getTensorDesc().getDims()[0]
        << ",C:" << outputsInfo.begin()->second->getTensorDesc().getDims()[1]
        << ",H:" << outputsInfo.begin()->second->getTensorDesc().getDims()[2]
        << ",W:" << outputsInfo.begin()->second->getTensorDesc().getDims()[3] << std::endl;

    execNetwork = ie.LoadNetwork(cnnNetwork, device);
    //execNetwork = ie.LoadNetwork(cnnNetwork, device, { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } });

    return cnnNetwork;
}

bool PoseDetection::getResults(std::vector<PoseDetection::Result>& results, cv::Mat &img)
{
    if (results_fetched) return false;
    results_fetched = true;

    //results.clear();
    float* detections = requestPtr->GetBlob(network_output_name)->buffer().as<float*>();

    //int H = 46, W = 46;
    //int midx = 2,  npairs = 20;
    std::vector<cv::Point> points(network_output_max_count);
    float SX = float(input_frame_width) / W;
    float SY = float(input_frame_height) / H;

    std::cout << "max size::" << network_output_max_count << std::endl;

    std::vector<cv::Mat> heatMaps(44);
    for (int i = 0; i < network_output_max_count; i++)
    {
        //切片对应身体部位的热图.
        cv::Mat heatMap(H, W, CV_32FC1, reinterpret_cast<void*>(const_cast<float*>((detections + i * 46 * 46))));

        double conf;
        cv::Point p(-1, -1), pm;        
        cv::minMaxLoc(heatMap, 0, &conf, 0, &pm);

        if (conf > 0.1) p = pm;
        points[i] = p;
    }

    
    //cv::Mat frame = cv::Mat::zeros(input_frame_width, input_frame_height);

    std::cout << "max size::" << network_output_max_count << std::endl;
    for (int n = 0; n < network_output_max_count; n++)
    {
        // lookup 2 connected body/hand parts
        //cv::Point2f a = points[POSE_PAIRS[midx][n][0]];
        //cv::Point2f b = points[POSE_PAIRS[midx][n][1]];

        cv::Point2f a = points[n];

        // we did not find enough confidence before
        //if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
        //    continue;
        if (a.x <= 0 || a.y <= 0) continue;

        // scale to image size
        a.x *= SX; a.y *= SY;
        //b.x *= SX; b.y *= SY;
        std::cout << "Point" << n << ": " << a.x << "," << a.y << std::endl;


        //cv::line(img, a, b, cv::Scalar(0, 200, 0), 2);
        cv::circle(img, a, 3, cv::Scalar(0, 0, 200), -1);
        //cv::circle(img, b, 3, cv::Scalar(0, 0, 200), -1);
    }
    

    return true;

}