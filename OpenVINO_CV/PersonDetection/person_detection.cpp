#include "person_detection.hpp"
#include "samples/ocv_common.hpp"


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
    if (!enquedFrames) return;  //����֡�����ڿգ�����
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

    //RESULT_READY�� һֱ�ȴ���ֱ����������������
    //STATUS_ONLY�� ������������״̬���������������жϵ�ǰ�̡߳�
    requestPtr->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

void PersonDetection::enqueue(const cv::Mat& frame)
{
    if (!requestPtr)
        requestPtr = execNetwork.CreateInferRequestPtr();    //�����ƶ�����

    input_frame_width = frame.cols;
    input_frame_height = frame.rows;

    InferenceEngine::Blob::Ptr inputBlobPtr = requestPtr->GetBlob(network_input_name);

    matU8ToBlob<uint8_t>(frame, inputBlobPtr);
    enquedFrames = 1;
}

/// <summary>
/// ��ȡ���磬��������/���
/// </summary>
/// <param name="ie"></param>
/// <returns></returns>
InferenceEngine::CNNNetwork PersonDetection::read(InferenceEngine::Core &ie)
{
    std::cout << "Path:" << model << std::endl;
    //��ȡ IR xml��bin�ļ�
    cnnNetwork = ie.ReadNetwork(model);
    //batch size
    cnnNetwork.setBatchSize(1);

    //ʹ��ģ�� person_detection_retail_0013
    //������ˣ�������
    //�����ģ���ļ�������������Ƿ��Ǻ��ˣ�����
    //������Ϣ��ʽ <LayoutName, LayoutInfo>

    //---------- ������������Ϣ ---------
    //  [BxCxHxW] name:"input",shape: [1x3x320x544] 
    InferenceEngine::InputsDataMap inputsInfo(cnnNetwork.getInputsInfo());
    //�����û��ṩ���������ݵľ���Ϊ U8
    inputsInfo.begin()->second->setPrecision(InferenceEngine::Precision::U8);
    inputsInfo.begin()->second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    inputsInfo.begin()->second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::BGR);

    network_input_name = inputsInfo.begin()->first;
    network_input_width = inputsInfo.begin()->second->getTensorDesc().getDims()[3];
    network_input_height = inputsInfo.begin()->second->getTensorDesc().getDims()[2];
    std::cout << "Inputs::" << network_input_name 
        << "," << network_input_width 
        << "," << network_input_height << std::endl;

    //---------- �����������Ϣ����ͬ����������ж����� ---------
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