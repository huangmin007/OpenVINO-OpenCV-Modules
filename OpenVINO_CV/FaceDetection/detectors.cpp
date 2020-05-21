#include "detectors.hpp"
#include "OpenVINO_CV.h"
//#include "samples/common.hpp"
#include "samples/ocv_common.hpp"

BaseDetection::BaseDetection(const std::string& topoName,
    const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    bool doRawOutputMessages)
    : topoName(topoName), pathToModel(pathToModel), deviceForInference(deviceForInference),
    maxBatch(maxBatch), isBatchDynamic(isBatchDynamic), isAsync(isAsync),
    enablingChecked(false), _enabled(false), doRawOutputMessages(doRawOutputMessages) {
    if (isAsync) {
        LOG("INFO") << "使用异步模式 " << topoName << std::endl;
    }
}

BaseDetection::~BaseDetection() {}

InferenceEngine::ExecutableNetwork* BaseDetection::operator ->() 
{
    return &net;
}
void BaseDetection::submitRequest() {
    if (!enabled() || request == nullptr) return;
    if (isAsync) {
        request->StartAsync();
    }
    else {
        request->Infer();
    }
}

void BaseDetection::wait() {
    if (!enabled() || !request || !isAsync)
        return;
    request->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

bool BaseDetection::enabled() const 
{
    if (!enablingChecked) 
    {
        _enabled = !pathToModel.empty();
        if (!_enabled) {
            LOG("INFO") << topoName << " 已停用" << std::endl;
        }
        enablingChecked = true;
    }
    return _enabled;
}

void BaseDetection::printPerformanceCounts(std::string fullDeviceName) 
{
    if (!enabled()) {
        return;
    }
    LOG("INFO") << "性能计数器 " << topoName << std::endl << std::endl;
    ::printPerformanceCounts(*request, std::cout, fullDeviceName, false);
}





FaceDetection::FaceDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync,
    double detectionThreshold, bool doRawOutputMessages,
    float bb_enlarge_coefficient, float bb_dx_coefficient, float bb_dy_coefficient)
    : BaseDetection("Face Detection", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
    detectionThreshold(detectionThreshold),
    maxProposalCount(0), objectSize(0), enquedFrames(0), width(0), height(0),
    network_input_width(0), network_input_height(0),
    bb_enlarge_coefficient(bb_enlarge_coefficient), bb_dx_coefficient(bb_dx_coefficient),
    bb_dy_coefficient(bb_dy_coefficient), resultsFetched(false) {}

void FaceDetection::submitRequest() 
{
    if (!enquedFrames) return;
    enquedFrames = 0;
    resultsFetched = false;
    results.clear();
    BaseDetection::submitRequest();
}

void FaceDetection::enqueue(const cv::Mat& frame) 
{
    if (!enabled()) return;

    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    width = static_cast<float>(frame.cols);
    height = static_cast<float>(frame.rows);

    InferenceEngine::Blob::Ptr  inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(frame, inputBlob);

    enquedFrames = 1;
}

InferenceEngine::CNNNetwork FaceDetection::read(const InferenceEngine::Core& ie) {
    LOG("INFO") << "为 人脸检测 加载网络文件" << std::endl;
    /** Read network model **/
    auto network = ie.ReadNetwork(pathToModel);
    /** Set batch size to 1 **/
    LOG("INFO") << "批量大小设置 " << maxBatch << std::endl;
    network.setBatchSize(maxBatch);
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check inputs -------------------------------------------------------------
    LOG("INFO") << "检查人脸检测网络输入 ..." << std::endl;
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("人脸检测网络应该只有一个输入");
    }
    InferenceEngine::InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);

    const InferenceEngine::SizeVector inputDims = inputInfoFirst->getTensorDesc().getDims();
    network_input_height = inputDims[2];
    network_input_width = inputDims[3];

    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------------
    LOG("INFO") << "检查人脸检测网络输出 ... " << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    if (outputInfo.size() == 1) {
        InferenceEngine::DataPtr& _output = outputInfo.begin()->second;
        output = outputInfo.begin()->first;
        const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("人脸检测网络输出层的最后一个维度应为7");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("不兼容的人脸检测网络输出尺寸应为4，但应为 " +
                std::to_string(outputDims.size()));
        }
        _output->setPrecision(InferenceEngine::Precision::FP32);
    }
    else {
        for (const auto& outputLayer : outputInfo) {
            const InferenceEngine::SizeVector outputDims = outputLayer.second->getTensorDesc().getDims();
            if (outputDims.size() == 2 && outputDims.back() == 5) {
                output = outputLayer.first;
                maxProposalCount = outputDims[0];
                objectSize = outputDims.back();
                outputLayer.second->setPrecision(InferenceEngine::Precision::FP32);
            }
            else if (outputDims.size() == 1 && outputLayer.second->getPrecision() == InferenceEngine::Precision::I32) {
                labels_output = outputLayer.first;
            }
        }
        if (output.empty() || labels_output.empty()) {
            //throw std::logic_error("Face Detection network must contain ether single Detection Output or 'boxes' [nx5] and 'labels' [n] at least, where 'n' is a number of detected objects.");
            throw std::logic_error("人脸检测网络必须至少包含以太单个检测输出或 'boxes' [nx5]和 'labels' [n]，其中 'n' 是检测到的对象数。");
        }
    }

    LOG("INFO") << "将人脸检测模型加载到 [" << deviceForInference << "] 硬件" << std::endl;
    input = inputInfo.begin()->first;
    return network;
}

void FaceDetection::fetchResults() {
    if (!enabled()) return;
    results.clear();
    if (resultsFetched) return;
    resultsFetched = true;
    const float* detections = request->GetBlob(output)->buffer().as<float*>();
    const int32_t* labels = !labels_output.empty() ? request->GetBlob(labels_output)->buffer().as<int32_t*>() : nullptr;

    for (int i = 0; i < maxProposalCount && objectSize == 5; i++) {
        Result r;
        r.label = labels[i];
        r.confidence = detections[i * objectSize + 4];

        if (r.confidence <= detectionThreshold && !doRawOutputMessages) {
            continue;
        }

        r.location.x = static_cast<int>(detections[i * objectSize + 0] / network_input_width * width);
        r.location.y = static_cast<int>(detections[i * objectSize + 1] / network_input_height * height);
        r.location.width = static_cast<int>(detections[i * objectSize + 2] / network_input_width * width - r.location.x);
        r.location.height = static_cast<int>(detections[i * objectSize + 3] / network_input_height * height - r.location.y);

        // 扩大正方形并扩大人脸边界框，以使人脸分析网络更可靠地运行
        int bb_width = r.location.width;
        int bb_height = r.location.height;

        int bb_center_x = r.location.x + bb_width / 2;
        int bb_center_y = r.location.y + bb_height / 2;

        int max_of_sizes = std::max(bb_width, bb_height);

        int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
        int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

        r.location.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
        r.location.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

        r.location.width = bb_new_width;
        r.location.height = bb_new_height;

        if (doRawOutputMessages) {
            std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                << r.location.height << ")"
                << ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
        }
        if (r.confidence > detectionThreshold) {
            results.push_back(r);
        }
    }

    for (int i = 0; i < maxProposalCount && objectSize == 7; i++) {
        float image_id = detections[i * objectSize + 0];
        if (image_id < 0) {
            break;
        }
        Result r;
        r.label = static_cast<int>(detections[i * objectSize + 1]);
        r.confidence = detections[i * objectSize + 2];

        if (r.confidence <= detectionThreshold && !doRawOutputMessages) {
            continue;
        }

        r.location.x = static_cast<int>(detections[i * objectSize + 3] * width);
        r.location.y = static_cast<int>(detections[i * objectSize + 4] * height);
        r.location.width = static_cast<int>(detections[i * objectSize + 5] * width - r.location.x);
        r.location.height = static_cast<int>(detections[i * objectSize + 6] * height - r.location.y);

        // 扩大正方形并扩大人脸边界框，以使人脸分析网络更可靠地运行
        int bb_width = r.location.width;
        int bb_height = r.location.height;

        int bb_center_x = r.location.x + bb_width / 2;
        int bb_center_y = r.location.y + bb_height / 2;

        int max_of_sizes = std::max(bb_width, bb_height);

        int bb_new_width = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);
        int bb_new_height = static_cast<int>(bb_enlarge_coefficient * max_of_sizes);

        r.location.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
        r.location.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

        r.location.width = bb_new_width;
        r.location.height = bb_new_height;

        if (doRawOutputMessages) {
            std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                << r.location.height << ")"
                << ((r.confidence > detectionThreshold) ? " WILL BE RENDERED!" : "") << std::endl;
        }
        if (r.confidence > detectionThreshold) {
            results.push_back(r);
        }
    }
}