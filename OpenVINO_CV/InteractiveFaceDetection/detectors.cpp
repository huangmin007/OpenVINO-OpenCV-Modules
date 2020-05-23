#include "detectors.hpp"
#include "static_functions.hpp"
#include "samples/ocv_common.hpp"

#include <ie_iextension.h>

// -------------------------- BaseDetection ----------------------------------
#pragma region BaseDetection
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
#pragma endregion


// -------------------------- FaceDetection ----------------------------------
#pragma region FaceDetection
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
    LOG("INFO") << "加载 人脸检测网络 文件 ..." << std::endl;
    /** Read network model **/
    auto network = ie.ReadNetwork(pathToModel);
    /** Set batch size to 1 **/
    LOG("INFO") << "批量大小设置 " << maxBatch << std::endl;
    network.setBatchSize(maxBatch);
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check inputs -------------------------------------------------------------
    LOG("INFO") << "检查 人脸检测网络 输入 ..." << std::endl;
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
    LOG("INFO") << "检查 人脸检测网络 输出 ... " << std::endl;
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
            throw std::logic_error("人脸检测网络必须至少包含以太单个检测输出或 'boxes' [nx5]和 'labels' [n]，其中 'n' 是检测到的对象数。");
        }
    }

    LOG("INFO") << "将 人脸检测模型 加载到 [" << deviceForInference << "]" << std::endl;
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
#pragma endregion


// -------------------------- AgeGenderDetection ----------------------------------
#pragma region AgeGenderDetection
AgeGenderDetection::AgeGenderDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
    : BaseDetection("Age/Gender", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
    enquedFaces(0) {
}

void AgeGenderDetection::submitRequest() {
    if (!enquedFaces)
        return;
    if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
    }
    BaseDetection::submitRequest();
    enquedFaces = 0;
}

void AgeGenderDetection::enqueue(const cv::Mat& face) {
    if (!enabled()) {
        return;
    }
    if (enquedFaces == maxBatch) {
        LOG("WARN") << "检测到的面部数量超过最大值 (" << maxBatch << ") 年龄/性别识别网络处理" << std::endl;
        return;
    }
    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    InferenceEngine::Blob::Ptr  inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

    enquedFaces++;
}

AgeGenderDetection::Result AgeGenderDetection::operator[] (int idx) const {
    InferenceEngine::Blob::Ptr  genderBlob = request->GetBlob(outputGender);
    InferenceEngine::Blob::Ptr  ageBlob = request->GetBlob(outputAge);

    AgeGenderDetection::Result r = { ageBlob->buffer().as<float*>()[idx] * 100,
                                         genderBlob->buffer().as<float*>()[idx * 2 + 1] };
    if (doRawOutputMessages) {
        std::cout << "[" << idx << "] element, male prob = " << r.maleProb << ", age = " << r.age << std::endl;
    }

    return r;
}

InferenceEngine::CNNNetwork AgeGenderDetection::read(const InferenceEngine::Core& ie) {
    LOG("INFO") << "加载 年龄/性别识别网络 文件 ... " << std::endl;
    // Read network
    auto network = ie.ReadNetwork(pathToModel);
    // Set maximum batch size to be used.
    network.setBatchSize(maxBatch);
    LOG("INFO") << "批量大小设置为 " << network.getBatchSize() << std::endl;

    // ---------------------------Check inputs -------------------------------------------------------------
    // Age/Gender Recognition network should have one input and two outputs
    LOG("INFO") << "检查 年龄/性别识别网络 输入 ... " << std::endl;
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("年龄/性别识别网络应该只有一个输入");
    }
    InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------------
    LOG("INFO") << "检查 年龄/性别识别网络 输出 ... " << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    if (outputInfo.size() != 2) {
        throw std::logic_error("年龄/性别识别网络应具有两个输出层");
    }
    auto it = outputInfo.begin();

    InferenceEngine::DataPtr ptrAgeOutput = (it++)->second;
    InferenceEngine::DataPtr ptrGenderOutput = (it++)->second;

    outputAge = ptrAgeOutput->getName();
    outputGender = ptrGenderOutput->getName();

    LOG("INFO") << "将 年龄/性别识别模型 加载到 [" << deviceForInference << "]" << std::endl;
    _enabled = true;
    return network;
}
#pragma endregion


// -------------------------- HeadPoseDetection ----------------------------------
#pragma region HeadPoseDetection 
HeadPoseDetection::HeadPoseDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
    : BaseDetection("Head Pose", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
    outputAngleR("angle_r_fc"), outputAngleP("angle_p_fc"), outputAngleY("angle_y_fc"), enquedFaces(0) {
}

void HeadPoseDetection::submitRequest() {
    if (!enquedFaces) return;
    if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
    }
    BaseDetection::submitRequest();
    enquedFaces = 0;
}

void HeadPoseDetection::enqueue(const cv::Mat& face) {
    if (!enabled()) {
        return;
    }
    if (enquedFaces == maxBatch) {
        LOG("WARN") << "检测到的面部数量超过最大值 (" << maxBatch << ") 头部姿态检测" << std::endl;
        return;
    }
    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    InferenceEngine::Blob::Ptr inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

    enquedFaces++;
}

HeadPoseDetection::Results HeadPoseDetection::operator[] (int idx) const {
    InferenceEngine::Blob::Ptr  angleR = request->GetBlob(outputAngleR);
    InferenceEngine::Blob::Ptr  angleP = request->GetBlob(outputAngleP);
    InferenceEngine::Blob::Ptr  angleY = request->GetBlob(outputAngleY);

    HeadPoseDetection::Results r = { angleR->buffer().as<float*>()[idx],
                                    angleP->buffer().as<float*>()[idx],
                                    angleY->buffer().as<float*>()[idx] };

    if (doRawOutputMessages) {
        std::cout << "[" << idx << "] element, yaw = " << r.angle_y <<
            ", pitch = " << r.angle_p <<
            ", roll = " << r.angle_r << std::endl;
    }

    return r;
}

InferenceEngine::CNNNetwork HeadPoseDetection::read(const InferenceEngine::Core& ie) {
    LOG("INFO") << "加载 头姿势态评估网络 文件 ... " << std::endl;
    // Read network model
    auto network = ie.ReadNetwork(pathToModel);
    // Set maximum batch size
    network.setBatchSize(maxBatch);
    LOG("INFO") << "批量大小设置为 " << network.getBatchSize() << std::endl;

    // ---------------------------Check inputs -------------------------------------------------------------
    LOG("INFO") << "检查 头姿势态评估网络 输入 ... " << std::endl;
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("头部姿势评估网络应该只有一个输入");
    }
    InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------------
    LOG("INFO") << "检查 头部姿态评估网络 输出" << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
    }
    for (const std::string& outName : { outputAngleR, outputAngleP, outputAngleY }) {
        if (outputInfo.find(outName) == outputInfo.end()) {
            throw std::logic_error("没有 " + outName + " 头姿势评估网络中的输出");
        }
    }

    LOG("INFO") << "将 头部姿态评估模型 加载到 [" << deviceForInference << "]" << std::endl;

    _enabled = true;
    return network;
}
#pragma endregion


// -------------------------- EmotionsDetection ----------------------------------
#pragma region EmotionsDetection
EmotionsDetection::EmotionsDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
    : BaseDetection("Emotions Recognition", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
    enquedFaces(0) {
}

void EmotionsDetection::submitRequest() {
    if (!enquedFaces) return;
    if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
    }
    BaseDetection::submitRequest();
    enquedFaces = 0;
}

void EmotionsDetection::enqueue(const cv::Mat& face) {
    if (!enabled()) {
        return;
    }
    if (enquedFaces == maxBatch) {
        LOG("WARN") << "检测到的面部数量超过最大值 (" << maxBatch << ") 情感识别网络处理" << std::endl;
        return;
    }
    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    InferenceEngine::Blob::Ptr inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

    enquedFaces++;
}

std::map<std::string, float> EmotionsDetection::operator[] (int idx) const {
    auto emotionsVecSize = emotionsVec.size();

    InferenceEngine::Blob::Ptr emotionsBlob = request->GetBlob(outputEmotions);

    /* emotions vector must have the same size as number of channels
     * in model output. Default output format is NCHW, so index 1 is checked */
    size_t numOfChannels = emotionsBlob->getTensorDesc().getDims().at(1);
    if (numOfChannels != emotionsVecSize) {
        throw std::logic_error("输出尺寸 (" + std::to_string(numOfChannels) +
            ") 识别网络的特征不等于所使用的情感向量大小  (" +
            std::to_string(emotionsVec.size()) + ")");
    }

    auto emotionsValues = emotionsBlob->buffer().as<float*>();
    auto outputIdxPos = emotionsValues + idx * emotionsVecSize;
    std::map<std::string, float> emotions;

    if (doRawOutputMessages) {
        std::cout << "[" << idx << "] element, predicted emotions (name = prob):" << std::endl;
    }

    for (size_t i = 0; i < emotionsVecSize; i++) {
        emotions[emotionsVec[i]] = outputIdxPos[i];

        if (doRawOutputMessages) {
            std::cout << emotionsVec[i] << " = " << outputIdxPos[i];
            if (emotionsVecSize - 1 != i) {
                std::cout << ", ";
            }
            else {
                std::cout << std::endl;
            }
        }
    }

    return emotions;
}

InferenceEngine::CNNNetwork EmotionsDetection::read(const InferenceEngine::Core& ie) {
    LOG("INFO") << "加载 情绪识别网络 文件 ... " << std::endl;
    // Read network model
    auto network = ie.ReadNetwork(pathToModel);
    // Set maximum batch size
    network.setBatchSize(maxBatch);
    LOG("INFO") << "批量大小设置为 " << network.getBatchSize() << " 情绪识别" << std::endl;
    // -----------------------------------------------------------------------------------------------------

    // Emotions Recognition network should have one input and one output.
    // ---------------------------Check inputs -------------------------------------------------------------
    LOG("INFO") << "检查 情绪识别网络 输入 ... " << std::endl;
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("情绪识别网络应该只有一个输入");
    }
    auto& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------------
    LOG("INFO") << "检查 情绪识别网络 输出" << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("情绪识别网络应具有一个输出层");
    }
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
    }

    outputEmotions = outputInfo.begin()->first;

    LOG("INFO") << "将 情绪识别模型 加载到 [" << deviceForInference << "]" << std::endl;
    _enabled = true;
    return network;
}
#pragma endregion


// -------------------------- FacialLandmarksDetection ----------------------------------
#pragma region FacialLandmarksDetection
FacialLandmarksDetection::FacialLandmarksDetection(const std::string& pathToModel,
    const std::string& deviceForInference,
    int maxBatch, bool isBatchDynamic, bool isAsync, bool doRawOutputMessages)
    : BaseDetection("Facial Landmarks", pathToModel, deviceForInference, maxBatch, isBatchDynamic, isAsync, doRawOutputMessages),
    outputFacialLandmarksBlobName("align_fc3"), enquedFaces(0) {
}

void FacialLandmarksDetection::submitRequest() {
    if (!enquedFaces) return;
    if (isBatchDynamic) {
        request->SetBatch(enquedFaces);
    }
    BaseDetection::submitRequest();
    enquedFaces = 0;
}

void FacialLandmarksDetection::enqueue(const cv::Mat& face) {
    if (!enabled()) {
        return;
    }
    if (enquedFaces == maxBatch) {
        LOG("WARN") << "检测到的面部数量超过最大值 (" << maxBatch << ") 人脸标记检测处理" << std::endl;
        return;
    }
    if (!request) {
        request = net.CreateInferRequestPtr();
    }

    InferenceEngine::Blob::Ptr inputBlob = request->GetBlob(input);

    matU8ToBlob<uint8_t>(face, inputBlob, enquedFaces);

    enquedFaces++;
}

std::vector<float> FacialLandmarksDetection::operator[] (int idx) const {
    std::vector<float> normedLandmarks;

    auto landmarksBlob = request->GetBlob(outputFacialLandmarksBlobName);
    auto n_lm = getTensorChannels(landmarksBlob->getTensorDesc());
    const float* normed_coordinates = request->GetBlob(outputFacialLandmarksBlobName)->buffer().as<float*>();

    if (doRawOutputMessages) {
        std::cout << "[" << idx << "] 元素, 规范的面部标记坐标 (x, y):" << std::endl;
    }

    auto begin = n_lm * idx;
    auto end = begin + n_lm / 2;
    for (auto i_lm = begin; i_lm < end; ++i_lm) {
        float normed_x = normed_coordinates[2 * i_lm];
        float normed_y = normed_coordinates[2 * i_lm + 1];

        if (doRawOutputMessages) {
            std::cout << normed_x << ", " << normed_y << std::endl;
        }

        normedLandmarks.push_back(normed_x);
        normedLandmarks.push_back(normed_y);
    }

    return normedLandmarks;
}

InferenceEngine::CNNNetwork FacialLandmarksDetection::read(const InferenceEngine::Core& ie) {
    LOG("INFO") << "加载 人脸标记检测网络 文件 ... " << std::endl;
    // Read network model
    auto network = ie.ReadNetwork(pathToModel);
    // Set maximum batch size
    network.setBatchSize(maxBatch);
    LOG("INFO") << "批量大小设置为  " << network.getBatchSize() << " 用于人脸标记检测网络" << std::endl;

    // ---------------------------Check inputs -------------------------------------------------------------
    LOG("INFO") << "检查 人脸标记检测网络 输入 ... " << std::endl;
    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("人脸标记检测网络应只有一个输入");
    }
    InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
    input = inputInfo.begin()->first;
    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Check outputs ------------------------------------------------------------
    LOG("INFO") << "检查 人脸标记检测网络 输出 ... " << std::endl;
    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
    const std::string outName = outputInfo.begin()->first;
    if (outName != outputFacialLandmarksBlobName) {
        throw std::logic_error("人脸标记检测网络输出层 未知： " + outName
            + ", should be " + outputFacialLandmarksBlobName);
    }
    InferenceEngine::Data& data = *outputInfo.begin()->second;
    data.setPrecision(InferenceEngine::Precision::FP32);
    const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
    if (outSizeVector.size() != 2 && outSizeVector.back() != 70) {
        throw std::logic_error("人脸标记检测网络输出层应具有2个维度，最后一个维度应为70");
    }

    LOG("INFO") << "将 人脸标记检测模型 加载到 [" << deviceForInference << "]" << std::endl;

    _enabled = true;
    return network;
}
#pragma endregion


#pragma region Load
Load::Load(BaseDetection& detector) : detector(detector)
{
}

void Load::into(InferenceEngine::Core& ie, const std::string& deviceName, bool enable_dynamic_batch) const {
    if (detector.enabled()) {
        std::map<std::string, std::string> config = { };
        bool isPossibleDynBatch = deviceName.find("CPU") != std::string::npos ||
            deviceName.find("GPU") != std::string::npos;

        if (enable_dynamic_batch && isPossibleDynBatch) {
            config[InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED] = InferenceEngine::PluginConfigParams::YES;
        }

        detector.net = ie.LoadNetwork(detector.read(ie), deviceName, config);
    }
}
#pragma endregion


#pragma region CallStat
CallStat::CallStat() :
    _number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) {
}

double CallStat::getSmoothedDuration() {
    //第一帧的持续时间需要对第一帧进行额外检查
    // 尚未计算可视化。
    if (_smoothed_duration < 0) {
        auto t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<ms>(t - _last_call_start).count();
    }
    return _smoothed_duration;
}

double CallStat::getTotalDuration() {
    return _total_duration;
}

double CallStat::getLastCallDuration() {
    return _last_call_duration;
}

void CallStat::calculateDuration() {
    auto t = std::chrono::high_resolution_clock::now();
    _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
    _number_of_calls++;
    _total_duration += _last_call_duration;
    if (_smoothed_duration < 0) {
        _smoothed_duration = _last_call_duration;
    }
    double alpha = 0.1;
    _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
}

void CallStat::setStartTime() {
    _last_call_start = std::chrono::high_resolution_clock::now();
}
#pragma endregion


#pragma region Timer
void Timer::start(const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        _timers[name] = CallStat();
    }
    _timers[name].setStartTime();
}

void Timer::finish(const std::string& name) {
    auto& timer = (*this)[name];
    timer.calculateDuration();
}

CallStat& Timer::operator[](const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        throw std::logic_error("没有名称的计时器 " + name + ".");
    }
    return _timers[name];
}
#pragma endregion