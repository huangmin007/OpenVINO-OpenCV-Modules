#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

struct BaseDetection 
{
    InferenceEngine::ExecutableNetwork net;
    InferenceEngine::InferRequest::Ptr request;
    std::string topoName;
    std::string pathToModel;        //模型文件路径
    std::string deviceForInference;
    const size_t maxBatch;
    bool isBatchDynamic;
    const bool isAsync;
    mutable bool enablingChecked;
    mutable bool _enabled;
    const bool doRawOutputMessages;

    BaseDetection(const std::string& topoName,
        const std::string& pathToModel,
        const std::string& deviceForInference,
        int maxBatch, bool isBatchDynamic, bool isAsync,
        bool doRawOutputMessages);

    virtual ~BaseDetection();

    InferenceEngine::ExecutableNetwork* operator ->();
    virtual InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) = 0;
    virtual void submitRequest();
    virtual void wait();
    bool enabled() const;
    void printPerformanceCounts(std::string fullDeviceName);
};


struct FaceDetection : BaseDetection 
{
    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::string input;
    std::string output;
    std::string labels_output;
    double detectionThreshold;
    int maxProposalCount;
    int objectSize;
    int enquedFrames;
    float width;
    float height;
    float network_input_width;
    float network_input_height;
    float bb_enlarge_coefficient;
    float bb_dx_coefficient;
    float bb_dy_coefficient;
    bool resultsFetched;
    std::vector<Result> results;    //推断结果数据

    FaceDetection(const std::string& pathToModel,
        const std::string& deviceForInference,
        int maxBatch, bool isBatchDynamic, bool isAsync,
        double detectionThreshold, bool doRawOutputMessages,
        float bb_enlarge_coefficient, float bb_dx_coefficient,
        float bb_dy_coefficient);

    InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) override;
    void submitRequest() override;      //提交请求

    //帧入队
    void enqueue(const cv::Mat& frame);     
    void fetchResults();
};

//头部姿态检测
struct HeadPoseDetection : BaseDetection 
{
    struct Results {
        float angle_r;
        float angle_p;
        float angle_y;
    };

    std::string input;
    std::string outputAngleR;
    std::string outputAngleP;
    std::string outputAngleY;
    size_t enquedFaces;
    cv::Mat cameraMatrix;

    HeadPoseDetection(const std::string& pathToModel,
        const std::string& deviceForInference,
        int maxBatch, bool isBatchDynamic, bool isAsync,
        bool doRawOutputMessages);

    InferenceEngine::CNNNetwork read(const InferenceEngine::Core& ie) override;
    void submitRequest() override;

    void enqueue(const cv::Mat& face);
    Results operator[] (int idx) const;
};