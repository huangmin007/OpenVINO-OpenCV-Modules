#pragma once

//#include <open_model_infer.hpp>

#include <static_functions.hpp>
#include <io_data_format.hpp>
#include <shared_mutex>
#include "timer.hpp"


namespace space
{    
    struct HumanPose
    {
        HumanPose(const std::vector<cv::Point2f>& keypoints = std::vector<cv::Point2f>(), const float& score = 0.0f);

        std::vector<cv::Point2f> keypoints;
        float score;
    };

    struct Frame
    {
        int id;
        int width;
        int height;
        int size0;
        int size1;
        int size2;

        friend std::ostream& operator << (std::ostream& stream, const Frame* frame)
        {
            stream << "[ID:" << frame->id <<
                " Width:" << frame->width <<
                " Height:" << frame->height <<
                " size0:" << frame->size0 <<
                " size1:" << frame->size1 <<
                " size2:" << frame->size2 <<
                "]";

            return stream;
        };
    };

    class HumanPoseDetection// : public OpenModelInferBase
    {
    public:
        HumanPoseDetection(bool is_debug = true);
        HumanPoseDetection(bool is_debug = true, InferenceEngine::ColorFormat color_format = InferenceEngine::ColorFormat::BGR);
        ~HumanPoseDetection();



        /// <summary>
        /// 配置网络模型
        /// </summary>
        /// <param name="ie">ie推断引擎对象</param>
        /// <param name="model_info">具有一定协议格式的模型参数，格式：(模型名称[:精度[:硬件]])，示例：human-pose-estimation-0001:FP32:CPU</param>
        /// <param name="is_reshape">是否重塑输入层，重新调整输入，指按原帧数据尺寸、数据指针位置做为输入(减少内存数据复制)，做异步推断请求</param>
        void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_info);

        /// <summary>
        /// 配置网络模型
        /// </summary>
        /// <param name="ie">ie推断引擎对象</param>
        /// <param name="model_path">ai模型文件路径</param>
        /// <param name="device">推断使用的硬件类型</param>
        /// <param name="is_reshape">是否重塑输入层，重新调整输入，指按原帧数据尺寸、数据指针位置做为输入(减少内存数据复制)，做异步推断请求</param>
        void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device);

        /// <summary>
        /// 异步推断请求，需实时提交图像帧；当 is_reshape 为 true 时只需调用一次，多次调用也不影响
        /// </summary>
        /// <param name="frame">图像帧对象</param>
        void RequestInfer(const cv::Mat& frame);

        //void RequestInfer(const std::string& name);


        /// <summary>
        /// 获取上一帧推断所用耗时 ms
        /// </summary>
        /// <returns></returns>
        double GetInferUseTime()
        {
            std::shared_lock<std::shared_mutex> lock(_time_mutex);
            return infer_use_time;
        }

        /// <summary>
        /// 获取每秒推断结果的数量
        /// </summary>
        /// <returns></returns>
        size_t GetFPS()
        {
            std::shared_lock<std::shared_mutex> lock(_fps_mutex);
            return fps;
        }

    protected:

        /// <summary>
        /// 设置网络输入/输出
        /// </summary>
        void SetNetworkIO();

        /// <summary>
        /// 设置推断请求异步回调
        /// </summary>
        void SetInferCallback();


        /// <summary>
        /// 解析指定的输出层数据；也可做是二次输入/输出，或是调试显示输出
        /// <param> 如果要在类内部解析，在子类内实现该函数；如果在类外部解析设置回调函数 SetCompletionCallback </param>
        /// </summary>
        void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status);

        /// <summary>
        /// 更新调试并显示，该函数是用于子类做输出显示的
        /// </summary>
        void UpdateDebugShow();

        void ConvertToSharedXML(const std::vector<HumanPose>& poses, const std::string shared_name);


        // IE 引擎插件配置
        std::map<std::string, std::string> ie_config = { };

        bool is_debug = true;			//是否输出部份调试信息
        std::stringstream debug_title;	//调试输出窗口标题

        InferenceEngine::ColorFormat color_format = InferenceEngine::ColorFormat::BGR;

        //CNN 网络对象
        InferenceEngine::CNNNetwork cnnNetwork;
        //可执行网络对象
        InferenceEngine::ExecutableNetwork execNetwork;

        //输入层数据信息
        InferenceEngine::InputsDataMap inputsInfo;
        //输出层数据信息
        InferenceEngine::OutputsDataMap outputsInfo;
        // 推断请求对象
        std::vector<InferenceEngine::InferRequest::Ptr> requestPtrs;

        size_t infer_count = 1;				//创建的推断请求数量
        bool is_configuration = false;		//是否已经设置网络配置

        //引用输入的帧对象，主要是需要帧的宽，高，通道，类型、数据指针信息
        cv::Mat frame_ptr;

        //用于测量的计时器
        Timer timer;
        //FPS
        size_t fps = 0;
        size_t fps_count = 0;
        std::shared_mutex _fps_mutex;

        //获取到推断结果所耗时间
        double infer_use_time = 0.0f;
        std::shared_mutex _time_mutex;
        std::chrono::steady_clock::time_point t0;
        std::chrono::steady_clock::time_point t1;

    private:
        cv::Mat debug_frame;
        std::vector<HumanPose> poses;
        std::vector<std::string> output_layers;


        HANDLE pMapFile;
        LPVOID pBuffer;
        OutputSourceData *osd_data;

        Frame* frame;
        HANDLE inFile;
        LPVOID inBuffer;

        float minPeaksDistance = 3.0f;
        static const int upsampleRatio = 4;
        static const size_t keypointsNumber = 18;
        float midPointsScoreThreshold = 0.05f;
        float foundMidPointsRatioThreshold = 0.8f;
        int minJointsNumber = 3;
        float minSubsetScore = 0.2f;
        int stride = 8;

        std::vector<HumanPose> extractPoses(const std::vector<cv::Mat>& heatMaps, const std::vector<cv::Mat>& pafs) const;
        void correctCoordinates(std::vector<HumanPose>& poses, const cv::Size& featureMapsSize, const cv::Size& imageSize) const;

        void RenderHumanPose(const std::vector<HumanPose>& poses, cv::Mat& image);
    };
}
