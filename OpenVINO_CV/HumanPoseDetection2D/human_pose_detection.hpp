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
        /// ��������ģ��
        /// </summary>
        /// <param name="ie">ie�ƶ��������</param>
        /// <param name="model_info">����һ��Э���ʽ��ģ�Ͳ�������ʽ��(ģ������[:����[:Ӳ��]])��ʾ����human-pose-estimation-0001:FP32:CPU</param>
        /// <param name="is_reshape">�Ƿ���������㣬���µ������룬ָ��ԭ֡���ݳߴ硢����ָ��λ����Ϊ����(�����ڴ����ݸ���)�����첽�ƶ�����</param>
        void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_info);

        /// <summary>
        /// ��������ģ��
        /// </summary>
        /// <param name="ie">ie�ƶ��������</param>
        /// <param name="model_path">aiģ���ļ�·��</param>
        /// <param name="device">�ƶ�ʹ�õ�Ӳ������</param>
        /// <param name="is_reshape">�Ƿ���������㣬���µ������룬ָ��ԭ֡���ݳߴ硢����ָ��λ����Ϊ����(�����ڴ����ݸ���)�����첽�ƶ�����</param>
        void ConfigNetwork(InferenceEngine::Core& ie, const std::string& model_path, const std::string& device);

        /// <summary>
        /// �첽�ƶ�������ʵʱ�ύͼ��֡���� is_reshape Ϊ true ʱֻ�����һ�Σ���ε���Ҳ��Ӱ��
        /// </summary>
        /// <param name="frame">ͼ��֡����</param>
        void RequestInfer(const cv::Mat& frame);

        //void RequestInfer(const std::string& name);


        /// <summary>
        /// ��ȡ��һ֡�ƶ����ú�ʱ ms
        /// </summary>
        /// <returns></returns>
        double GetInferUseTime()
        {
            std::shared_lock<std::shared_mutex> lock(_time_mutex);
            return infer_use_time;
        }

        /// <summary>
        /// ��ȡÿ���ƶϽ��������
        /// </summary>
        /// <returns></returns>
        size_t GetFPS()
        {
            std::shared_lock<std::shared_mutex> lock(_fps_mutex);
            return fps;
        }

    protected:

        /// <summary>
        /// ������������/���
        /// </summary>
        void SetNetworkIO();

        /// <summary>
        /// �����ƶ������첽�ص�
        /// </summary>
        void SetInferCallback();


        /// <summary>
        /// ����ָ������������ݣ�Ҳ�����Ƕ�������/��������ǵ�����ʾ���
        /// <param> ���Ҫ�����ڲ���������������ʵ�ָú�������������ⲿ�������ûص����� SetCompletionCallback </param>
        /// </summary>
        void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status);

        /// <summary>
        /// ���µ��Բ���ʾ���ú��������������������ʾ��
        /// </summary>
        void UpdateDebugShow();

        void ConvertToSharedXML(const std::vector<HumanPose>& poses, const std::string shared_name);


        // IE ����������
        std::map<std::string, std::string> ie_config = { };

        bool is_debug = true;			//�Ƿ�������ݵ�����Ϣ
        std::stringstream debug_title;	//����������ڱ���

        InferenceEngine::ColorFormat color_format = InferenceEngine::ColorFormat::BGR;

        //CNN �������
        InferenceEngine::CNNNetwork cnnNetwork;
        //��ִ���������
        InferenceEngine::ExecutableNetwork execNetwork;

        //�����������Ϣ
        InferenceEngine::InputsDataMap inputsInfo;
        //�����������Ϣ
        InferenceEngine::OutputsDataMap outputsInfo;
        // �ƶ��������
        std::vector<InferenceEngine::InferRequest::Ptr> requestPtrs;

        size_t infer_count = 1;				//�������ƶ���������
        bool is_configuration = false;		//�Ƿ��Ѿ�������������

        //���������֡������Ҫ����Ҫ֡�Ŀ��ߣ�ͨ�������͡�����ָ����Ϣ
        cv::Mat frame_ptr;

        //���ڲ����ļ�ʱ��
        Timer timer;
        //FPS
        size_t fps = 0;
        size_t fps_count = 0;
        std::shared_mutex _fps_mutex;

        //��ȡ���ƶϽ������ʱ��
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
