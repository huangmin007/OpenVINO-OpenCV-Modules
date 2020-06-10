#include "pose_detection.hpp"
#include "static_functions.hpp"
//#include "samples/ocv_common.hpp"
#include <opencv2/opencv.hpp>

using namespace space;


PoseDetection::PoseDetection() :OpenModelInferBase({}, true)
{
}
PoseDetection::~PoseDetection()
{
}


void PoseDetection::ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
{
    InferenceEngine::Blob::Ptr data;
    request->GetBlob(outputsInfo.begin()->first.c_str(), data, 0);

    float* detections = data->buffer().as<float*>();

    int H = 46, W = 46;
    std::vector<cv::Point> points(44);

    float SX = frame_ptr.cols / W;
    float SY = frame_ptr.rows / H;
    //std::vector<cv::Mat> heatMaps(44);

    for (int i = 0; i < 44; i++)
    {
        //切片对应身体部位的热图.
        cv::Mat heatMap(H, W, CV_32F, (void*)(detections + i * 46 * 46));

        double conf;
        cv::Point p(-1, -1), pm;        
        cv::minMaxLoc(heatMap, 0, &conf, 0, &pm);

        if (conf > 0.1) p = pm;
        points[i] = p;
    }

    
    //cv::Mat frame = cv::Mat::zeros(input_frame_width, input_frame_height);
    cv::Mat img;
    frame_ptr.copyTo(img);

    for (int n = 0; n < 44; n++)
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
        //std::cout << "Point" << n << ": " << a.x << "," << a.y << std::endl;


        //cv::line(img, a, b, cv::Scalar(0, 200, 0), 2);
        cv::circle(img, a, 3, cv::Scalar(0, 0, 200), -1);
        //cv::circle(img, b, 3, cv::Scalar(0, 0, 200), -1);
    }

    cv::imshow("Test", img);
    cv::waitKey(1);
    
}

void PoseDetection::UpdateDebugShow()
{

}