#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <open_model_infer.hpp>

using namespace space;

class PoseDetection:public OpenModelInferBase
{
public:
	struct Result
	{
		int label;
		float confidence;
		std::vector<cv::Point2f> keypoints;
	};
	PoseDetection();
	~PoseDetection();


protected:
	float confidenceThreshold;	//÷√–≈∂»„–÷µ
	void UpdateDebugShow() override;
	void ParsingOutputData(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status) override;

private:
	int H, W;
	const int POSE_PAIRS[3][20][2] = {
		{   // COCO body
			{ 1,2 },{ 1,5 },{ 2,3 },
			{ 3,4 },{ 5,6 },{ 6,7 },
			{ 1,8 },{ 8,9 },{ 9,10 },
			{ 1,11 },{ 11,12 },{ 12,13 },
			{ 1,0 },{ 0,14 },
			{ 14,16 },{ 0,15 },{ 15,17 }
		},
		{   // MPI body
			{ 0,1 },{ 1,2 },{ 2,3 },
			{ 3,4 },{ 1,5 },{ 5,6 },
			{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
			{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
		},
		{   // hand
			{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
			{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
			{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
			{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
			{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
		} };
};

