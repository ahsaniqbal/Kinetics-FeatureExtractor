#include "Converter.h"
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace cv::gpu;

typedef std::vector<float> ResultFlow;
typedef unsigned int uint;

#define CAST(v, L, H) ((v) >= (H) ? (H) : (v) <= (L) ? (L) : (v))

enum SmallerDimension { row, col };
SmallerDimension getSmallerDimension(const Mat& frame);
void scaleFramePreserveAR(Mat& frame);


void init(const Mat& frame, Mat& currentColor, Mat& previousColor, Mat& currentGray, Mat& previousGray) {
	currentColor.create(frame.size(), CV_8UC3);
	currentGray.create(frame.size(), CV_8UC1);
	
	previousColor.create(frame.size(), CV_8UC3);
	previousGray.create(frame.size(), CV_8UC1);

	frame.copyTo(previousColor);
	scaleFramePreserveAR(previousColor);
	cvtColor(previousColor, previousGray, CV_BGR2GRAY);
}

void convert(const std::vector<Mat*>& input, std::vector<float>& result, int bound) {
	uint rows = 224;
	uint cols = 224;
	Rect rec((input.at(0)->cols / 2) - (cols / 2), (input.at(0)->rows / 2) - (rows / 2), cols, rows);
	result[0] = input.size()/2;
	for (uint i=0; i<input.size(); i+=2) {
		Mat roiU = (*input.at(i))(rec);
		Mat roiV = (*input.at(i+1))(rec);
		for (uint j=0; j<rows; j++) {
			float* rowU = roiU.ptr<float>(j);
			float* rowV = roiV.ptr<float>(j);
			for (uint k=0; k<cols; k++) {
				result[1 + i * rows * cols + j * rows * 2 + k * 2] = CAST(rowU[k], -1*bound, bound)/bound;
				result[1 + i * rows * cols + j * rows * 2 + k * 2 + 1] = CAST(rowV[k], -1*bound, bound)/bound;
			}	
		}
	}
}

void cleanUp(std::vector<Mat*>& input) {
	for (uint i=0; i<input.size(); i++) {
		delete input.at(i);
	}
}

SmallerDimension getSmallerDimension(const Mat& frame) {
	return frame.rows >= frame.cols ? col : row;
}

void scaleFramePreserveAR(Mat& frame) {
	SmallerDimension smallDim = getSmallerDimension(frame);
	float scaleFactor = 1.0f;
	switch(smallDim) {
		case row:
			scaleFactor = 256.0f / frame.rows;
			break;
		case col:
		default:
			scaleFactor = 256.0f / frame.cols;
			break;
	}
	//std::cout<<frame.rows<<"::"<<frame.cols<<std::endl;
	resize(frame, frame, Size(round(scaleFactor * frame.cols), round(scaleFactor * frame.rows)));
	std::cout<<frame.rows<<"::"<<frame.cols<<std::endl;
}

void calculateOpFlow(const char* video, int step, std::vector<Mat*>& result) {	
	if (!video) {
		std::cout<<"Unable to open video";
		return;
	}

	VideoCapture capture(video);
	if(!capture.isOpened()) {
		std::cout<<"Unable to open the video";
		return;
	}

	int frameNum = 0;
	Mat currentColor, previousColor, previousGray, currentGray, frame, *flowU, *flowV;
	GpuMat previousGray_d, currentGray_d, flowU_d, flowV_d;

	setDevice(0);
	OpticalFlowDual_TVL1_GPU alg_tvl1;

	while(true) {
		capture>>frame;
		if (frame.empty()) {
			break;
		}
		if (frameNum == 0) {
			init(frame, currentColor, previousColor, currentGray, previousGray);
			
			capture.set(CV_CAP_PROP_POS_FRAMES, frameNum + step);
			frameNum += step;
			continue;
		}
		frame.copyTo(currentColor);
		scaleFramePreserveAR(currentColor);
		cvtColor(currentColor, currentGray, CV_BGR2GRAY);

		previousGray_d.upload(previousGray);
		currentGray_d.upload(currentGray);

		alg_tvl1(previousGray_d, currentGray_d, flowU_d, flowV_d);

		flowU = new Mat();
		flowV = new Mat();		

		flowU_d.download(*flowU);
		flowV_d.download(*flowV);

		result.push_back(flowU);
		result.push_back(flowV);

		std::swap(previousGray, currentGray);
		capture.set(CV_CAP_PROP_POS_FRAMES, frameNum + step);
		frameNum += step;
	}
}

ResultFlow getOpticalFlow(const char* video, int step, int bound) {
	std::vector<Mat*> flows;
	calculateOpFlow(video, step, flows);

	if (flows.size() == 0) {
		ResultFlow ret;
		return ret;
	}

	std::vector<float> result(1 + flows.size() * 224 * 224, 0);
	convert(flows, result, bound);
	cleanUp(flows);
	return result;	
}


BOOST_PYTHON_MODULE(libpyOpFlow) {
	using namespace boost::python;

	class_<ResultFlow>("ResultFlow")
		.def(vector_indexing_suite<ResultFlow>());

	def("getOpticalFlow", getOpticalFlow);	
}
