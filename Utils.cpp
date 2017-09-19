#include "Utils.h"

void Utils::calculateOpticalFlow(const Mat& previous, const Mat& current, Mat& flowU, Mat& flowV) {
	setDevice(0);
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	
	GpuMat current_d, previous_d, flowU_d, flowV_d;
	current_d.upload(current);
	previous_d.upload(previous);
	alg_tvl1(previous_d, current_d, flowU_d, flowV_d);
	flowU_d.download(flowU);
	flowV_d.download(flowV);		
}

SmallerDimension Utils::getSmallerDimension(const Mat& frame) {
	return frame.rows >= frame.cols ? col : row;
}

void Utils::scaleFramePerserveAR(Mat& frame) {
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
	resize(frame, frame, Size(round(scaleFactor * frame.cols), round(scaleFactor * frame.rows)));	
}

np::ndarray Utils::convertRGBFramesToNPArray(const std::list<Mat>& frames, uint batchSize, uint temporalWindow) {
	uint width = 224;
	uint height = 224;

	Py_Initialize();
	np::initialize();
	
	p::tuple shape = p::make_tuple(batchSize, temporalWindow, width, height, 3);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFrames = np::zeros(shape, dtype);

	float *resultData = reinterpret_cast<float*>(resultFrames.get_data());
	Rect rec((frames.front().cols / 2) - width / 2, (frames.front().rows / 2) - height / 2, width, height);
	for (uint i=0; i<batchSize; i++) {

		std::list<Mat>::const_iterator start = frames.begin();
		std::advance(start, i);
		std::list<Mat>::const_iterator end = frames.begin();
		std::advance(end, i + 1 + temporalWindow);

		uint j=0;
		for(std::list<Mat>::const_iterator it=start; it != end; ++it, j++) {
			Mat roi = (*it)(rec);
			for (uint k=0; k<height; k++) {
				Vec3b *row = roi.ptr<Vec3b>(k);
				for (uint l=0; l<width; l++) {
					uint index = i * (temporalWindow + 1) * height * width * 3 + j * height * width * 3 + k * width * 3 + l * 3;
					resultData[index + 0] = ((float)row[l].val[2] - 127.5f) / 127.5f;
					resultData[index + 1] = ((float)row[l].val[1] - 127.5f) / 127.5f;
					resultData[index + 2] = ((float)row[l].val[0] - 127.5f) / 127.5f;					
				}
			}	
		}
	}
	return resultFrames;
}

np::ndarray Utils::convertOpticalFlowsToNPArray(const std::list<Mat>& flows, uint batchSize, uint temporalWindow, float bound) {
	uint width = 224;
	uint height = 224;

	Py_Initialize();
	np::initialize();
	
	p::tuple shape = p::make_tuple(batchSize, temporalWindow, width, height, 2);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFlows = np::zeros(shape, dtype);

	float *resultData = reinterpret_cast<float*>(resultFlows.get_data());
	Rect rec((flows.front().cols / 2) - width/2, (flows.front().rows / 2) - height/2, width, height);	

	for (uint i=0; i<batchSize; i++) {
		
		std::list<Mat>::const_iterator start = flows.begin();
		std::advance(start, i);
		std::list<Mat>::const_iterator end = flows.begin();
		std::advance(end, i + 1 + temporalWindow);
		uint j=0;
		for (std::list<Mat>::const_iterator it = flows.begin(); it != end; j+=2) {
			Mat roiU = (*it)(rec);
			std::advance(it, 1);
			Mat roiV = (*it)(rec);
			for (uint k=0; k<height; k++) {
				float *rowU = roiU.ptr<float>(k);
				float *rowV = roiV.ptr<float>(k);
				for (uint l=0; l<width; l++) {
					uint index = i * (temporalWindow + 1) * height * width * 2 + (j/2) * height * width * 2 + k * width * 2 + l * 2;
					resultData[index + 0] = CAST(rowU[l], -bound, bound) / bound;
					resultData[index + 1] = CAST(rowV[l], -bound, bound) / bound;					
				}
			}
		}
	}
	return resultFlows;	
}