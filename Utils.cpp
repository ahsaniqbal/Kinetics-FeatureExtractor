#include "Utils.h"

void Utils::populateOpticalFlow(const std::list<Mat>& frames, std::list<Mat>& flows) {
	Mat previous, current, flow;
	for (std::list<Mat>::const_iterator it=frames.begin(); it != frames.end(); ++it) {
		if (it == frames.begin()) {
			cvtColor(*it, previous, CV_BGR2GRAY);
			continue;
		}
		cvtColor(*it, current, CV_BGR2GRAY);
		calculateOpticalFlow(previous, current, flow);
		flows.push_back(flow.clone());
		std::swap(previous, current);
	}
}

void Utils::calculateOpticalFlow(const Mat& previous, const Mat& current, Mat& result) {
	setDevice(0);
	OpticalFlowDual_TVL1_GPU alg_tvl1;

	Mat flowU, flowV;
	GpuMat current_d, previous_d, flowU_d, flowV_d;
	current_d.upload(current);
	previous_d.upload(previous);

	alg_tvl1(previous_d, current_d, flowU_d, flowV_d);
	flowU_d.download(flowU);
	flowV_d.download(flowV);		

	std::vector<Mat> channels;
	channels.push_back(flowU);
	channels.push_back(flowV);

	merge(channels, result);
}

SmallerDimension Utils::getSmallerDimension(const Mat& frame) {
	return frame.rows >= frame.cols ? col : row;
}

void Utils::scaleFramePerserveAR(const Mat& frame, Mat& result) {
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
	resize(frame, result, Size(round(scaleFactor * frame.cols), round(scaleFactor * frame.rows)));	
}

void Utils::crop(const Mat& frame, Mat& result) {
	uint width = 224;
	uint height = 224;
	scaleFramePerserveAR(frame, result);
	Rect rec((result.cols / 2) - width/2, (result.rows / 2) - height/2, width, height);
	result = (result)(rec);
}

void Utils::writeRGBFrame(const Mat& frame, float* array, uint startIndex) {
 	for (uint i=0; i<(uint)frame.rows; i++) {
 		const Vec3b *row = frame.ptr<Vec3b>(i);
 		for (uint j=0; j<(uint)frame.cols; j++) {
 			uint index = startIndex + i * frame.cols * 3 + j * 3;
 			array[index + 0] = ((float)row[j].val[2] - 127.5f) / 127.5f;
 			array[index + 1] = ((float)row[j].val[1] - 127.5f) / 127.5f;
 			array[index + 2] = ((float)row[j].val[0] - 127.5f) / 127.5f;
 		}
 	}
}

void Utils::writeFlowFrame(const Mat& frame, float* array, uint startIndex, float bound) {
	for (uint i=0; i<(uint)frame.rows; i++) {
		const Vec2f* row = frame.ptr<Vec2f>(i);
		for (uint j=0; j<(uint)frame.cols; j++) {
			uint index = startIndex + i * frame.cols * 2 + j * 2;
			array[index + 0] = CAST(row[j].val[0], -bound, bound) / bound;
			array[index + 1] = CAST(row[j].val[1], -bound, bound) / bound;
		}
	}
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
	for (uint i=0; i<batchSize; i++) {

		std::list<Mat>::const_iterator start = frames.begin();
		std::advance(start, i);
		std::list<Mat>::const_iterator end = frames.begin();
		std::advance(end, i + temporalWindow);

		uint j=0;
		for(std::list<Mat>::const_iterator it=start; it != end; ++it, j++) {
			Mat roi;
			crop(*it, roi);
			uint startIndex = i * temporalWindow * height * width * 3 + j * height * width * 3;
			writeRGBFrame(roi, resultData, startIndex);
		}
	}
	return resultFrames;
}

np::ndarray Utils::convertRGBFramesToNPArray(const std::list<Mat>& frames) {
	uint width = 224;
	uint height = 224;

	Py_Initialize();
	np::initialize();
	
	p::tuple shape = p::make_tuple(1, frames.size(), width, height, 3);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFrames = np::zeros(shape, dtype);	
	
	float *resultData = reinterpret_cast<float*>(resultFrames.get_data());

	uint i=0;
	for(std::list<Mat>::const_iterator it = frames.begin(); it != frames.end(); ++it, i++) {
		Mat roi;
		crop(*it, roi);
		uint startIndex = i * height * width * 3;
		writeRGBFrame(roi, resultData, startIndex);
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

	for (uint i=0; i<batchSize; i++) {
		
		std::list<Mat>::const_iterator start = flows.begin();
		std::advance(start, i);
		std::list<Mat>::const_iterator end = flows.begin();
		std::advance(end, i + temporalWindow);

		uint j=0;
		for (std::list<Mat>::const_iterator it = start; it != end; ++it, j++) {
			Mat roi;
			crop(*it, roi);
			uint startIndex = i * temporalWindow * height * width * 2 + j * height * width * 2;
			writeFlowFrame(roi, resultData, startIndex, bound);
		}
	}
	return resultFlows;	
}

np::ndarray Utils::convertOpticalFlowsToNPArray(const std::list<Mat>& flows, float bound) {
	uint width = 224;
	uint height = 224;

	Py_Initialize();
	np::initialize();
	
	p::tuple shape = p::make_tuple(1, flows.size(), width, height, 2);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFlows = np::zeros(shape, dtype);

	float *resultData = reinterpret_cast<float*>(resultFlows.get_data());	

	uint i=0;
	for (std::list<Mat>::const_iterator it = flows.begin(); it != flows.end(); ++it, i++) {
		Mat roi;
		crop(*it, roi);
		uint startIndex = i * height * width * 2;
		writeFlowFrame(roi, resultData, startIndex, bound);
	}

	return resultFlows;
}