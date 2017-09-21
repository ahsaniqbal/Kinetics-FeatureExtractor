#include "PreProcessor.h"

void PreProcessor::initialize(const char* video) {
	frames.clear();
	flows.clear();
	if (!video) {
		PyErr_SetString(PyExc_TypeError, "Video File shouldn't be null or empty");
		p::throw_error_already_set();
	}

	populateFrames(video);
	if (frames.size() == 0) {
		PyErr_SetString(PyExc_TypeError, "Unable to Process video");
		p::throw_error_already_set();	
	}

	populateOpticalFlows();
	if (flows.size() == 0) {
		PyErr_SetString(PyExc_TypeError, "Unable to Process video");
		p::throw_error_already_set();			
	}
}

void PreProcessor::populateFrames(const char* video) {
	VideoCapture capture;
	capture.open(video);
	if (!capture.isOpened()) {
		PyErr_SetString(PyExc_TypeError, "Unable to open the video");
		p::throw_error_already_set();
	}

	double fps = capture.get(CV_CAP_PROP_FPS);
	int step = 1;
	/*if (!std::isnan(fps) && fps > 25.0) {
		step = (int)round(fps/25.0);
	}*/

	Mat frame;
	int frameNum = 0;
	while(true) {
		capture>>frame;
		if (frame.empty()) {
			break;
		}
		capture.set(CV_CAP_PROP_POS_FRAMES, frameNum + step);
		frameNum += step;
		Utils::scaleFramePerserveAR(frame);
		frames.push_back(frame);
	}
}

void PreProcessor::populateOpticalFlows() {
	setDevice(0);
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	
	Mat currentGray, previousGray, flowU, flowV;
	GpuMat currentGray_d, previousGray_d, flowU_d, flowV_d;
	for (uint i=0; i<frames.size(); i++) {
		if ( i == 0 ) {
			cvtColor(frames.at(i), previousGray, CV_BGR2GRAY);

			continue;
		}
		cvtColor(frames.at(i), currentGray, CV_BGR2GRAY);

		currentGray_d.upload(currentGray);
		previousGray_d.upload(previousGray);

		alg_tvl1(previousGray_d, currentGray_d, flowU_d, flowV_d);
		flowU_d.download(flowU);
		flowV_d.download(flowV);

		flows.push_back(flowU.clone());
		flows.push_back(flowV.clone());
		std::swap(previousGray, currentGray);
	}
}

np::ndarray PreProcessor::getOpticalFlows(float bound) {
	uint width = 224;
	uint height = 224;

	Py_Initialize();
	np::initialize();
	
	p::tuple shape = p::make_tuple(flows.size()/2, width, height, 2);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFlows = np::zeros(shape, dtype);

	float *resultData = reinterpret_cast<float*>(resultFlows.get_data());
	Rect rec((flows.at(0).cols / 2) - width/2, (flows.at(0).rows / 2) - height/2, width, height);	
	for (uint i=0; i<flows.size(); i+=2) {
		Mat roiU = flows.at(i)(rec);
		Mat roiV = flows.at(i+1)(rec);
		for (uint j=0; j<height; j++) {
			float* rowU = roiU.ptr<float>(j);
			float* rowV = roiV.ptr<float>(j);
			for (uint k=0; k<width; k++) {
				uint index = (i/2) * height * width * 2 + j * width * 2 + k * 2;
				resultData[index + 0] = CAST(rowU[k], -bound, bound) / bound;
				resultData[index + 1] = CAST(rowV[k], -bound, bound) / bound;
			}
		}
	}
	return resultFlows;
}

np::ndarray PreProcessor::getFrames() {
	uint width = 224;
	uint height = 224;

	Py_Initialize();
	np::initialize();
	
	p::tuple shape = p::make_tuple(frames.size(), width, height, 3);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFrames = np::zeros(shape, dtype);

	float *resultData = reinterpret_cast<float*>(resultFrames.get_data());
	Rect rec((frames.at(0).cols / 2) - width / 2, (frames.at(0).rows / 2) - height / 2, width, height);
	for (uint i=0; i<frames.size(); i++) {
		Mat roi = frames.at(i)(rec);
		for (uint j=0; j<height; j++) {
			Vec3b *row = roi.ptr<Vec3b>(j);
			for (uint k=0; k<width; k++) {
				uint index = i * height * width * 3 + j * width * 3 + k * 3;
				resultData[index + 0] = ((float)row[k].val[2] - 127.5f) / 127.5f;
				resultData[index + 1] = ((float)row[k].val[1] - 127.5f) / 127.5f;
				resultData[index + 2] = ((float)row[k].val[0] - 127.5f) / 127.5f;
			}
		}
	}	
	return resultFrames;
}

