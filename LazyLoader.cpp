#include "LazyLoader.h"
#include <boost/python.hpp>
using namespace boost::python;
#define CAST(v, L, H) ((v) >= (H) ? (H) : (v) <= (L) ? (L) : (v))

void LazyLoader::initializeLazy(const char* video, const uint batchSize, const uint temporalWindow) {
	frames.clear();
	flows.clear();
	if (!video) {
		PyErr_SetString(PyExc_TypeError, "Video File shouldn't be null or empty");
		p::throw_error_already_set();
	}
	capture.open(video);
	frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
}


void LazyLoader::createBatch(const uint batchSize, const uint temporalWindow) {
	initFramesLazy(batchSize, temporalWindow);
	initFlowLazy();
}

void LazyLoader::appendFrame(Mat& mat, const uint count) {
	for (uint i=0; i<count; i++) {
		frames.push_back(mat.clone());
	}
}

bool LazyLoader::hasNextBatch() {
	return frameCount>0;
}
void LazyLoader::initFramesLazy(const uint batchSize, const uint temporalWindow) {
	if (!capture.isOpened()) {
		PyErr_SetString(PyExc_TypeError, "Unable to open the video");
		p::throw_error_already_set();
	}

	uint temporalWindowHalf = temporalWindow / 2;
	Mat frame;
	
	//+1 to make sure that the size of optical flow volume is same for complete batch
	for (uint i=0; i<batchSize+(temporalWindowHalf)+1; i++) {
		capture>>frame;
		if (frame.empty()) {
			if (i==0) {
				PyErr_SetString(PyExc_TypeError, "Video is curropted, unable to read first frame");
				p::throw_error_already_set();
			}
			break;
		}
		Utils::scaleFramePerserveAR(frame);
		frames.push_back(frame.clone());
	}
	//appending the first frame, to make sure that for each frame we have half temporal window on either side
	for (uint i=0; i<temporalWindowHalf; i++) {
		frames.push_front(frames.front().clone());
	}
	//+1 to make sure that the size of optical flow volume is same for complete batch
	while(frames.size() != std::min(batchSize + temporalWindow, frameCount + temporalWindow) + 1) {
		frames.push_back(frames.back().clone());
	}
}

void LazyLoader::initFlowLazy() {
	Mat previous, current, flowU, flowV;
	for (std::list<Mat>::iterator it=frames.begin(); it != frames.end(); ++it) {
		if (it == frames.begin()) {
			cvtColor(*it, previous, CV_BGR2GRAY);
			continue;
		}
		cvtColor(*it, current, CV_BGR2GRAY);
		Utils::calculateOpticalFlow(previous, current, flowU, flowV);
		std::swap(previous, current);
	}
}

np::ndarray LazyLoader::nextBatchFrames() {

	np::ndarray resultFrames = Utils::convertRGBFramesToNPArray(frames, std::min(batchSize, frameCount), temporalWindow);
	//TODO:: How Many should I pop
	for (uint i=0; i<batchSize; i++) {
		frames.pop_front();
		frameCount--;
	}
	Mat frame, flowU, flowV;
	for (uint i=0; i<batchSize; i++) {
		capture>>frame;
		if (frame.empty()) {
			break;
		}
		Utils::scaleFramePerserveAR(frame);
		Utils::calculateOpticalFlow(frames.back(), frame, flowU, flowV);
		frames.push_back(frame.clone());
		flows.push_back(flowU.clone());
		flows.push_back(flowV.clone());
	}
	while(frames.size() != std::min(batchSize + temporalWindow, frameCount + temporalWindow)+1) {
		frames.push_back(frames.back().clone());
		Utils::calculateOpticalFlow(frames.back(), frames.back(), flowU, flowV);
		flows.push_back(flowU.clone());
		flows.push_back(flowV.clone());	
	}
	return resultFrames;	
}

np::ndarray LazyLoader::nextBatchFlows() {
	np::ndarray resultFlows = Utils::convertOpticalFlowsToNPArray(flows, std::min(batchSize, frameCount), temporalWindow, 20.0f);
	for (uint i=0; i<batchSize; i++) {
		flows.pop_front();
		flows.pop_front();
	}
	return resultFlows;
}

