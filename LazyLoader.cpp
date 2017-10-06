#include "LazyLoader.h"

void LazyLoader::initializeLazy(const char* video, const uint batchSize, const uint temporalWindow) {
	frames.clear();
	flows.clear();
	if (!video) {
		PyErr_SetString(PyExc_TypeError, "Video File shouldn't be null or empty");
		p::throw_error_already_set();
	}
	capture.open(video);
	this->frameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);

	this->batchSize = batchSize;
	this->temporalWindow = temporalWindow;

	std::cout<<this->frameCount<<std::endl;
	createBatch();
}


void LazyLoader::createBatch() {
	initFramesLazy();
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
void LazyLoader::initFramesLazy() {
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
		//Utils::scaleFramePerserveAR(frame);
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
	Utils::populateOpticalFlow(frames, flows);
}

np::ndarray LazyLoader::nextBatchFrames() {
	batchToLoad = std::min(batchSize, frameCount);
	np::ndarray resultFrames = Utils::convertRGBFramesToNPArray(frames, batchToLoad, temporalWindow);
	
	for (uint i=0; i<batchSize && frameCount != 0; i++) {
		frames.pop_front();
		frameCount--;
	}

	if (frameCount > 0) {
		Mat frame, flow;
		Mat prevGray, currGray;
		cvtColor(frames.back(), prevGray, CV_BGR2GRAY);
		for (uint i=0; i<batchSize; i++) {
			capture>>frame;
			if (frame.empty()) {
				break;
			}
			//Utils::scaleFramePerserveAR(frame);
			cvtColor(frame, currGray, CV_BGR2GRAY);
			
			Utils::calculateOpticalFlow(prevGray, currGray, flow);
			frames.push_back(frame.clone());
			flows.push_back(flow.clone());

			std::swap(prevGray, currGray);
		}

		while(frames.size() != std::min(batchSize + temporalWindow, frameCount + temporalWindow)+1) {
			frames.push_back(frames.back().clone());
			flows.push_back(Mat::zeros(flows.back().size(), flows.back().type()));
		}
	}
	else {
		frames.clear();
	}

	return resultFrames;	
}

np::ndarray LazyLoader::nextBatchFlows() {
	np::ndarray resultFlows = Utils::convertOpticalFlowsToNPArray(flows, batchToLoad, temporalWindow, 20.0f);
	
	if (frameCount > 0) {
		for (uint i=0; i<batchSize; i++) {
			flows.pop_front();
		}		
	}
	else {
		flows.clear();
	}
	/*p::tuple shape = p::make_tuple(1, 1, 1, 1, 3);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray resultFlows = np::zeros(shape, dtype);*/
	return resultFlows;
}

