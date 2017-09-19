#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <Python.h>
#include <vector>
#include "Utils.h"
using namespace std;
using namespace cv;
using namespace cv::gpu;


#ifndef _LAZYLOADER_H_
#define _LAZYLOADER_H_

class LazyLoader {
	VideoCapture capture;
	std::list<Mat> frames;
	std::list<Mat> flows;
	uint batchSize;
	uint temporalWindow;
	
	uint frameCount;

	void initFramesLazy(const uint batchSize, const uint temporalWindow);
	void initFlowLazy();

	void createBatch(const uint batchSize, const uint temporalWindow);
	void appendFrame(Mat& mat, const uint count);
public:
	LazyLoader() : capture() {}
	~LazyLoader() { capture.release(); }

	void initializeLazy(const char* videoFile, const uint batchSize, const uint temporalWindow);
	//np::ndarray getFrames(int batchSize, int temporalWindow);
	//np::ndarray getOpticalFlows(float bound, int batchSize, int temporalWindow);
	
	bool hasNextBatch();
	np::ndarray nextBatchFrames();
	np::ndarray nextBatchFlows();
};

#endif