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
#include <boost/python.hpp>
using namespace boost::python;
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
	uint batchToLoad;
	bool isOnlyForRGB;

	void initFramesLazy();
	void initFlowLazy();

	void createBatch();
	void appendFrame(Mat& mat, const uint count);
public:
	LazyLoader() : capture() { this->isOnlyForRGB = false; }
	~LazyLoader() { capture.release(); }

	void initializeLazy(const char* videoFile, const uint batchSize, const uint temporalWindow, bool isOnlyRGB);
	
	bool hasNextBatch();
	np::ndarray nextBatchFrames();
	np::ndarray nextBatchFlows();
};

#endif
