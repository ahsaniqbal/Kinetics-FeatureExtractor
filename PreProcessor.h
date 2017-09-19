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
namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef _PREPROCESSOR_H_
#define _PREPROCESSOR_H_

class PreProcessor {
private:
	std::vector<Mat> frames;
	std::vector<Mat> flows;

	void populateFrames(const char* video);
	void populateOpticalFlows();
	
public:
	PreProcessor() {}
	~PreProcessor() { }
	void initialize(const char* videoFile);

	np::ndarray getOpticalFlows(float bound);
	np::ndarray getFrames();
	
};
#endif