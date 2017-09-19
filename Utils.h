#include <iostream>
#include <iterator>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
#define CAST(v, L, H) ((v) >= (H) ? (H) : (v) <= (L) ? (L) : (v))
using namespace std;
using namespace cv;
using namespace cv::gpu;
namespace p = boost::python;
namespace np = boost::python::numpy;

typedef unsigned int uint;

#ifndef _UTILS_H_
#define _UTILS_H_

enum SmallerDimension { row, col };

class Utils {
public:
	static void calculateOpticalFlow(const Mat& previous, const Mat& current, Mat& result);
	static SmallerDimension getSmallerDimension(const Mat& frame);
	static void scaleFramePerserveAR(Mat& frame);

	static np::ndarray convertRGBFramesToNPArray(const std::list<Mat>& frames, uint batchSize, uint temporalWindow);
	static np::ndarray convertOpticalFlowsToNPArray(const std::list<Mat>& flows, uint batchSize, uint temporalWindow, float bound);
};
#endif