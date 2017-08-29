#include "Converter.h"
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace cv::gpu;

typedef std::vector<float> FloatVec;
typedef std::vector<FloatVec > VecOfFloatVec;
typedef unsigned int uint;

#define CAST(v, L, H) ((v) >= (H) ? (H) : (v) <= (L) ? (L) : (v))


struct RGBTo01Transformer {
	float operator()(float &elem) const {
		return (elem - 127.5f) / 127.5f;
	}
};
struct OpticalFlowTransformer {
private:
	float bound;
public:
	OpticalFlowTransformer(float _bound): bound(_bound) {}

	float operator()(float &elem) const {
		return ((elem >= bound ? bound : elem <= -bound ? -bound : elem) / bound);
	}
};
struct ROIExtractor {
private:
	uint destWidth, destHeight, x, y;
public:
	ROIExtractor(uint _destWidth, uint _destHeight, uint _sourceWidth, uint _sourceHeight): destWidth(_destWidth), destHeight(_destHeight) {
		x = (_sourceWidth / 2) - (destWidth / 2);
		y = (_sourceHeight / 2) - (destHeight / 2);
	}

	Mat operator()(Mat & elem) const {
		return elem(Rect(x, y, destWidth, destHeight));
	}
};
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





void convertFlows(const std::vector<Mat>& input, std::vector<float>& result, float bound) {
	uint rows = 224;
	uint cols = 224;
	Rect rec((input.at(0).cols / 2) - (cols / 2), (input.at(0).rows / 2) - (rows / 2), cols, rows);
	result[0] = input.size()/2;
	for (uint i=0; i<input.size(); i+=2) {
		Mat roiU = (input.at(i))(rec);
		Mat roiV = (input.at(i+1))(rec);
		for (uint j=0; j<rows; j++) {
			float* rowU = roiU.ptr<float>(j);
			float* rowV = roiV.ptr<float>(j);
			for (uint k=0; k<cols; k++) {
				result[1 + (i * cols * rows) + (j * cols * 2) + k * 2 + 0] = CAST(rowU[k], -1*bound, bound) / bound;
				result[1 + (i * cols * rows) + (j * cols * 2) + k * 2 + 1] = CAST(rowV[k], -1*bound, bound) / bound;
			}	
		}
	}
}
void convertFrames(std::vector<Mat>& input, std::vector<float>& result) {
	uint height = 224;
	uint width = 224;
	Rect rec((input.at(0).cols / 2) - (width / 2), (input.at(0).rows / 2) - (height / 2), width, height);
	result[0] = input.size() - 1;

	for (uint i=0; i<input.size() - 1; i++) {
		Mat roi = (input.at(i))(rec);
		for (uint j=0; j<(uint)roi.rows; j++) {
			Vec3b *row = roi.ptr<Vec3b>(j);
			for (uint k=0; k<(uint)roi.cols; k++) {
				uint index = 1 + (i * 3 * roi.cols * roi.rows) + (j * 3 * roi.cols) + k * 3;
				//copy BGR mat to vector as RGB
				result[index + 0] = ((float)row[k].val[2] - 127.5f) / 127.5f;
				result[index + 1] = ((float)row[k].val[1] - 127.5f) / 127.5f;
				result[index + 2] = ((float)row[k].val[0] - 127.5f) / 127.5f;
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
	resize(frame, frame, Size(round(scaleFactor * frame.cols), round(scaleFactor * frame.rows)));
}

//TODO:: check video file name is not empty
void getFrames(const char* video, int step, std::vector<Mat>& resultFrames) {
	VideoCapture capture(video);
	Mat frame;
	if (!capture.isOpened()) {
		std::cout<<"Unable to open the video";
		return;
	}

	int frameNum = 0;
	while(true) {
		capture>>frame;
		if (frame.empty()) {
			break;
		}
		capture.set(CV_CAP_PROP_POS_FRAMES, frameNum + step);
		frameNum += step;
		scaleFramePreserveAR(frame);
		resultFrames.push_back(frame);
	}
}
void calculateOpFlow(const std::vector<Mat>& frames, std::vector<Mat>& flows) {
	if (frames.size() == 0) {
		return;
	}

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

VecOfFloatVec getOpticalFlow(const char* video, int step, int bound) {
	VecOfFloatVec result;
	if (video != NULL ) {
		
		std::vector<Mat> frames, flows;
		getFrames(video, step, frames);
		calculateOpFlow(frames, flows);
		
		if (frames.size() > 0 && flows.size() > 0) {
			std::vector<float> resultFrames(1 + (frames.size() - 1) * 224 * 224 * 3, 0);		
			std::vector<float> resultFlows(1 + flows.size() * 224 * 224, 0);
			convertFlows(flows, resultFlows, bound);
			convertFrames(frames, resultFrames);

			result.push_back(resultFrames);
			result.push_back(resultFlows);
		}
	}

	return result;	
}


BOOST_PYTHON_MODULE(libpyOpFlow) {
	using namespace boost::python;

	class_<FloatVec>("FloatVec")
		.def(vector_indexing_suite<FloatVec>());
	class_<VecOfFloatVec>("VecOfFloatVec")
		.def(vector_indexing_suite<VecOfFloatVec>());

	def("getOpticalFlow", getOpticalFlow);	
}
