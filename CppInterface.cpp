#include "PreProcessor.h"
#include "LazyLoader.h"

BOOST_PYTHON_MODULE(libCppInterface) {
	class_<PreProcessor>("PreProcessor")
		.def("initialize", &PreProcessor::initialize)
		.def("getOpticalFlows", &PreProcessor::getOpticalFlows)
		.def("getFrames", &PreProcessor::getFrames);

	class_<LazyLoader>("LazyLoader")
		.def("initializeLazy", &LazyLoader::initializeLazy)
		.def("hasNextBatch", &LazyLoader::hasNextBatch)
		.def("nextBatchFrames", &LazyLoader::nextBatchFrames)
		.def("nextBatchFlows", &LazyLoader::nextBatchFlows);
}