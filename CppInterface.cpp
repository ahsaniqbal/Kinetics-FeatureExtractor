#include "ActiveLoader.h"
#include "LazyLoader.h"

BOOST_PYTHON_MODULE(libCppInterface) {
	class_<ActiveLoader>("ActiveLoader")
		.def("initialize", &ActiveLoader::initialize)
		.def("getOpticalFlows", &ActiveLoader::getOpticalFlows)
		.def("getFrames", &ActiveLoader::getFrames);

	class_<LazyLoader>("LazyLoader")
		.def("initializeLazy", &LazyLoader::initializeLazy)
		.def("hasNextBatch", &LazyLoader::hasNextBatch)
		.def("nextBatchFrames", &LazyLoader::nextBatchFrames)
		.def("nextBatchFlows", &LazyLoader::nextBatchFlows);
}