#include "Converter.h"

void NumpyAllocator::allocate(int dims, const int* sizes, int type, int*& refcount, uchar*&datastart, uchar*& data, size_t* step) {
	PyEnsureGIL gil;
	int depth = CV_MAT_DEPTH(type);
	int channels = CV_MAT_CN(type);

	int typeNum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
			depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
			depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT : NPY_DOUBLE;
			
	
	npy_intp _sizes[CV_MAX_DIM+1];
	for (int i=0; i<dims; i++) {
		_sizes[i] = sizes[i];
	}
	
	if (channels > 1) {
		_sizes[dims++] = channels;	
	}

	PyObject* npArray = PyArray_SimpleNew(dims, _sizes, typeNum);
	if (!npArray) {
		CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typeNum, dims));
	}

	refcount = refcountFromPyObject(npArray);
	npy_intp* _strides = PyArray_STRIDES(npArray);

	for (int i=0; i<dims - (channels>1); i++) {
		step[i] = (size_t)_strides[i];	
	}
	datastart = data = (uchar*)PyArray_DATA(npArray);
}
void NumpyAllocator::deallocate(int *refcount, uchar* a, uchar* b) {
	PyEnsureGIL gil;
	if (!refcount) {
		return;	
	}
	PyObject* obj = pyObjectFromRefcount(refcount);
	Py_INCREF(obj);
	Py_DECREF(obj);
}

NumpyAllocator g_numpyAllocator;

Converter::Converter() {
	init();
}
void Converter::init() {
	import_array();
}
PyObject* Converter::toNumpy(const Mat& mat) {
	if (!mat.data)
		Py_RETURN_NONE;
	Mat temp, *p = (Mat*)&mat;

	if (!p->refcount || p->allocator != &g_numpyAllocator) {
		temp.allocator = &g_numpyAllocator;
		mat.copyTo(temp);
		p = &temp;
	}
	p->addref();
	return pyObjectFromRefcount(p->refcount);

}
