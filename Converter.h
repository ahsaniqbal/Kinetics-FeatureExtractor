#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/video/tracking.hpp>
#include <Python.h>
#include "numpy/ndarrayobject.h"

using namespace cv;

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator: public MatAllocator 
{
public:
	NumpyAllocator() {}
	~NumpyAllocator() {}

	void allocate(int dims, const int* sizes, int type, int*& refcount, uchar*&datastart, uchar*& data, size_t* step); 
	void deallocate(int *refcount, uchar* a, uchar* b);
};

class PyEnsureGIL
{
public:
	PyEnsureGIL() : _state(PyGILState_Ensure()) {}
	~PyEnsureGIL() {
		PyGILState_Release(_state);
	}
private:
	PyGILState_STATE _state;
};

class Converter
{
private:
	void init();
public:
	Converter();
	PyObject* toNumpy(const Mat& mat);
};
