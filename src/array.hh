#ifndef ARRAY_HH
#define ARRAY_HH

#include <climits>

enum class Dtype : char {DOUBLE = 0, COMPLEX, LONG, NONE};
const Dtype default_dtype = Dtype::DOUBLE;
int dtype_converter(const PyObject *ob, Dtype *dtype);

const int max_ndim = 3;
typedef unsigned short Shape_t;

// Set NO_OVERFLOW_DANGER if the linear index of an array entry calculated for
// a valid shape can never overflow.
// KEEP THE FOLLOWING CONSISTENT WITH THE PREVIOUS DEFINITIONS.
#if ((((size_t)-1)>>1) >= 3 * USHRT_MAX)
#ifdef NDEBUG
#define NO_OVERFLOW_DANGER
#endif
#endif

struct Array {
    PyObject_VAR_HEAD
    Py_ssize_t *buffer_internal; // Can be removed for Python 3.3, see array.cc
    Dtype dtype;
    unsigned char ndim;
    Shape_t shape[max_ndim];
    long ob_item[1];
};

PyAPI_DATA(PyTypeObject) Array_type;

#define array_check_exact(op) (Py_TYPE(op) == &Array_type)

Array *array_make(PyObject *shape_obj, Dtype dtype);

#endif // !ARRAY_HH
