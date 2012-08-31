#include <Python.h>
#include "array.hh"
#include "arithmetic.hh"
#include "functions.hh"

namespace {

int dtype_converter(const PyObject *ob, Dtype *dtype)
{
    if (ob == Py_None) {
        *dtype = default_dtype;
    } else if (ob == (PyObject *)(&PyInt_Type) ||
               ob == (PyObject *)(&PyLong_Type)) {
        *dtype = Dtype::LONG;
    } else if (ob == (PyObject *)(&PyFloat_Type)) {
        *dtype = Dtype::DOUBLE;
    } else if (ob == (PyObject *)(&PyComplex_Type)) {
        *dtype = Dtype::COMPLEX;
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid dtype.");
        return 0;
    }
    return 1;
}

template <typename T>
PyObject *filled(size_t ndim, const size_t *shape, int value)
{
    size_t size;
    Array<T> *result = Array<T>::make(ndim, shape, &size);
    if (!result) return 0;
    T *data = result->data();
    for (size_t i = 0; i < size; ++i) data[i] = value;
    return (PyObject*)result;
}

PyObject *(*filled_dtable[])(size_t, const size_t*, int) =
    DTYPE_DISPATCH(filled);

PyObject *filled_pyargs(PyObject *args, int value)
{
    PyObject *pyshape;
    Dtype dtype = default_dtype;
    if (!PyArg_ParseTuple(args, "O|O&", &pyshape, dtype_converter, &dtype))
        return 0;

    size_t shape[max_ndim];
    Py_ssize_t ndim = load_index_seq_as_ulong(
        pyshape, shape, max_ndim, "Negative dimensions are not allowed.");
    if (ndim == -1) return 0;

    return filled_dtable[int(dtype)](ndim, shape, value);
}

PyObject *zeros(PyObject *, PyObject *args)
{
    return filled_pyargs(args, 0);
}

PyObject *ones(PyObject *, PyObject *args)
{
    return filled_pyargs(args, 1);
}

template <typename T>
PyObject *identity(size_t n)
{
    size_t size, shape[] = {n, n};
    Array<T> *result = Array<T>::make(2, shape, &size);
    if (!result) return 0;

    T *p = result->data();
    for (size_t i = 1; i < n; ++i) {
        *p++ = 1;
        for (T *e = p + n; p < e; ++p)
            *p = 0;
    }
    if (n) *p = 1;

    return (PyObject*)result;
}

PyObject *(*identity_dtable[])(size_t) = DTYPE_DISPATCH(identity);

PyObject *identity(PyObject *, PyObject *args)
{
    long n;
    Dtype dtype = default_dtype;
    if (!PyArg_ParseTuple(args, "l|O&", &n, dtype_converter, &dtype))
        return 0;
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "Negative dimensions are not allowed.");
        return 0;
    }

    return identity_dtable[int(dtype)](n);
}

PyObject *array(PyObject *, PyObject *args)
{
    PyObject *src;
    Dtype dtype = Dtype::NONE;
    if (!PyArg_ParseTuple(args, "O|O&", &src, dtype_converter, &dtype))
        return 0;
    return array_from_arraylike(src, &dtype);
}

PyObject *(*array_scalar_product_dtable[])(PyObject*, PyObject*) =
    DTYPE_DISPATCH(array_scalar_product);
PyObject *(*array_matrix_product_dtable[])(PyObject*, PyObject*) =
    DTYPE_DISPATCH(array_matrix_product);

PyObject *dot(PyObject *, PyObject *args)
{
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return 0;
    Dtype dtype_a = get_dtype(a), dtype_b = get_dtype(b);

    // Make sure a and b are tinyarrays.
    if (dtype_a != Dtype::NONE) {
        Py_INCREF(a);
    } else {
        a = array_from_arraylike(a, &dtype_a);
        if (!a) return 0;
    }
    if (dtype_b != Dtype::NONE) {
        Py_INCREF(b);
    } else {
        b = array_from_arraylike(b, &dtype_b);
        if (!b) {
            Py_DECREF(a);
            return 0;
        }
    }

    PyObject *result = 0;
    size_t ndim_a, ndim_b;
    reinterpret_cast<Array_base*>(a)->ndim_shape(&ndim_a, 0);
    reinterpret_cast<Array_base*>(b)->ndim_shape(&ndim_b, 0);
    if (ndim_a == 0 || ndim_b == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "dot does not support zero-dimensional arrays yet.");
        goto end;
    }
    if (dtype_a != dtype_b) {
        PyErr_SetString(PyExc_ValueError,
                        "Dtype must be the same for now.");
        goto end;
    }

    if (ndim_a == 1 && ndim_b == 1)
        result = array_scalar_product_dtable[int(dtype_a)](a, b);
    else
        result = array_matrix_product_dtable[int(dtype_a)](a, b);

end:
    Py_DECREF(a);
    Py_DECREF(b);
    return result;
}

} // Anonymous namespace

PyMethodDef functions[] = {
    {"zeros", zeros, METH_VARARGS},
    {"ones", ones, METH_VARARGS},
    {"identity", identity, METH_VARARGS},
    {"array", array, METH_VARARGS},
    {"dot", dot, METH_VARARGS},
    {0, 0, 0, 0}                // Sentinel
};
