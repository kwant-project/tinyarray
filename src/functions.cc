// Copyright 2012-2013 Tinyarray authors.
//
// This file is part of Tinyarray.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://git.kwant-project.org/tinyarray/about/LICENSE.  A list of Tinyarray
// authors can be found in the README file at the top-level directory of this
// distribution and at http://git.kwant-project.org/tinyarray/about/.

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
        *dtype = LONG;
    } else if (ob == (PyObject *)(&PyFloat_Type)) {
        *dtype = DOUBLE;
    } else if (ob == (PyObject *)(&PyComplex_Type)) {
        *dtype = COMPLEX;
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid dtype.");
        return 0;
    }
    return 1;
}

template <typename T>
PyObject *reconstruct(int ndim, const size_t *shape,
                      const void *src_data_, unsigned size_in_bytes)
{
    const T *src_data = (const T*)src_data_;
    size_t size;
    Array<T> *result = Array<T>::make(ndim, shape, &size);
    if (!result) return 0;
    if (size * sizeof(T) != size_in_bytes) {
        PyErr_SetString(PyExc_ValueError,
                        "Data length mismatch during tinyarray unpickling.");
        return 0;
    }
    T *data = result->data();
    for (size_t i = 0; i < size; ++i) data[i] = src_data[i];
    return (PyObject*)result;
}

PyObject *(*reconstruct_dtable[])(int, const size_t*, const void*, unsigned) =
    DTYPE_DISPATCH(reconstruct);

PyObject *reconstruct(PyObject *, PyObject *args)
{
    PyObject *pyshape;
    Format format;
    const void *data;
    unsigned size_in_bytes;
    if (!PyArg_ParseTuple(args, "Ois#", &pyshape, &format,
                          &data, &size_in_bytes))
        return 0;

    Dtype dtype = Dtype(0);
    while (true) {
        if (format_by_dtype[int(dtype)] == format) break;
        dtype = Dtype(int(dtype) + 1);
        if (dtype == NONE) {
            if (format < 0 || format > UNKNOWN)
                format = UNKNOWN;
            PyErr_Format(PyExc_TypeError, "Cannot unpickle %s.",
                         format_names[format]);
            return 0;
        }
    }

    unsigned long shape_as_ulong[max_ndim];
    int ndim = load_index_seq_as_ulong(pyshape, shape_as_ulong, max_ndim,
                                       "Negative dimensions are not allowed.");
    if (ndim == -1) return 0;

    size_t shape[max_ndim];
    for (int d = 0; d < ndim; ++d) shape[d] = shape_as_ulong[d];
    return reconstruct_dtable[int(dtype)](ndim, shape, data, size_in_bytes);
}

template <typename T>
PyObject *filled(int ndim, const size_t *shape, int value)
{
    size_t size;
    Array<T> *result = Array<T>::make(ndim, shape, &size);
    if (!result) return 0;
    T *data = result->data();
    for (size_t i = 0; i < size; ++i) data[i] = value;
    return (PyObject*)result;
}

PyObject *(*filled_dtable[])(int, const size_t*, int) =
    DTYPE_DISPATCH(filled);

PyObject *filled(PyObject *args, int value)
{
    PyObject *pyshape;
    Dtype dtype = default_dtype;
    if (!PyArg_ParseTuple(args, "O|O&", &pyshape, dtype_converter, &dtype))
        return 0;

    unsigned long shape_as_ulong[max_ndim];
    int ndim = load_index_seq_as_ulong(pyshape, shape_as_ulong, max_ndim,
                                       "Negative dimensions are not allowed.");
    if (ndim == -1) return 0;

    size_t shape[max_ndim];
    for (int d = 0; d < ndim; ++d) shape[d] = shape_as_ulong[d];
    return filled_dtable[int(dtype)](ndim, shape, value);
}

PyObject *zeros(PyObject *, PyObject *args)
{
    return filled(args, 0);
}

PyDoc_STRVAR(zeros_doc,
"zeros(shape, dtype="DEFAULT_DTYPE")\n\n\
Return an array of given shape and type, filled with zeros.");

PyObject *ones(PyObject *, PyObject *args)
{
    return filled(args, 1);
}

PyDoc_STRVAR(ones_doc,
"ones(shape, dtype="DEFAULT_DTYPE")\n\n\
Return an array of given shape and type, filled with ones.");

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

PyDoc_STRVAR(identity_doc,
"identity(n, dtype="DEFAULT_DTYPE")\n\n\
Return an identity matrix of given size and dtype.");

PyObject *array(PyObject *, PyObject *args)
{
    PyObject *src;
    Dtype dtype = NONE;
    if (!PyArg_ParseTuple(args, "O|O&", &src, dtype_converter, &dtype))
        return 0;
    return array_from_arraylike(src, &dtype);
}

PyDoc_STRVAR(array_doc,
"array(object, [dtype])\n\n\
Create an array from something array-like.\n\
Valid inputs are numbers, sequences (of sequences, ...) of numbers, NumPy\n\
and tinyarray arrays, and objects supporting the buffer protocol.");

PyObject *matrix(PyObject *, PyObject *args)
{
    PyObject *src;
    Dtype dtype = NONE;
    if (!PyArg_ParseTuple(args, "O|O&", &src, dtype_converter, &dtype))
        return 0;
    return matrix_from_arraylike(src, &dtype);
}

PyDoc_STRVAR(matrix_doc,
"matrix(object, [dtype])\n\n\
Create an 2-d array from something array-like.\n\
Valid inputs are the same as for array(), however the input is promoted to 2-d.\n\
A ``ValueError`` is raised if the input has more than 2 dimensions.");

PyObject *(*transpose_dtable[])(PyObject*, PyObject *) =
  DTYPE_DISPATCH(transpose);

PyObject *transpose(PyObject *, PyObject *args)
{
    PyObject *a;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    Dtype dtype = NONE;
    a = array_from_arraylike(a, &dtype);
    if (!a) return 0;
    return transpose_dtable[int(dtype)](a, 0);
}

PyDoc_STRVAR(transpose_doc,
"transpose(a)\n\n\
Return a copy of the given array with reversed order of dimensions.");

PyObject *dot(PyObject *, PyObject *args)
{
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b)) return 0;
    return dot_product(a, b);
}

PyDoc_STRVAR(dot_doc,
"dot(a, b)\n\n\
Return the dot product of two arrays.\n\n\
For 2-d arrays, the dot product is equivalent to matrix multiplication; for 1-d\n\
arrays, to a scalar product of vectors.  In the general case, it is equivalent\n\
to a sum over the last axis of a and the second-to-last of b, e.g.::\n\n\
    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])");

template <template <typename> class Op>
PyObject *binary_ufunc(PyObject *, PyObject *args)
{
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b)) return 0;
    return Binary_op<Op>::apply(a, b);
}

template <template <typename> class Op>
PyObject *unary_ufunc(PyObject *, PyObject *args)
{
    static PyObject *(*operation_dtable[])(PyObject*) = {
        apply_unary_ufunc<Op<long> >,
        apply_unary_ufunc<Op<double> >,
        apply_unary_ufunc<Op<Complex> >
    };

    PyObject *a;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    Dtype dtype = NONE;
    a = array_from_arraylike(a, &dtype);
    if (!a) return 0;
    PyObject *result = operation_dtable[int(dtype)](a);
    Py_DECREF(a);
    return result;
}

template <typename Kind>
PyObject *unary_ufunc_round(PyObject *, PyObject *args)
{
    static PyObject *(*operation_dtable[])(PyObject*) = {
        apply_unary_ufunc<Round<Kind, long> >,
        apply_unary_ufunc<Round<Kind, double> >,
        apply_unary_ufunc<Round<Kind, Complex> >
    };

    PyObject *a;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    Dtype dtype = NONE;
    a = array_from_arraylike(a, &dtype);
    if (!a) return 0;
    PyObject *result = operation_dtable[int(dtype)](a);
    Py_DECREF(a);
    return result;
}

PyDoc_STRVAR(binary_ufunc_doc,
"Operates elementwise on two arrays, returns an array of the same shape.");

PyDoc_STRVAR(unary_ufunc_doc,
"Operates elementwise on an array, returns an array of the same shape.");

} // Anonymous namespace

PyMethodDef functions[] = {
    {"_reconstruct", reconstruct, METH_VARARGS},
    {"zeros", zeros, METH_VARARGS, zeros_doc},
    {"ones", ones, METH_VARARGS, ones_doc},
    {"identity", identity, METH_VARARGS, identity_doc},
    {"array", array, METH_VARARGS, array_doc},
    {"matrix", matrix, METH_VARARGS, matrix_doc},
    {"transpose", transpose, METH_VARARGS, transpose_doc},
    {"dot", dot, METH_VARARGS, dot_doc},

    {"add", binary_ufunc<Add>, METH_VARARGS, binary_ufunc_doc},
    {"subtract", binary_ufunc<Subtract>, METH_VARARGS, binary_ufunc_doc},
    {"multiply", binary_ufunc<Multiply>, METH_VARARGS, binary_ufunc_doc},
    {"divide", binary_ufunc<Divide>, METH_VARARGS, binary_ufunc_doc},
    {"remainder", binary_ufunc<Remainder>, METH_VARARGS, binary_ufunc_doc},
    {"floor_divide", binary_ufunc<Floor_divide>, METH_VARARGS,
     binary_ufunc_doc},

    {"negative", unary_ufunc<Negative>, METH_VARARGS, unary_ufunc_doc},
    {"abs", unary_ufunc<Absolute>, METH_VARARGS, unary_ufunc_doc},
    {"absolute", unary_ufunc<Absolute>, METH_VARARGS, unary_ufunc_doc},
    {"conjugate", unary_ufunc<Conjugate>, METH_VARARGS, unary_ufunc_doc},
    {"round", unary_ufunc_round<Nearest>, METH_VARARGS, unary_ufunc_doc},
    {"floor", unary_ufunc_round<Floor>, METH_VARARGS, unary_ufunc_doc},
    {"ceil", unary_ufunc_round<Ceil>, METH_VARARGS, unary_ufunc_doc},

    {0, 0, 0, 0}                // Sentinel
};
