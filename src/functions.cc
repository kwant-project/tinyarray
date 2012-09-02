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

PyObject *(*transpose_dtable[])(PyObject*) = DTYPE_DISPATCH(transpose);

PyObject *transpose(PyObject *, PyObject *args)
{
    PyObject *a;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    Dtype dtype = Dtype::NONE;
    a = array_from_arraylike(a, &dtype);
    if (!a) return 0;
    return transpose_dtable[int(dtype)](a);
}

PyObject *dot(PyObject *, PyObject *args)
{
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b)) return 0;
    return dot_product(a, b);
}

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
    static_assert(int(Dtype::LONG) == 0 && int(Dtype::DOUBLE) == 1 &&
                  int(Dtype::COMPLEX) == 2 && int(Dtype::NONE) == 3,
                  "Update me.");
    static PyObject *(*operation_dtable[])(PyObject*) = {
        apply_unary_ufunc<Op<long> >,
        apply_unary_ufunc<Op<double> >,
        apply_unary_ufunc<Op<Complex> >
    };

    PyObject *a;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    Dtype dtype = Dtype::NONE;
    a = array_from_arraylike(a, &dtype);
    if (!a) return 0;
    PyObject *result = operation_dtable[int(dtype)](a);
    Py_DECREF(a);
    return result;
}

template <typename Kind>
PyObject *unary_ufunc_round(PyObject *, PyObject *args)
{
    static_assert(int(Dtype::LONG) == 0 && int(Dtype::DOUBLE) == 1 &&
                  int(Dtype::COMPLEX) == 2 && int(Dtype::NONE) == 3,
                  "Update me.");
    static PyObject *(*operation_dtable[])(PyObject*) = {
        apply_unary_ufunc<Round<Kind, long> >,
        apply_unary_ufunc<Round<Kind, double> >,
        apply_unary_ufunc<Round<Kind, Complex> >
    };

    PyObject *a;
    if (!PyArg_ParseTuple(args, "O", &a)) return 0;
    Dtype dtype = Dtype::NONE;
    a = array_from_arraylike(a, &dtype);
    if (!a) return 0;
    PyObject *result = operation_dtable[int(dtype)](a);
    Py_DECREF(a);
    return result;
}

} // Anonymous namespace

// TODO: once we can require gcc 4.7, use the following type aliases instead of
// the function unary_ufunc_round which is a stopgap.
//
// template <typename T> using Round_nearest = Round<Nearest, T>;
// template <typename T> using Round_floor = Round<Floor, T>;
// template <typename T> using Round_ceil = Round<Ceil, T>;

PyMethodDef functions[] = {
    {"zeros", zeros, METH_VARARGS},
    {"ones", ones, METH_VARARGS},
    {"identity", identity, METH_VARARGS},
    {"array", array, METH_VARARGS},
    {"transpose", transpose, METH_VARARGS},
    {"dot", dot, METH_VARARGS},

    {"add", binary_ufunc<Add>, METH_VARARGS},
    {"subtract", binary_ufunc<Subtract>, METH_VARARGS},
    {"multiply", binary_ufunc<Multiply>, METH_VARARGS},
    {"divide", binary_ufunc<Divide>, METH_VARARGS},
    {"remainder", binary_ufunc<Remainder>, METH_VARARGS},
    {"floor_divide", binary_ufunc<Floor_divide>, METH_VARARGS},

    {"negative", unary_ufunc<Negative>, METH_VARARGS},
    {"abs", unary_ufunc<Absolute>, METH_VARARGS},
    {"absolute", unary_ufunc<Absolute>, METH_VARARGS},
    {"conjugate", unary_ufunc<Conjugate>, METH_VARARGS},
// {"round", unary_ufunc<Round_nearest>, METH_VARARGS},
// {"floor", unary_ufunc<Round_floor>, METH_VARARGS},
// {"ceil", unary_ufunc<Round_ceil>, METH_VARARGS},
    {"round", unary_ufunc_round<Nearest>, METH_VARARGS},
    {"floor", unary_ufunc_round<Floor>, METH_VARARGS},
    {"ceil", unary_ufunc_round<Ceil>, METH_VARARGS},

    {0, 0, 0, 0}                // Sentinel
};
