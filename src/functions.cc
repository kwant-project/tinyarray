#include <Python.h>
#include <limits>
#include "array.hh"
#include "functions.hh"

using namespace std;

static PyObject *empty(PyObject *module, PyObject *args)
{
    PyObject *shape;
    Dtype dtype = default_dtype;
    if (!PyArg_ParseTuple(args, "O|O&", &shape, dtype_converter, &dtype))
        return 0;
    return (PyObject *)array_make(shape, dtype);
}

static PyObject *zeros(PyObject *module, PyObject *args)
{
    Array *a = (Array*)empty(module, args);
    if (a)
        for (long *p = a->ob_item, *e = p + a->ob_size; p < e; ++p)
            *p = 0;
    return (PyObject *)a;
}

static PyObject *ones(PyObject *module, PyObject *args)
{
    Array *a = (Array*)empty(module, args);
    if (a)
        for (long *p = a->ob_item, *e = p + a->ob_size; p < e; ++p)
            *p = 1;
    return (PyObject *)a;
}

static PyObject *identity(PyObject *module, PyObject *args)
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
    if (n > std::numeric_limits<Shape_t>::max()) {
        PyErr_SetString(PyExc_ValueError,
                        "Requested size is too big.");
        return 0;
    }

    Array *a = PyObject_NewVar(Array, &Array_type, n * n);
    if (!a) return 0;
    a->buffer_internal = 0;
    a->ndim = 2;
    a->shape[0] = a->shape[1] = n;

    long *p = a->ob_item;
    for (int i = 1; i < n; ++i) {
        *p++ = 1;
        for (long *e = p + n; p < e; ++p)
            *p = 0;
    }
    if (n) *p = 1;
    return (PyObject *)a;
}

static Array *array_from_sequence(PyObject *top_level_seq,
                                  Dtype dtype=Dtype::NONE)
{
    assert(PySequence_Check(top_level_seq));

    int d = 0, ndim = 0;
    long shape[max_ndim];

    // seqs is the stack of sequences being processed, all returned by
    // PySequence_Fast.  ps[d] and es[d] are the begin and end of the elements
    // of seqs[d - 1].
    PyObject *seqs[max_ndim], **ps[max_ndim], **es[max_ndim];
    es[0] = 1 + (ps[0] = &top_level_seq);

    Py_ssize_t ob_size = 1;
    long *dest = 0;             // Init. only needed to suppress a warning.
    Array *result = 0;
    while (true) {
        // See http://projects.scipy.org/numpy/ticket/2199.
        const char *msg = "A sequence does not support sequence protocol - "
            "this is probably due to a bug in numpy for 0-d arrays.";
        seqs[d] = PySequence_Fast(*ps[d]++, msg);
        if (!seqs[d]) {--d; goto fail_gracefully;}
        Py_ssize_t len = PySequence_Fast_GET_SIZE(seqs[d]);
        if (result) {
            // Check that the length of current sequence agrees with the shape.
            if (len != shape[d]) {
                PyErr_SetString(PyExc_ValueError,
                                "Input has irregular shape.");
                goto fail_gracefully;
            }
        } else {
            // It's the first time we visit a sequence at this depth.
            if (len > std::numeric_limits<Shape_t>::max()) {
                PyErr_SetString(PyExc_ValueError,
                                "Source sequence too long for tinyarray.");
                goto fail_gracefully;
            }
            shape[d] = len;
            ob_size *= len;
            ++ndim;
            if (ndim > max_ndim) {
                PyErr_SetString(PyExc_ValueError, "Too many dimensions.");
                goto fail_gracefully;
            }
        }
        PyObject **p = PySequence_Fast_ITEMS(seqs[d]);
        PyObject **e = p + len;
        if (len && PySequence_Check(p[0])) {
            if (d + 1 == ndim && result) {
                PyErr_SetString(PyExc_ValueError,
                                "Input has irregular nesting depth.");
                goto fail_gracefully;
            }
            ++d;
            ps[d] = p;
            es[d] = e;
        } else {
            if (!result) {
                // Allocate and initialize the result.
                result = PyObject_NewVar(Array, &Array_type, ob_size);
                if (!result) goto fail_gracefully;
                result->buffer_internal = 0;
                result->ndim = ndim;
                for (int d = 0; d < ndim; ++d) result->shape[d] = shape[d];
                dest = result->ob_item;
            }
            // Read-in a leaf sequence.
            while (p < e) {
                long value = PyInt_AsLong(*p++);
                if (value == -1 and PyErr_Occurred()) goto fail_gracefully;
                *dest++ = value;
            }
            Py_DECREF(seqs[d]);

            while (ps[d] == es[d]) {
                if (d == 0) {
                    // Success!
                    assert(result->ob_item + ob_size == dest);
                    return result;
                }
                --d;
                Py_DECREF(seqs[d]);
            }
            if (!PySequence_Check(*ps[d])) {
                --d;
                PyErr_SetString(PyExc_ValueError,
                                "Input has irregular nesting depth.");
                goto fail_gracefully;
            }
        }
    }

fail_gracefully:
    if (result) Py_DECREF(result);
    for (; d >= 0; --d) Py_DECREF(seqs[d]);
    return 0;
}

static Array *array_from_anything(PyObject *src, Dtype dtype=Dtype::NONE)
{
    if (PySequence_Check(src)) {
        return array_from_sequence(src, Dtype::NONE);
    } else {
        // `src' is no sequence, so it must be a number.
        long value = PyInt_AsLong(src);
        if (value == -1 and PyErr_Occurred()) return 0;
        Array *result = PyObject_NewVar(Array, &Array_type, 1);
        if (!result) return 0;
        result->buffer_internal = 0;
        result->ndim = 0;
        result->ob_item[0] = value;
        return result;
    }
}

static PyObject *array(PyObject *module, PyObject *args)
{
    PyObject *src;
    Dtype dtype = Dtype::NONE;
    if (!PyArg_ParseTuple(args, "O|O&", &src, dtype_converter, &dtype))
        return 0;
    return (PyObject*)array_from_anything(src, dtype);
}

static PyObject *scalar_product(Array *a, Array *b)
{
    assert(a->ndim == 1);
    assert(b->ndim == 1);
    Py_ssize_t n = a->shape[0];
    if (n != b->shape[0]) {
        PyErr_SetString(PyExc_ValueError,
                        "`a` and `b` must have same length");
        return 0;
    }
    long *a_data = a->ob_item, *b_data = b->ob_item;
    long result = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        result += a_data[i] * b_data[i];
    }
    return PyInt_FromLong(result);
}

// This routine is not heavily optimized.  It's performance has been measured
// to be adequate, given that it will be called from Python.  The actual
// calculation of the matrix product typically uses less than half of the
// execution time of tinyarray.dot for two 3 by 3 matrices.
static PyObject *matrix_product(Array *a, Array *b)
{
    const int a_ndim = a->ndim, b_ndim = b->ndim;
    assert(a_ndim > 0);
    assert(b_ndim > 0);
    const int ndim = a_ndim + b_ndim - 2;
    assert(ndim > 0);
    if (ndim > max_ndim) {
        PyErr_SetString(PyExc_ValueError,
                        "Result would have too many dimensions.");
        return 0;
    }
    const int n = a->shape[a_ndim - 1];
    int shape[ndim];

    int d = 0, a0 = 1;
    for (int id = 0, e = a_ndim - 1; id < e; ++id)
        a0 *= shape[d++] = a->shape[id];
    Py_ssize_t ob_size = a0;
    int b0 = 1;
    for (int id = 0, e = b_ndim - 2; id < e; ++id)
        b0 *= shape[d++] = b->shape[id];
    ob_size *= b0;
    int b1, n2;
    if (b_ndim == 1) {
        n2 = b->shape[0];
        b1 = 1;
    } else {
        n2 = b->shape[b->ndim - 2];
        ob_size *= b1 = shape[d++] = b->shape[b->ndim - 1];
    }
    if (n2 != n) {
        PyErr_SetString(PyExc_ValueError, "Matrices are not aligned.");
        return 0;
    }

    Array *result = PyObject_NewVar(Array, &Array_type, ob_size);
    if (!result) return 0;
    result->buffer_internal = 0;
    result->ndim = ndim;
    for (d = 0; d < ndim; ++d) result->shape[d] = shape[d];

    long *dest = result->ob_item;
    const long *src_a = a->ob_item;
    for (int i = 0; i < a0; ++i, src_a += n) {
        const long *src_b = b->ob_item;
        for (int j = 0; j < b0; ++j, src_b += (n - 1) * b1) {
            for (int k = 0; k < b1; ++k, ++src_b) {
                long sum = 0;
                for (int l = 0; l < n; ++l)
                    sum += src_a[l] * src_b[l * b1];
                *dest++ = sum;
            }
        }
    }

    return (PyObject*)result;
}

static PyObject *dot(PyObject *module, PyObject *args)
{
    Array *a, *b;
    bool free_a = false, free_b = false;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return 0;
    if (!array_check_exact(a)) {
        a = array_from_anything((PyObject*)a);
        if (!a) return 0;
        free_a = true;
    }
    if (!array_check_exact(b)) {
        b = array_from_anything((PyObject*)b);
        if (!b) return 0;
        free_b = true;
    }
    if (a->ndim == 0 || b->ndim == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "dot does not support zero-dimensional arrays yet.");
        return 0;
    }

    PyObject *result;
    if (a->ndim == 1 && b->ndim == 1)
        result = scalar_product(a, b);
    else
        result = matrix_product(a, b);

    if (free_a) Py_DECREF(a);
    if (free_b) Py_DECREF(b);
    return result;
}

PyMethodDef functions[] = {
    {"zeros", zeros, METH_VARARGS},
    {"ones", ones, METH_VARARGS},
    {"identity", identity, METH_VARARGS},
    {"array", array, METH_VARARGS},
    {"dot", dot, METH_VARARGS},
    {0, 0, 0, 0}                // Sentinel
};
