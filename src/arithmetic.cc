#include <Python.h>
#include <algorithm>
#include <complex>
#include "array.hh"
#include "arithmetic.hh"
#include "conversion.hh"

template <typename T>
PyObject *array_scalar_product(PyObject *a_, PyObject *b_)
{
    assert(Array<T>::check_exact(a_)); Array<T> *a = (Array<T>*)a_;
    assert(Array<T>::check_exact(b_)); Array<T> *b = (Array<T>*)b_;
    int ndim_a, ndim_b;
    size_t *shape_a, *shape_b;
    a->ndim_shape(&ndim_a, &shape_a);
    b->ndim_shape(&ndim_b, &shape_b);
    assert(ndim_a == 1);
    assert(ndim_b == 1);
    size_t n = shape_a[0];
    if (n != shape_b[0]) {
        PyErr_SetString(PyExc_ValueError,
                        "Both arguments must have same length.");
        return 0;
    }
    T *data_a = a->data(), *data_b = b->data();
    // It's important not to start with result = 0.  This leads to wrong
    // results with regard to the sign of zero as 0.0 + -0.0 is 0.0.
    assert(n > 0);
    T result = data_a[0] * data_b[0];
    for (size_t i = 1; i < n; ++i) {
        result += data_a[i] * data_b[i];
    }
    return pyobject_from_number(result);
}

// This routine is not heavily optimized.  It's performance has been measured
// to be adequate, given that it will be called from Python.  The actual
// calculation of the matrix product typically uses less than half of the
// execution time of tinyarray.dot for two 3 by 3 matrices.
template <typename T>
PyObject *array_matrix_product(PyObject *a_, PyObject *b_)
{
    assert(Array<T>::check_exact(a_)); Array<T> *a = (Array<T>*)a_;
    assert(Array<T>::check_exact(b_)); Array<T> *b = (Array<T>*)b_;
    int ndim_a, ndim_b;
    size_t *shape_a, *shape_b;
    a->ndim_shape(&ndim_a, &shape_a);
    b->ndim_shape(&ndim_b, &shape_b);
    assert(ndim_a > 0);
    assert(ndim_b > 0);
    int ndim = ndim_a + ndim_b - 2;
    assert(ndim > 0);
    if (ndim > max_ndim) {
        PyErr_SetString(PyExc_ValueError,
                        "Result would have too many dimensions.");
        return 0;
    }
    const size_t n = shape_a[ndim_a - 1];
    size_t shape[ndim];

    size_t d = 0, a0 = 1;
    for (int id = 0, e = ndim_a - 1; id < e; ++id)
        a0 *= shape[d++] = shape_a[id];
    size_t b0 = 1;
    for (int id = 0, e = ndim_b - 2; id < e; ++id)
        b0 *= shape[d++] = shape_b[id];
    size_t b1, n2;
    if (ndim_b == 1) {
        n2 = shape_b[0];
        b1 = 1;
    } else {
        n2 = shape_b[ndim_b - 2];
        b1 = shape[d++] = shape_b[ndim_b - 1];
    }
    if (n2 != n) {
        PyErr_SetString(PyExc_ValueError, "Matrices are not aligned.");
        return 0;
    }

    Array<T> *result = Array<T>::make(ndim, shape);
    if (!result) return 0;

    const T *data_a = a->data(), *data_b = b->data();
    T *dest = result->data();
    const T *src_a = data_a;
    for (size_t i = 0; i < a0; ++i, src_a += n) {
        const T *src_b = data_b;
        for (size_t j = 0; j < b0; ++j, src_b += (n - 1) * b1) {
            for (size_t k = 0; k < b1; ++k, ++src_b) {
                // It's important not to start with sum = 0.  This leads to
                // wrong results with regard to the sign of zero as 0.0 + -0.0
                // is 0.0.
                assert(n > 0);
                T sum = src_a[0] * src_b[0];
                for (size_t l = 1; l < n; ++l)
                    sum += src_a[l] * src_b[l * b1];
                *dest++ = sum;
            }
        }
    }

    return (PyObject*)result;
}

PyNumberMethods as_number = {
    (binaryfunc)0/*add*/,            // nb_add
    (binaryfunc)0,              // nb_subtract
    (binaryfunc)0,              // nb_multiply
    (binaryfunc)0,              // nb_divide
    (binaryfunc)0,              // nb_remainder
    (binaryfunc)0,              // nb_divmod
    (ternaryfunc)0,             // nb_power
    (unaryfunc)0,               // nb_negative
    (unaryfunc)0,               // nb_positive
    (unaryfunc)0,               // nb_absolute
    (inquiry)0,                 // nb_nonzero
    (unaryfunc)0,               // nb_invert
    (binaryfunc)0,              // nb_lshift
    (binaryfunc)0,              // nb_rshift
    (binaryfunc)0,              // nb_and
    (binaryfunc)0,              // nb_xor
    (binaryfunc)0,              // nb_or
    (coercion)0,                // nb_coerce
    (unaryfunc)0,               // nb_int
    (unaryfunc)0,               // nb_long
    (unaryfunc)0,               // nb_float
    (unaryfunc)0,               // nb_oct
    (unaryfunc)0,               // nb_hex

    (binaryfunc)0,              // nb_inplace_add
    (binaryfunc)0,              // nb_inplace_subtract
    (binaryfunc)0,              // nb_inplace_multiply
    (binaryfunc)0,              // nb_inplace_divide
    (binaryfunc)0,              // nb_inplace_remainder
    (ternaryfunc)0,             // nb_inplace_power
    (binaryfunc)0,              // nb_inplace_lshift
    (binaryfunc)0,              // nb_inplace_rshift
    (binaryfunc)0,              // nb_inplace_and
    (binaryfunc)0,              // nb_inplace_xor
    (binaryfunc)0,              // nb_inplace_or

    (binaryfunc)0,              // nb_floor_divide
    (binaryfunc)0,              // nb_true_divide
    (binaryfunc)0,              // nb_inplace_floor_divide
    (binaryfunc)0,              // nb_inplace_true_divide

    (unaryfunc)0                // nb_index
};

// Explicit instantiations.
template
PyObject *array_scalar_product<long>(PyObject*, PyObject*);
template
PyObject *array_scalar_product<double>(PyObject*, PyObject*);
template
PyObject *array_scalar_product<Complex>(PyObject*, PyObject*);
template
PyObject *array_matrix_product<long>(PyObject*, PyObject*);
template
PyObject *array_matrix_product<double>(PyObject*, PyObject*);
template
PyObject *array_matrix_product<Complex>(PyObject*, PyObject*);
