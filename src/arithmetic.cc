// Copyright 2012-2013 Tinyarray authors.
//
// This file is part of Tinyarray.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://kwant-project.org/tinyarray/license.  A list of Tinyarray authors can
// be found in the README file at the top-level directory of this distribution
// and at http://kwant-project.org/tinyarray.

#include <Python.h>
#include <limits>
#include <cmath>
#include <sstream>
#include <functional>
#include <algorithm>
#include "array.hh"
#include "arithmetic.hh"
#include "conversion.hh"

// This module assumes C99 behavior of division:
// int(-3) / int(2) == -1

namespace {

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
    if (n == 0) return pyobject_from_number(T(0));
    assert(n > 0);
    T result = data_a[0] * data_b[0];
    for (size_t i = 1; i < n; ++i) {
        result += data_a[i] * data_b[i];
    }
    return pyobject_from_number(result);
}

PyObject *(*array_scalar_product_dtable[])(PyObject*, PyObject*) =
    DTYPE_DISPATCH(array_scalar_product);

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
    size_t shape[max_ndim];

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

    size_t size;
    Array<T> *result = Array<T>::make(ndim, shape, &size);
    if (!result) return 0;

    T *dest = result->data();
    if (n == 0) {
        for (size_t i = 0; i < size; ++i) dest[i] = 0;
    } else {
        assert(n > 0);
        const T *data_a = a->data(), *data_b = b->data();
        const T *src_a = data_a;
        for (size_t i = 0; i < a0; ++i, src_a += n) {
            const T *src_b = data_b;
            for (size_t j = 0; j < b0; ++j, src_b += (n - 1) * b1) {
                for (size_t k = 0; k < b1; ++k, ++src_b) {
                    // It's important not to start with sum = 0.  This leads to
                    // wrong results with regard to the sign of zero as 0.0 +
                    // -0.0 is 0.0.
                    T sum = src_a[0] * src_b[0];
                    for (size_t l = 1; l < n; ++l)
                        sum += src_a[l] * src_b[l * b1];
                    *dest++ = sum;
                }
            }
        }
    }

    return (PyObject*)result;
}

PyObject *(*array_matrix_product_dtable[])(PyObject*, PyObject*) =
    DTYPE_DISPATCH(array_matrix_product);

PyObject *apply_binary_ufunc(Binary_ufunc **ufunc_dtable,
                             PyObject *a, PyObject *b)
{
    Dtype dtype;
    if (coerce_to_arrays(&a, &b, &dtype) < 0) return 0;

    int ndim_a, ndim_b;
    size_t *shape_a, *shape_b;
    reinterpret_cast<Array_base*>(a)->ndim_shape(&ndim_a, &shape_a);
    reinterpret_cast<Array_base*>(b)->ndim_shape(&ndim_b, &shape_b);

    PyObject *result = 0;
    int ndim = std::max(ndim_a, ndim_b);
    size_t stride_a = 1, stride_b = 1, shape[max_ndim];;
    ptrdiff_t hops_a[max_ndim], hops_b[max_ndim];
    for (int d = ndim - 1, d_a = ndim_a - 1, d_b = ndim_b - 1;
         d >= 0; --d, --d_a, --d_b) {
        size_t ext_a = d_a >= 0 ? shape_a[d_a] : 1;
        size_t ext_b = d_b >= 0 ? shape_b[d_b] : 1;

        if (ext_a == ext_b) {
            hops_a[d] = stride_a;
            hops_b[d] = stride_b;
            shape[d] = ext_a;
            stride_a *= ext_a;
            stride_b *= ext_b;
        } else if (ext_a == 1) {
            hops_a[d] = 0;
            hops_b[d] = stride_b;
            stride_b *= shape[d] = ext_b;
        } else if (ext_b == 1) {
            hops_a[d] = stride_a;
            hops_b[d] = 0;
            stride_a *= shape[d] = ext_a;
        } else {
            std::ostringstream s;
            s << "Operands could not be broadcast together with shapes (";
            for (int d = 0; d < ndim_a; ++d) {
                s << shape_a[d];
                if (d + 1 < ndim_a) s << ", ";
            }
            s << ") and (";
            for (int d = 0; d < ndim_b; ++d) {
                s << shape_b[d];
                if (d + 1 < ndim_b) s << ", ";
            }
            s << ").";
            PyErr_SetString(PyExc_ValueError, s.str().c_str());
            goto end;
        }
    }
    for (int d = 1; d < ndim; ++d)
    {
        hops_a[d - 1] -= hops_a[d] * shape[d];
        hops_b[d - 1] -= hops_b[d] * shape[d];
    }

    result = ufunc_dtable[int(dtype)](ndim, shape, a, hops_a, b, hops_b);

end:
    Py_DECREF(a);
    Py_DECREF(b);
    return result;
}

} // Anonymous namespace

template <template <typename> class Op>
template <typename T>
PyObject *Binary_op<Op>::ufunc(int ndim, const size_t *shape,
                               PyObject *a_, const ptrdiff_t *hops_a,
                               PyObject *b_, const ptrdiff_t *hops_b)
{
    Op<T> operation;

    assert(Array<T>::check_exact(a_)); Array<T> *a = (Array<T>*)a_;
    assert(Array<T>::check_exact(b_)); Array<T> *b = (Array<T>*)b_;

    T *src_a = a->data(), *src_b = b->data();

    if (ndim == 0) {
        T result;
        if (operation(result, *src_a, *src_b)) return 0;
        return (PyObject*)pyobject_from_number(result);
    }

    Array<T> *result = Array<T>::make(ndim, shape);
    if (result == 0) return 0;
    T *dest = result->data();

    int d = 0;
    size_t i[max_ndim];
    --ndim;
    i[0] = shape[0];
    while (true) {
        if (i[d]) {
            --i[d];
            if (d == ndim) {
                if (operation(*dest++, *src_a, *src_b)) {
                    Py_DECREF(result);
                    return 0;
                }
                src_a += hops_a[d];
                src_b += hops_b[d];
            } else {
                ++d;
                i[d] = shape[d];
            }
        } else {
            if (d == 0) return (PyObject*)result;
            --d;
            src_a += hops_a[d];
            src_b += hops_b[d];
        }
    }
}

template <template <typename> class Op>
PyObject *Binary_op<Op>::apply(PyObject *a, PyObject *b)
{
    return apply_binary_ufunc(dtable, a, b);
}

template <template <typename> class Op>
Binary_ufunc *Binary_op<Op>::dtable[] = DTYPE_DISPATCH(ufunc);

template <typename T>
struct Add {
    bool operator()(T &result, T x, T y) {
        result = x + y;
        return false;
    }
};

template <typename T>
struct Subtract {
    bool operator()(T &result, T x, T y) {
        result = x - y;
        return false;
    }
};

template <typename T>
struct Multiply {
    bool operator()(T &result, T x, T y) {
        result = x * y;
        return false;
    }
};

template <typename T>
struct Remainder {
    bool operator()(T &result, T x, T y);
};

template <>
bool Remainder<long>::operator()(long &result, long x, long y)
{
    if (y == 0 || (y == -1 && x == std::numeric_limits<long>::min())) {
        const char *msg = (y == 0) ?
            "Integer modulo by zero." : "Integer modulo overflow.";
        if (PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1) < 0) return true;
        result = 0;
        return false;
    }
    long x_mod_y = x % y;
    result = ((x ^ y) >= 0 /*same sign*/) ? x_mod_y : -x_mod_y;
    return false;
}

template <>
bool Remainder<double>::operator()(double &result, double x, double y)
{
    result = x - std::floor(x / y) * y;
    return false;
}

template <>
template <>
PyObject *Binary_op<Remainder>::ufunc<Complex>(int, const size_t*,
                                               PyObject*, const ptrdiff_t*,
                                               PyObject*, const ptrdiff_t*)
{
    PyErr_SetString(PyExc_TypeError,
                    "Modulo is not defined for complex numbers.");
    return 0;
}

template <typename T>
struct Floor_divide {
    bool operator()(T &result, T x, T y);
};

template <>
bool Floor_divide<long>::operator()(long &result, long x, long y)
{
    if (y == 0 || (y == -1 && x == std::numeric_limits<long>::min())) {
        const char *msg = (y == 0) ?
            "Integer division by zero." : "Integer division overflow.";
        if (PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1) < 0) return true;
        result = 0;
        return false;
    }
    long x_div_y = x / y;
    result = ((x ^ y) >= 0 /*same sign*/ || (x % y) == 0) ?
        x_div_y : x_div_y - 1;
    return false;
}

template <>
bool Floor_divide<double>::operator()(double &result, double x, double y)
{
    result = std::floor(x / y);
    return false;
}

template <>
template <>
PyObject *Binary_op<Floor_divide>::ufunc<Complex>(int, const size_t*,
                                                  PyObject*, const ptrdiff_t*,
                                                  PyObject*, const ptrdiff_t*)
{
    PyErr_SetString(PyExc_TypeError,
                    "Floor divide is not defined for complex numbers.");
    return 0;
}

template <typename T>
struct Divide {
    bool operator()(T &result, T x, T y) {
        result = x / y;
        return false;
    }
};

template <>
bool Divide<long>::operator()(long &result, long x, long y)
{
    Floor_divide<long> floor_divide;
    return floor_divide(result, x, y);
}

PyObject *dot_product(PyObject *a, PyObject *b)
{
    Dtype dtype;
    if (coerce_to_arrays(&a, &b, &dtype) < 0) return 0;

    PyObject *result = 0;
    int ndim_a, ndim_b;
    reinterpret_cast<Array_base*>(a)->ndim_shape(&ndim_a, 0);
    reinterpret_cast<Array_base*>(b)->ndim_shape(&ndim_b, 0);
    if (ndim_a == 0 || ndim_b == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "dot does not support zero-dimensional arrays yet.");
        goto end;
    }

    if (ndim_a == 1 && ndim_b == 1)
        result = array_scalar_product_dtable[int(dtype)](a, b);
    else
        result = array_matrix_product_dtable[int(dtype)](a, b);

end:
    Py_DECREF(a);
    Py_DECREF(b);
    return result;
}

template <typename Op>
PyObject *apply_unary_ufunc(PyObject *a_)
{
    typedef typename Op::IType IT;
    typedef typename Op::OType OT;
    Op operation;

    if (Op::error) {
        PyErr_SetString(PyExc_TypeError, Op::error);
        return 0;
    }

    assert(Array<IT>::check_exact(a_)); Array<IT> *a = (Array<IT>*)a_;
    int ndim;
    size_t *shape;
    a->ndim_shape(&ndim, &shape);
    if (ndim == 0)
        return (PyObject*)pyobject_from_number(operation(*a->data()));

    if (Op::unchanged) {
        Py_INCREF(a_);
        return a_;
    }

    size_t size;
    Array<OT> *result = Array<OT>::make(ndim, shape, &size);
    if (result == 0) return 0;
    IT *src = a->data();
    OT *dest = result->data();
    for (size_t i = 0; i < size; ++i) dest[i] = operation(src[i]);
    return (PyObject*)result;
}

template <typename T>
struct Negative {
    typedef T IType;
    typedef T OType;
    static const char *error;
    static const bool unchanged = false;
    T operator()(T x) { return -x; }
};

template <typename T>
const char *Negative<T>::error = 0;

template <typename T>
struct Positive {
    typedef T IType;
    typedef T OType;
    static const char *error;
    static const bool unchanged = true;
    T operator()(T x) { return x; }
};

template <typename T>
const char *Positive<T>::error = 0;

template <typename T>
struct Absolute {
    typedef T IType;
    typedef T OType;
    static const char *error;
    static const bool unchanged = false;
    T operator()(T x) { return std::abs(x); }
};

template <>
struct Absolute<Complex> {
    typedef Complex IType;
    typedef double OType;
    static const char *error;
    static const bool unchanged = false;
    double operator()(Complex x) { return std::abs(x); }
};

template <typename T>
const char *Absolute<T>::error = 0;
// Needed for gcc 4.4.
const char *Absolute<Complex>::error = 0;

template <typename T>
struct Conjugate {
    typedef T IType;
    typedef T OType;
    static const char *error;
    static const bool unchanged = true;
    T operator()(T x) { return x; }
};

template <>
struct Conjugate<Complex> {
    typedef Complex IType;
    typedef Complex OType;
    static const char *error;
    static const bool unchanged = false;
    Complex operator()(Complex x) { return std::conj(x); }
};

template <typename T>
const char *Conjugate<T>::error = 0;
const char *Conjugate<Complex>::error = 0;

// Integers are not changed by any kind of rounding.
template <typename Kind>
struct Round<Kind, long> {
    typedef long IType;
    typedef long OType;
    static const char *error;
    static const bool unchanged = true;
    long operator()(long x) { return x; }
};

template <typename Kind>
const char *Round<Kind, long>::error = 0;

template <typename Kind>
struct Round<Kind, double> {
    typedef double IType;
    typedef double OType;
    static const char *error;
    static const bool unchanged = false;
    double operator()(double x) {
        Kind rounding_kind;
        return rounding_kind(x);
    }
};

template <typename Kind>
const char *Round<Kind, double>::error = 0;

template <typename Kind>
struct Round<Kind, Complex> {
    typedef Complex IType;
    typedef Complex OType;
    static const char *error;
    static const bool unchanged = false;
    Complex operator()(Complex) {
        return std::numeric_limits<Complex>::quiet_NaN();
    }
};

template <typename Kind>
const char *Round<Kind, Complex>::error =
    "Rounding is not defined for complex numbers.";

// The following three types are used as Kind template parameter for Round.

struct Nearest {
    // Rounding to nearest even, same as numpy.
    double operator()(double x) {
        double y = std::floor(x), r = x - y;
        if (r > 0.5) {
            ++y;
        } else if (r == 0.5) {
            r = y - 2.0 * std::floor(0.5 * y);
            if (r == 1) ++y;
        }
        if (y == 0 && x < 0) y = -0.0;
        return y;
    }
};

struct Floor { double operator()(double x) { return std::floor(x); } };

struct Ceil { double operator()(double x) { return std::ceil(x); } };

template <typename T>
PyNumberMethods Array<T>::as_number = {
    Binary_op<Add>::apply,           // nb_add
    Binary_op<Subtract>::apply,      // nb_subtract
    Binary_op<Multiply>::apply,      // nb_multiply
    Binary_op<Divide>::apply,        // nb_divide
    Binary_op<Remainder>::apply,     // nb_remainder
    (binaryfunc)0,                   // nb_divmod
    (ternaryfunc)0,                  // nb_power
    apply_unary_ufunc<Negative<T> >, // nb_negative
    apply_unary_ufunc<Positive<T> >, // nb_positive
    apply_unary_ufunc<Absolute<T> >, // nb_absolute
    (inquiry)0,                      // nb_nonzero
    (unaryfunc)0,                    // nb_invert
    (binaryfunc)0,                   // nb_lshift
    (binaryfunc)0,                   // nb_rshift
    (binaryfunc)0,                   // nb_and
    (binaryfunc)0,                   // nb_xor
    (binaryfunc)0,                   // nb_or
    (coercion)0,                     // nb_coerce
    (unaryfunc)0,                    // nb_int
    (unaryfunc)0,                    // nb_long
    (unaryfunc)0,                    // nb_float
    (unaryfunc)0,                    // nb_oct
    (unaryfunc)0,                    // nb_hex

    (binaryfunc)0,                   // nb_inplace_add
    (binaryfunc)0,                   // nb_inplace_subtract
    (binaryfunc)0,                   // nb_inplace_multiply
    (binaryfunc)0,                   // nb_inplace_divide
    (binaryfunc)0,                   // nb_inplace_remainder
    (ternaryfunc)0,                  // nb_inplace_power
    (binaryfunc)0,                   // nb_inplace_lshift
    (binaryfunc)0,                   // nb_inplace_rshift
    (binaryfunc)0,                   // nb_inplace_and
    (binaryfunc)0,                   // nb_inplace_xor
    (binaryfunc)0,                   // nb_inplace_or

    Binary_op<Floor_divide>::apply,  // nb_floor_divide
    (binaryfunc)0,                   // nb_true_divide
    (binaryfunc)0,                   // nb_inplace_floor_divide
    (binaryfunc)0,                   // nb_inplace_true_divide

    (unaryfunc)0                     // nb_index
};

// Explicit instantiations.
template PyNumberMethods Array<long>::as_number;
template PyNumberMethods Array<double>::as_number;
template PyNumberMethods Array<Complex>::as_number;

template PyObject *apply_unary_ufunc<Conjugate<long> >(PyObject*);
template PyObject *apply_unary_ufunc<Conjugate<double> >(PyObject*);
template PyObject *apply_unary_ufunc<Conjugate<Complex> >(PyObject*);

template PyObject *apply_unary_ufunc<Round<Nearest, long> >(PyObject*);
template PyObject *apply_unary_ufunc<Round<Nearest, double> >(PyObject*);
template PyObject *apply_unary_ufunc<Round<Nearest, Complex> >(PyObject*);

template PyObject *apply_unary_ufunc<Round<Floor, long> >(PyObject*);
template PyObject *apply_unary_ufunc<Round<Floor, double> >(PyObject*);
template PyObject *apply_unary_ufunc<Round<Floor, Complex> >(PyObject*);

template PyObject *apply_unary_ufunc<Round<Ceil, long> >(PyObject*);
template PyObject *apply_unary_ufunc<Round<Ceil, double> >(PyObject*);
template PyObject *apply_unary_ufunc<Round<Ceil, Complex> >(PyObject*);
