// Copyright 2012-2013 Tinyarray authors.
//
// This file is part of Tinyarray.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://git.kwant-project.org/tinyarray/about/LICENSE.  A list of Tinyarray
// authors can be found in the README file at the top-level directory of this
// distribution and at http://git.kwant-project.org/tinyarray/about/.

#ifndef CONVERSION_HH
#define CONVERSION_HH

#include <complex>
typedef std::complex<double> Complex;

inline PyObject *pyobject_from_number(long x)
{
    return PyInt_FromLong(x);
}

inline PyObject *pyobject_from_number(double x)
{
    return PyFloat_FromDouble(x);
}

inline PyObject *pyobject_from_number(Complex x)
{
    Py_complex *p = (Py_complex*)&x;
    return PyComplex_FromCComplex(*p);
}

template <typename T>
T number_from_pyobject(PyObject *obj);

template <>
inline double number_from_pyobject(PyObject *obj)
{
    return PyFloat_AsDouble(obj);
}

template <>
inline long number_from_pyobject(PyObject *obj)
{
    return PyInt_AsLong(obj);
}

template <>
inline Complex number_from_pyobject(PyObject *obj)
{
    Py_complex temp = PyComplex_AsCComplex(obj);
    return Complex(temp.real, temp.imag);
}

template <typename T>
T number_from_pyobject_exact(PyObject *obj);

template <>
inline double number_from_pyobject_exact(PyObject *obj)
{
    return PyFloat_AsDouble(obj);
}

template <>
inline long number_from_pyobject_exact(PyObject *obj)
{
    // This function will fail when the conversion to long is not exact.
    return PyNumber_AsSsize_t(obj, PyExc_TypeError);
}

template <>
inline Complex number_from_pyobject_exact(PyObject *obj)
{
    Py_complex temp = PyComplex_AsCComplex(obj);
    return Complex(temp.real, temp.imag);
}

template<typename Tdest, typename Tsrc>
inline Tdest number_from_ptr(const void *data)
{
    return static_cast<Tdest>(*(reinterpret_cast<const Tsrc *>(data)));
};

// Specializations for the cases that can fail

template<>
inline long number_from_ptr<long, unsigned int>(const void *data)
{
    const unsigned int *ptr = reinterpret_cast<const unsigned int*>(data);

    if (*ptr > static_cast<unsigned int>(std::numeric_limits<long>::max())) {
        PyErr_Format(PyExc_OverflowError,
                     "Integer too large for long");
        return -1;
    } else {
        return static_cast<long>(*ptr);;
    }
}

template<>
inline long number_from_ptr<long, unsigned long>(const void *data)
{
    const unsigned long *ptr = reinterpret_cast<const unsigned long *>(data);

    if (*ptr > static_cast<unsigned long>(std::numeric_limits<long>::max())) {
        PyErr_Format(PyExc_OverflowError,
                     "Integer too large for long");
        return -1;
    } else {
        return static_cast<long>(*ptr);;
    }
}

template<>
inline long number_from_ptr<long, long long>(const void *data)
{
    const long long *ptr = reinterpret_cast<const long long*>(data);

    if (*ptr > std::numeric_limits<long>::max() ||
        *ptr < std::numeric_limits<long>::min()) {
        PyErr_Format(PyExc_OverflowError,
                     "Integer too large for long");
        return -1;
    } else {
        return static_cast<long>(*ptr);;
    }
}

template<>
inline long number_from_ptr<long, unsigned long long>(const void *data)
{
    const unsigned long long *ptr =
        reinterpret_cast<const unsigned long long*>(data);

    if (*ptr >
       static_cast<unsigned long long>(std::numeric_limits<long>::max())) {
        PyErr_Format(PyExc_OverflowError,
                     "Integer too large for long");
        return -1;
    } else {
        return static_cast<long>(*ptr);;
    }
}

template<typename Tdest, typename Tsrc>
inline Tdest _int_from_floatptr_exact(const void *data)
{
    const Tsrc *ptr = reinterpret_cast<const Tsrc*>(data);
    Tdest result = static_cast<Tdest>(*ptr);

    // Note: the > max and < min tests are unreliable if the float obtained
    // after rounding has less precision than necessary (which is typically the
    // case for float and double). The two other tests are supposed to catch
    // those problems.
    if (*ptr > std::numeric_limits<Tdest>::max() ||
        *ptr < std::numeric_limits<Tdest>::min() ||
        (*ptr > 0 && result < 0) || (*ptr < 0 && result > 0)) {
        PyErr_Format(PyExc_OverflowError,
                     "Float too large to be represented by long");
        return -1;
    } else
    {
        return result;
    }
}

template<>
inline long number_from_ptr<long, float>(const void *data)
{
    return _int_from_floatptr_exact<long, float>(data);
}

template<>
inline long number_from_ptr<long, double>(const void *data)
{
    return _int_from_floatptr_exact<long, double>(data);
}

template<>
inline long number_from_ptr<long, long double>(const void *data)
{
    return _int_from_floatptr_exact<long, long double>(data);
}

#endif // !CONVERSION_HH
