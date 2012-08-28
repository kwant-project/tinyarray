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

#endif // !CONVERSION_HH
