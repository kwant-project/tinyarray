#ifndef ARITHMETIC_HH
#define ARITHMETIC_HH

template <typename T>
PyObject *array_scalar_product(PyObject *a, PyObject *b);
template <typename T>
PyObject *array_matrix_product(PyObject *a, PyObject *b);

extern PyNumberMethods as_number;

#endif // !ARITHMETIC_HH
