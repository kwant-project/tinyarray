#ifndef ARITHMETIC_HH
#define ARITHMETIC_HH

#include <cstddef>

PyObject *dot_product(PyObject *a, PyObject *b);

typedef PyObject *Binary_ufunc(int, const size_t*,
                               PyObject*, const ptrdiff_t*,
                               PyObject*, const ptrdiff_t*);

template <template <typename> class Op>
class Binary_op {
public:
    static PyObject *apply(PyObject *a, PyObject *b);
private:
    template <typename T>
    static PyObject *ufunc(int ndim, const size_t *shape,
                           PyObject *a_, const ptrdiff_t *hops_a,
                           PyObject *b_, const ptrdiff_t *hops_b);

    static Binary_ufunc *dtable[];
};

template <typename T> struct Add;
template <typename T> struct Subtract;
template <typename T> struct Multiply;
template <typename T> struct Divide;
template <typename T> struct Remainder;
template <typename T> struct Floor_divide;

#endif // !ARITHMETIC_HH
