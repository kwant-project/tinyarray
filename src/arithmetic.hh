// Copyright 2012-2013 Tinyarray authors.
//
// This file is part of Tinyarray.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://git.kwant-project.org/tinyarray/about/LICENSE.  A list of Tinyarray
// authors can be found in the README file at the top-level directory of this
// distribution and at http://git.kwant-project.org/tinyarray/about/.

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

// Binary operations
template <typename T> struct Add;
template <typename T> struct Subtract;
template <typename T> struct Multiply;
template <typename T> struct Divide;
template <typename T> struct Remainder;
template <typename T> struct Floor_divide;


template <typename Op> PyObject *apply_unary_ufunc(PyObject *a);

// Unaray operations
template <typename T> struct Negative;
template <typename T> struct Positive;
template <typename T> struct Absolute;
template <typename T> struct Conjugate;
template <typename Kind, typename T> struct Round;

// Kinds of rounding, to be used with Round.
struct Nearest;
struct Floor;
struct Ceil;


#endif // !ARITHMETIC_HH
