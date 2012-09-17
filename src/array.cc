#include <Python.h>
#include <cstddef>
#include <sstream>
#include <limits>
#include "array.hh"
#include "arithmetic.hh"
#include "functions.hh"
#include "conversion.hh"

template <>
const char *Array<long>::pyformat = "l";
template <>
const char *Array<double>::pyformat = "d";
template <>
const char *Array<Complex>::pyformat = "Zd";

const char *dtype_names[] = {"int", "float", "complex"};

namespace {

PyObject *int_str, *long_str, *float_str, *complex_str, *index_str;

Dtype dtype_of_scalar(PyObject *obj)
{
    if (PyComplex_Check(obj)) return Dtype::COMPLEX;
    if (PyFloat_Check(obj)) return Dtype::DOUBLE;
    if (PyInt_Check(obj)) return Dtype::LONG;
    // TODO: The following line should be removed for Python 3.
    if (PyLong_Check(obj)) return Dtype::LONG;
    if (PyObject_HasAttr(obj, index_str)) return Dtype::LONG;

    // I'm not sure about this paragraph.  Does the existence of a __complex__
    // method already signify that the number is to be interpreted as a complex
    // number?  What about __float__?  Perhaps the following code is useless.
    // In practice (with built-in and numpy numerical types) it never plays a
    // role anyway.
    if (PyObject_HasAttr(obj, complex_str)) return Dtype::COMPLEX;
    if (PyObject_HasAttr(obj, float_str)) return Dtype::DOUBLE;
    if (PyObject_HasAttr(obj, int_str)) return Dtype::LONG;
    // TODO: The following line should be removed for Python 3.
    if (PyObject_HasAttr(obj, long_str)) return Dtype::LONG;

    return Dtype::NONE;
}

template<typename O, typename I>
PyObject *convert_array(PyObject *in_, int ndim, size_t *shape)
{
    assert(Array<I>::check_exact(in_)); Array<I> *in = (Array<I>*)in_;
    size_t size;
    if (ndim == -1) {
        assert(shape == 0);
        in->ndim_shape(&ndim, &shape);
    } else {
#ifndef NDEBUG
        int in_ndim;
        size_t *in_shape;
        in->ndim_shape(&in_ndim, &in_shape);
        assert(shape);
        assert(calc_size(ndim, shape) == calc_size(in_ndim, in_shape));
#endif
    }
    Array<O> *out = Array<O>::make(ndim, shape, &size);
    I *src = in->data();
    O *dest = out->data();
    for (size_t i = 0; i < size; ++i) dest[i] = src[i];
    return (PyObject*)out;
}

typedef PyObject *Convert_array(PyObject*, int, size_t*);

Convert_array *convert_array_dtable[][3] = {
    {convert_array<long, long>,
     convert_array<long, double>,
     0},
    {convert_array<double, long>,
     convert_array<double, double>,
     0},
    {convert_array<Complex, long>,
     convert_array<Complex, double>,
     convert_array<Complex, Complex>}
};

PyObject *convert_array(Dtype dtype_out, PyObject *in, Dtype dtype_in,
                        int ndim = -1, size_t *shape = 0)
{
    if (dtype_in == Dtype::NONE)
        dtype_in = get_dtype(in);
    assert(get_dtype(in) == get_dtype(in));
    Convert_array *func = convert_array_dtable[int(dtype_out)][int(dtype_in)];
    if (!func) {

        PyErr_Format(PyExc_TypeError, "Cannot convert %s to %s.",
                     dtype_names[int(dtype_in)], dtype_names[int(dtype_out)]);
        return 0;
    }
    return func(in, ndim, shape);
}

const char *seq_err_msg =
    "A sequence does not support sequence protocol - "
    "this is probably due to a bug in numpy for 0-d arrays.";

// This function determines the shape of an array like sequence (or sequence of
// sequences or number) given to it as first parameter.  The dtype is
// determined from the first element of the sequence.
//
// All four arguments after the first one are written to. `shape' and `seqs'
// must have space for at least `max_ndim' elements.
//
// After successful execution `seqs' will contain `ndim' new references
// returned by PySequence_Fast.
int examine_arraylike(PyObject *arraylike, int *ndim, size_t *shape,
                      PyObject **seqs, Dtype *dtype)
{
    PyObject *p = arraylike;
    int d = -1;
    while (true) {
        if (PySequence_Check(p)) {
            ++d;
            if (d == ptrdiff_t(max_ndim)) {
                // Strings are, in a way, infinitely nested sequences because
                // the first element of a string is a again a string.
                if (PyString_Check(p))
                    PyErr_SetString(PyExc_TypeError, "Expecting a number.");
                else
                    PyErr_SetString(PyExc_ValueError, "Too many dimensions.");
                --d;
                goto fail;
            }
        } else {
            // We are in the innermost sequence.  Determine the dtype if
            // requested.
            if (dtype) {
                *dtype = dtype_of_scalar(p);
                if (*dtype == Dtype::NONE) {
                    PyErr_SetString(PyExc_TypeError, "Expecting a number.");
                    goto fail;
                }
            }
            break;
        }

        // See http://projects.scipy.org/numpy/ticket/2199.
        seqs[d] = PySequence_Fast(p, seq_err_msg);
        if (!seqs[d]) {--d; goto fail;}

        if ((shape[d] = PySequence_Fast_GET_SIZE(seqs[d]))) {
            p = *PySequence_Fast_ITEMS(seqs[d]);
        } else {
            // We are in the innermost sequence which is empty.
            if (dtype) *dtype = Dtype::NONE;
            break;
        }
    }
    *ndim = d + 1;
    return 0;

fail:
    for (; d >= 0; --d) Py_DECREF(seqs[d]);
    return -1;
}

// This function is designed to be run after examine_arraylike.  It takes care
// of releasing the references passed to it in seqs.
template <typename T>
int readin_arraylike(T *dest, int ndim, const size_t *shape,
                     PyObject *arraylike, PyObject **seqs, bool exact)
{
    if (ndim == 0) {
        T value;
        if (exact)
            value = number_from_pyobject_exact<T>(arraylike);
        else
            value = number_from_pyobject<T>(arraylike);
        if (value == T(-1) && PyErr_Occurred()) return -1;
        *dest++ = value;
        return 0;
    }

    // seqs is the stack of sequences being processed, all returned by
    // PySequence_Fast.  ps[d] and es[d] are the begin and end of the elements
    // of seqs[d - 1].
    PyObject **ps[max_ndim], **es[max_ndim];
    es[0] = ps[0] = 0;

    for (int d = 1; d < ndim; ++d) {
        PyObject **p = PySequence_Fast_ITEMS(seqs[d - 1]);
        ps[d] = p + 1;
        es[d] = p + shape[d - 1];
    }

    int d = ndim - 1;
    size_t len = shape[d];
    PyObject **p = PySequence_Fast_ITEMS(seqs[d]), **e = p + len;
    while (true) {
        if (len && PySequence_Check(p[0])) {
            if (d + 1 == ndim) {
                PyErr_SetString(PyExc_ValueError,
                                "Input has irregular nesting depth.");
                goto fail;
            }
            ++d;
            ps[d] = p;
            es[d] = e;
        } else {
            // Read-in a leaf sequence.
            while (p < e) {
                T value;
                if (exact)
                    value = number_from_pyobject_exact<T>(*p++);
                else
                    value = number_from_pyobject<T>(*p++);
                if (value == T(-1) && PyErr_Occurred()) goto fail;
                *dest++ = value;
            }
            Py_DECREF(seqs[d]);

            while (ps[d] == es[d]) {
                if (d == 0) {
                    // Success!
                    return 0;
                }
                --d;
                Py_DECREF(seqs[d]);
            }
            if (!PySequence_Check(*ps[d])) {
                --d;
                PyErr_SetString(PyExc_ValueError,
                                "Input has irregular nesting depth.");
                goto fail;
            }
        }

        // See http://projects.scipy.org/numpy/ticket/2199.
        seqs[d] = PySequence_Fast(*ps[d]++, seq_err_msg);
        if (!seqs[d]) {--d; goto fail;}
        len = PySequence_Fast_GET_SIZE(seqs[d]);

        // Verify that the length of the current sequence agrees with the
        // shape.
        if (len != shape[d]) {
            PyErr_SetString(PyExc_ValueError,
                            "Input has irregular shape.");
            goto fail;
        }

        p = PySequence_Fast_ITEMS(seqs[d]);
        e = p + len;
    }

fail:
    while (true) {
        Py_DECREF(seqs[d]);
        if (d == 0) break;
        --d;
    }
    return -1;
}

template <typename T>
PyObject *make_and_readin_array(PyObject *in, int ndim_in, int ndim_out,
                                const size_t *shape_out, PyObject **seqs,
                                bool exact)
{
    Array<T> *result = Array<T>::make(ndim_out, shape_out);
    assert(ndim_out >= ndim_in);
#ifndef NDEBUG
    for (int d = 0, e = ndim_out - ndim_in; d < e; ++d)
        assert(shape_out[d] == 1);
#endif
    if (result == 0) return 0;
    if (readin_arraylike<T>(result->data(), ndim_in,
                            shape_out + ndim_out - ndim_in, in, seqs, exact)
        == -1) {
        Py_DECREF(result);
        return 0;
    }
    return (PyObject*)result;
}

PyObject *(*make_and_readin_array_dtable[])(
    PyObject*, int, int, const size_t*, PyObject**, bool) =
    DTYPE_DISPATCH(make_and_readin_array);

template <typename T>
PyObject *to_pystring(Array<T> *self, PyObject* to_str(PyObject *),
                      const char *header, const char *trailer,
                      const char *indent, const char *separator)
{
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);

    std::ostringstream o;
    o << header;

    const T *p = self->data();
    if (ndim > 0) {
        int d = 0;
        size_t i[max_ndim];
        i[0] = shape[0];

        o << '[';
        while (true) {
            if (i[d]) {
                --i[d];
                if (d < ndim - 1) {
                    o << '[';
                    ++d;
                    i[d] = shape[d];
                } else {
                    PyObject *num = pyobject_from_number(*p++);
                    PyObject *str = to_str(num);
                    o << PyString_AsString(str);
                    Py_DECREF(str);
                    Py_DECREF(num);
                    if (i[d] > 0) o << separator << ' ';
                }
            } else {
                o << ']';
                if (d == 0) break;
                --d;
                if (i[d]) {
                    o << separator << "\n " << indent;
                    for (int i = 0; i < d; ++i) o << ' ';
                }
            }
        }
    } else {
        PyObject *num = pyobject_from_number(*p);
        PyObject *str = to_str(num);
        o << PyString_AsString(str);
        Py_DECREF(str);
        Py_DECREF(num);
    }
    o << trailer;

    return PyString_FromString(o.str().c_str());
}

template <typename T>
PyObject *repr(Array<T> *self)
{
    return to_pystring(self, PyObject_Repr, "array(", ")", "      ", ",");
}

template <typename T>
PyObject *str(Array<T> *self)
{
    return to_pystring(self, PyObject_Str, "", "", "", "");
}

PyDoc_STRVAR(doc, "array docstring: to be written\n");

Py_ssize_t len(Array_base *self)
{
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    if (ndim == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object.");
        return -1;
    }
    return shape[0];
}

Py_ssize_t index_from_key(int ndim, const size_t *shape, PyObject *key)
{
    long indices[max_ndim];
    Py_ssize_t res = load_index_seq_as_long(key, indices, max_ndim);
    if (res == -1) {
        PyErr_SetString(PyExc_IndexError, "Invalid index.");
        return -1;
    }
    if (int(res) != ndim) {
        PyErr_SetString(PyExc_IndexError, "Number of indices "
                        "must be equal to number of dimensions.");
        return -1;
    }

    int d = 0;
    Py_ssize_t s = shape[0];
    Py_ssize_t index = indices[0];
    if (index < 0) index += s;
    if (index < 0 || index >= s) goto out_of_range;
    for (d = 1; d < ndim; ++d) {
        s = shape[d];
        Py_ssize_t i = indices[d];
        if (i < 0) i += s;
        if (i < 0 || i >= s) goto out_of_range;
        index *= s;
        index += i;
    }
    return index;

out_of_range:
    PyErr_Format(PyExc_IndexError, "Index %ld out of range "
                 "(-%ld <= index < %ld) in dimension %d.",
                 indices[d], s, s, d);
    return -1;
}

template <typename T>
PyObject *getitem(Array<T> *self, PyObject *key)
{
    if (PySlice_Check(key)) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "Slices are not implemented.");
        return 0;
    } else {
        int ndim;
        size_t *shape;
        self->ndim_shape(&ndim, &shape);
        T *data = self->data();
        Py_ssize_t index = index_from_key(ndim, shape, key);
        if (index == -1) return 0;
        return pyobject_from_number(data[index]);
    }
}

template <typename T>
PyObject *seq_getitem(Array<T> *self, Py_ssize_t index)
{
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    assert(ndim != 0);

    if (index < 0) index += shape[0];
    if (size_t(index) >= shape[0]) {
        PyErr_SetString(PyExc_IndexError, "Invalid index.");
        return 0;
    }

    T *src = self->data();
    if (ndim == 1) {
        assert(index >= 0);
        assert(size_t(index) < shape[0]);
        return pyobject_from_number(src[index]);
    }

    assert(ndim > 1);
    size_t item_size;
    Array<T> *result = Array<T>::make(ndim - 1, shape + 1, &item_size);
    if (!result) return 0;
    src += index * item_size;
    T *dest = result->data();
    for (size_t i = 0; i < item_size; ++i) dest[i] = src[i];
    return (PyObject*)result;
}

template <typename T>
int getbuffer(Array<T> *self, Py_buffer *view, int flags)
{
    int ndim;
    size_t *shape, size;

    assert(view);
    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
        PyErr_SetString(PyExc_BufferError,
                        "Tinyarrays are not Fortran contiguous.");
        goto fail;
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
        PyErr_SetString(PyExc_BufferError, "Tinyarrays are not writeable");
        goto fail;
    }

    self->ndim_shape(&ndim, &shape);
    size = calc_size(ndim, shape);

    view->buf = self->data();
    view->itemsize = sizeof(T);
    view->len = size * view->itemsize;
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
        view->format = (char*)Array<T>::pyformat;
    else
        view->format = 0;
    if ((flags & PyBUF_ND) == PyBUF_ND) {
        view->ndim = ndim;
        view->shape = (Py_ssize_t*)shape;
        // From the documentation it's not clear whether it is allowed not to
        // set strides to NULL (for C continuous arrays), but it works, so we
        // do it.  However, there is a bug in current numpy
        // (http://projects.scipy.org/numpy/ticket/2197) which requires strides
        // if view->len == 0.  Because we don't have proper strides, we just
        // set strides to shape.  This dirty trick seems to work well -- no one
        // looks at strides when len == 0.
        if (size != 0)
            view->strides = 0;
        else
            view->strides = (Py_ssize_t*)shape;
    } else {
        view->ndim = 0;
        view->shape = 0;
        view->strides = 0;
    }
    view->internal = 0;
    view->suboffsets = 0;

    // Success.
    Py_INCREF(self);
    view->obj = (PyObject*)self;

    return 0;

fail:
    view->obj = 0;
    return -1;
}

// Given the same input, these hash functions return the same result as those
// in Python.  As tinyarrays compare equal to equivalent tuples it is important
// for the hashes to agree.  If not, there will be problems with dictionaries.

long hash(long x)
{
    return x != -1 ? x : -2;
}

long hash(double x)
{
    const double two_to_31st = 1L << 31;

    double intpart, fractpart;
    fractpart = modf(x, &intpart);
    if (fractpart == 0 &&
        intpart >= std::numeric_limits<long>::min() &&
        intpart <= std::numeric_limits<long>::max()) {
        return hash(long(intpart));
    }

    int expo;
    x = frexp(x, &expo) * two_to_31st;
    long hipart = x;                         // Take the top 32 bits.
    x = (x - double(hipart)) * two_to_31st;  // Get the next 32 bits.
    return hash(hipart + (long)x + (expo << 15));
}

long hash(Complex x)
{
    // x.imag == 0  =>  hash(x.imag) == 0  =>  hash(x) == hash(x.real)
    return hash(x.real()) + 1000003L * hash(x.imag());
}

// This routine calculates the hash of a multi-dimensional array.  The hash is
// equal to that of an arrangement of nested tuples equivalent to the array.
template <typename T>
long hash(Array<T> *self)
{
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    T *p = self->data();
    if (ndim == 0) return hash(*p);

    const long mult_init = 1000003, r_init = 0x345678;
    const long mul_addend = 82520, r_addend = 97531;
    Py_ssize_t i[max_ndim];
    long mult[max_ndim], r[max_ndim];
    --ndim;                     // For convenience.
    int d = 0;
    i[0] = shape[0];
    mult[0] = mult_init;
    r[0] = r_init;
    while (true) {
        if (i[d]) {
            --i[d];
            if (d == ndim) {
                r[d] = (r[d] ^ hash(*p++)) * mult[d];
                mult[d] += mul_addend + 2 * i[d];
            } else {
                ++d;
                i[d] = shape[d];
                mult[d] = mult_init;
                r[d] = r_init;
            }
        } else {
            if (d == 0) return hash(r[0] + r_addend);
            --d;
            r[d] = (r[d] ^ hash(r[d+1] + r_addend)) * mult[d];
            mult[d] += mul_addend + 2 * i[d];
        }
    }
}

template <typename T>
bool is_equal_data(PyObject *a_, PyObject *b_, size_t size)
{
    assert(Array<T>::check_exact(a_)); Array<T> *a = (Array<T>*)a_;
    assert(Array<T>::check_exact(b_)); Array<T> *b = (Array<T>*)b_;
    T *data_a = a->data();
    T *data_b = b->data();
    for (size_t i = 0; i < size; ++i)
        if (data_a[i] != data_b[i]) return false;
    return true;
}

bool (*is_equal_data_dtable[])(PyObject*, PyObject*, size_t) =
    DTYPE_DISPATCH(is_equal_data);

PyObject *richcompare(PyObject *a, PyObject *b, int op)
{
    if (op != Py_EQ && op != Py_NE) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    bool equal = (a == b);
    if (equal) goto done;

    Dtype dtype;
    if (coerce_to_arrays(&a, &b, &dtype) < 0) return 0;

    int ndim_a, ndim_b;
    size_t *shape_a, *shape_b;
    reinterpret_cast<Array_base*>(a)->ndim_shape(&ndim_a, &shape_a);
    reinterpret_cast<Array_base*>(b)->ndim_shape(&ndim_b, &shape_b);
    if (ndim_a != ndim_b) goto done;
    for (int d = 0; d < ndim_a; ++d)
        if (shape_a[d] != shape_b[d]) goto done;
    equal = is_equal_data_dtable[int(dtype)](a, b, calc_size(ndim_a, shape_a));

done:
    PyObject *result = ((op == Py_EQ) == equal) ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

PyObject *get_dtype_py(PyObject *self, void *)
{
    static PyObject *dtypes[] = {
        (PyObject*)&PyInt_Type,
        (PyObject*)&PyFloat_Type,
        (PyObject*)&PyComplex_Type
    };
    int dtype = int(get_dtype(self));
    assert(dtype < int(Dtype::NONE));
    return dtypes[dtype];
}

PyObject *get_ndim(Array_base *self, void *)
{
    int ndim;
    self->ndim_shape(&ndim, 0);
    return PyLong_FromLong(ndim);
}

PyObject *get_size(Array_base *self, void *)
{
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    return PyLong_FromSize_t(calc_size(ndim, shape));
}

PyObject *get_shape(Array_base *self, void *)
{
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    size_t result_shape = ndim;
    Array<long> *result = Array<long>::make(1, &result_shape);
    if (!result) return 0;
    long *data = result->data();
    for (int d = 0; d < ndim; ++d) data[d] = shape[d];
    return (PyObject*)result;
}

PyGetSetDef getset[] = {
    {(char*)"dtype", get_dtype_py, 0, 0, 0},
    {(char*)"ndim", (getter)get_ndim, 0, 0, 0},
    {(char*)"size", (getter)get_size, 0, 0, 0},
    {(char*)"shape", (getter)get_shape, 0, 0, 0},
    {0, 0, 0, 0, 0}               // Sentinel
};

// **************** Iterator ****************

template <typename T>
class Array_iter {
public:
    static Array_iter *make(Array<T> *array);
    static PyObject *next(Array_iter<T> *self);
    static PyObject *len(Array_iter<T> *self);
private:
    static PyMethodDef methods[];
    static PyTypeObject pytype;
    static const char *pyname;
    PyObject ob_base;
    size_t index;
    Array<T> *array;            // Set to 0 when iterator is exhausted.
};

template <>
const char *Array_iter<long>::pyname = "tinyarray.ndarrayiter_int";
template <>
const char *Array_iter<double>::pyname = "tinyarray.ndarrayiter_float";
template <>
const char *Array_iter<Complex>::pyname = "tinyarray.ndarrayiter_complex";

template <typename T>
Array_iter<T> *Array_iter<T>::make(Array<T> *array)
{
    int ndim;
    assert(Array<T>::check_exact((PyObject*)array));
    array->ndim_shape(&ndim, 0);
    if (ndim == 0) {
        PyErr_SetString(PyExc_TypeError, "Iteration over a 0-d array.");
        return 0;
    }
    Array_iter<T> *ret = PyObject_New(Array_iter<T>, &Array_iter<T>::pytype);
    if (ret == 0) return 0;
    ret->index = 0;
    Py_INCREF(array);
    ret->array = array;
    return ret;
}

template <typename T>
PyObject *Array_iter<T>::next(Array_iter<T> *self)
{
    Array<T> *array = self->array;
    if (array == 0) return 0;
    int ndim;
    size_t *shape;
    array->ndim_shape(&ndim, &shape);
    assert(ndim != 0);

    if (self->index == shape[0]) {
        // End of iteration.
        Py_DECREF(array);
        self->array = 0;
        return 0;
    }

    T *src = array->data();

    if (ndim == 1) {
        assert(size_t(self->index) < shape[0]);
        return pyobject_from_number(src[self->index++]);
    }

    assert(ndim > 1);
    size_t item_size;
    Array<T> *result = Array<T>::make(ndim - 1, shape + 1, &item_size);
    if (!result) return 0;
    src += item_size * self->index++;
    T *dest = result->data();
    for (size_t i = 0; i < item_size; ++i) dest[i] = src[i];
    return (PyObject*)result;
}

template <typename T>
PyObject *Array_iter<T>::len(Array_iter<T> *self)
{
    Py_ssize_t len = 0;
    if (self->array) {
#ifndef NDEBUG
        int ndim;
        self->array->ndim_shape(&ndim, 0);
        assert(ndim != 0);
#endif
        size_t *shape;
        self->array->ndim_shape(0, &shape);
        len = shape[0] - self->index;
    }
    return PyInt_FromSsize_t(len);
}

template <typename T>
PyMethodDef Array_iter<T>::methods[] = {
    {"__length_hint__", (PyCFunction)Array_iter<T>::len, METH_NOARGS,
     "Private method returning an estimate of len(list(it))."},
    {0, 0}                      // Sentinel
};

template <typename T>
PyTypeObject Array_iter<T>::pytype = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    pyname,                     // tp_name
    sizeof(Array_iter<T>),      // tp_basicsize
    0,                          // tp_itemsize
    // methods
    (destructor)PyObject_Del,   // tp_dealloc
    0,                          // tp_print
    0,                          // tp_getattr
    0,                          // tp_setattr
    0,                          // tp_compare
    0,                          // tp_repr
    0,                          // tp_as_number
    0,                          // tp_as_sequence
    0,                          // tp_as_mapping
    0,                          // tp_hash
    0,                          // tp_call
    0,                          // tp_str
    PyObject_GenericGetAttr,    // tp_getattro
    0,                          // tp_setattro
    0,                          // tp_as_buffer
    Py_TPFLAGS_DEFAULT,         // tp_flags
    0,                          // tp_doc
    0,                          // tp_traverse
    0,                          // tp_clear
    0,                          // tp_richcompare
    0,                          // tp_weaklistoffset
    PyObject_SelfIter,          // tp_iter
    (iternextfunc)Array_iter<T>::next, // tp_iternext
    Array_iter<T>::methods      // tp_methods
};

// The following explicit instantiations are necessary for GCC 4.6 but not for
// GCC 4.7.  I don't know why.

template PyObject *repr(Array<long>*);
template PyObject *repr(Array<double>*);
template PyObject *repr(Array<Complex>*);

template PyObject *str(Array<long>*);
template PyObject *str(Array<double>*);
template PyObject *str(Array<Complex>*);

template long hash(Array<long>*);
template long hash(Array<double>*);
template long hash(Array<Complex>*);

template int getbuffer(Array<long>*, Py_buffer*, int);
template int getbuffer(Array<double>*, Py_buffer*, int);
template int getbuffer(Array<Complex>*, Py_buffer*, int);

template PyObject *getitem(Array<long>*, PyObject*);
template PyObject *getitem(Array<double>*, PyObject*);
template PyObject *getitem(Array<Complex>*, PyObject*);

template PyObject *seq_getitem(Array<long>*, Py_ssize_t);
template PyObject *seq_getitem(Array<double>*, Py_ssize_t);
template PyObject *seq_getitem(Array<Complex>*, Py_ssize_t);

} // Anonymous namespace

// **************** Public interface ****************

extern "C"
void inittinyarray()
{
    if (PyType_Ready(&Array<long>::pytype) < 0) return;
    if (PyType_Ready(&Array<double>::pytype) < 0) return;
    if (PyType_Ready(&Array<Complex>::pytype) < 0) return;

    PyObject* m = Py_InitModule("tinyarray", functions);

    Py_INCREF(&Array<long>::pytype);
    Py_INCREF(&Array<double>::pytype);
    Py_INCREF(&Array<Complex>::pytype);

    PyModule_AddObject(m, "ndarray_int",
                       (PyObject *)&Array<long>::pytype);
    PyModule_AddObject(m, "ndarray_float",
                       (PyObject *)&Array<double>::pytype);
    PyModule_AddObject(m, "ndarray_complex",
                       (PyObject *)&Array<Complex>::pytype);

    // We never release these references but this is not a problem.  The Python
    // interpreter does the same, see try_complex_special_method in
    // complexobject.c
    int_str = PyString_InternFromString("__int__");
    if (int_str == 0) return;
    long_str = PyString_InternFromString("__long__");
    if (long_str == 0) return;
    float_str = PyString_InternFromString("__float__");
    if (float_str == 0) return;
    complex_str = PyString_InternFromString("__complex__");
    if (complex_str == 0) return;
    index_str = PyString_InternFromString("__index__");
    if (complex_str == 0) return;
}

Py_ssize_t load_index_seq_as_long(PyObject *obj, long *out, Py_ssize_t maxlen)
{
    assert(maxlen >= 1);
    Py_ssize_t len;
    if (PySequence_Check(obj)) {
        obj = PySequence_Fast(obj, "Bug in tinyarray, load_index_seq_as_long");
        if (!obj) return -1;
        len = PySequence_Fast_GET_SIZE(obj);
        if (len > maxlen) {
            PyErr_Format(PyExc_ValueError, "Sequence too long."
                         " Maximum length is %ld.", maxlen);
            goto fail;
        }
        for (PyObject **p = PySequence_Fast_ITEMS(obj), **e = p + len; p < e;
             ++p, ++out) {
            PyObject *index = PyNumber_Index(*p);
            if (index == 0) goto fail;
            *out = PyInt_AsLong(index);
            Py_DECREF(index);
            if (*out == -1 and PyErr_Occurred()) goto fail;
        }
    } else {
        len = 1;
        *out = PyInt_AsLong(obj);
        if (*out == -1 and PyErr_Occurred()) return -1;
    }
    return len;

fail:
    Py_DECREF(obj);
    return -1;
}

Py_ssize_t load_index_seq_as_ulong(PyObject *obj, unsigned long *uout,
                                   Py_ssize_t maxlen, const char *errmsg)
{
    long *out = reinterpret_cast<long*>(uout);
    Py_ssize_t len = load_index_seq_as_long(obj, out, maxlen);
    if (len == -1) return -1;
    for (Py_ssize_t i = 0; i < len; ++i)
        if (out[i] < 0) {
            if (errmsg == 0)
                errmsg = "Sequence may not contain negative values.";
            PyErr_SetString(PyExc_ValueError, errmsg);
            return -1;
        }
    return len;
}

// If *dtype == Dtype::NONE the simplest fitting dtype (at least dtype_min)
// will be used and written back to *dtype.  Any other value of *dtype requests
// an array of the given dtype.
PyObject *array_from_arraylike(PyObject *in, Dtype *dtype, Dtype dtype_min)
{
    Dtype dtype_in = get_dtype(in), dt = *dtype;
    int ndim;
    size_t shape[max_ndim];
    PyObject *seqs[max_ndim], *result;
    if (dtype_in != Dtype::NONE) {
        // in is already an array
        if (dt == Dtype::NONE)
            dt = Dtype(std::max(int(dtype_in), int(dtype_min)));
        if (dt == dtype_in)
            Py_INCREF(result = in);
        else
            result = convert_array(dt, in, dtype_in);
    } else if (dt == Dtype::NONE) {
        // No specific dtype has been requested.  It will be determined by the
        // input.
        PyObject *seqs_copy[max_ndim];
        if (examine_arraylike(in, &ndim, shape, seqs, &dt) == -1)
            return 0;
        for (int d = 0; d < ndim; ++d) Py_INCREF(seqs_copy[d] = seqs[d]);
        if (dt == Dtype::NONE) {
            assert(shape[ndim - 1] == 0);
            dt = default_dtype;
        }
        if (int(dt) < int(dtype_min)) dt = dtype_min;
        while (true) {
            result = make_and_readin_array_dtable[int(dt)](
                in, ndim, ndim, shape, seqs, true);
            if (result) break;
            dt = Dtype(int(dt) + 1);
            if (dt == Dtype::NONE) {
                result = 0;
                break;
            }
            PyErr_Clear();
            for (int d = 0; d < ndim; ++d) Py_INCREF(seqs[d] = seqs_copy[d]);
        }
        for (int d = 0; d < ndim; ++d) Py_DECREF(seqs_copy[d]);
    } else {
        // A specific dtype has been requested.
        if (examine_arraylike(in, &ndim, shape, seqs, 0) == -1)
            return 0;
        else
            result = make_and_readin_array_dtable[int(dt)](
                in, ndim, ndim, shape, seqs, false);
    }
    *dtype = dt;
    return result;
}

// If *dtype == Dtype::NONE the simplest fitting dtype (at least dtype_min)
// will be used and written back to *dtype.  Any other value of *dtype requests
// an array of the given dtype.
PyObject *matrix_from_arraylike(PyObject *in, Dtype *dtype, Dtype dtype_min)
{
    Dtype dtype_in = get_dtype(in), dt = *dtype;
    int ndim;
    size_t shape[max_ndim];
    PyObject *seqs[max_ndim], *result;
    if (dtype_in != Dtype::NONE) {
        // `in` is already an array.
        if (dt == Dtype::NONE)
            dt = Dtype(std::max(int(dtype_in), int(dtype_min)));
        size_t *in_shape;
        reinterpret_cast<Array_base*>(in)->ndim_shape(&ndim, &in_shape);
        if (ndim == 2) {
            if (dt == dtype_in)
                Py_INCREF(result = in);
            else
                result = convert_array(dt, in, dtype_in);
        } else if (ndim < 2) {
            shape[0] = 1;
            shape[1] = (ndim == 0) ? 1 : in_shape[0];
            result = convert_array(dt, in, dtype_in, 2, shape);
        } else {
            PyErr_SetString(PyExc_ValueError, "Matrix must be 2-dimensional.");
            result = 0;
        }
    } else {
        // `in` is not an array.
        if (examine_arraylike(in, &ndim, shape, seqs,
                              dt == Dtype::NONE ? &dt : 0) == -1) return 0;
        if (ndim != 2) {
            if (ndim > 2) {
                PyErr_SetString(PyExc_ValueError,
                                "Matrix must be 2-dimensional.");
                return 0;
            }
            shape[1] = (ndim == 0) ? 1 : shape[0];
            shape[0] = 1;
        }
        if (*dtype == Dtype::NONE) {
            // No specific dtype has been requested.  It will be determined by
            // the input.
            PyObject *seqs_copy[max_ndim];
            for (int d = 0; d < ndim; ++d) Py_INCREF(seqs_copy[d] = seqs[d]);
            if (dt == Dtype::NONE) {
                assert(shape[1] == 0);
                dt = default_dtype;
            }
            if (int(dt) < int(dtype_min)) dt = dtype_min;
            while (true) {
                result = make_and_readin_array_dtable[int(dt)](
                    in, ndim, 2, shape, seqs, true);
                if (result) break;
                dt = Dtype(int(dt) + 1);
                if (dt == Dtype::NONE) {
                    result = 0;
                    break;
                }
                PyErr_Clear();
                for (int d = 0; d < ndim; ++d)
                    Py_INCREF(seqs[d] = seqs_copy[d]);
            }
            for (int d = 0; d < ndim; ++d) Py_DECREF(seqs_copy[d]);
        } else {
            // A specific dtype has been requested.
            result = make_and_readin_array_dtable[int(dt)](
                in, ndim, 2, shape, seqs, false);
        }
    }
    *dtype = dt;
    return result;
}

int coerce_to_arrays(PyObject **a_, PyObject **b_, Dtype *coerced_dtype)
{
    PyObject *a = *a_, *b = *b_;

    // Make sure a and b are tinyarrays.
    Dtype dtype_a = Dtype::NONE, dtype_b = Dtype::NONE, dtype;
    a = array_from_arraylike(a, &dtype_a);
    if (!a) return -1;
    b = array_from_arraylike(b, &dtype_b, dtype_a);
    if (!b) {
        Py_DECREF(a);
        return -1;
    }

    // Promote to a common dtype.
    dtype = Dtype(std::max(int(dtype_a), int(dtype_b)));
    if (dtype_a != dtype) {
        PyObject *temp = convert_array(dtype, a, dtype_a);
        if (temp == 0) goto fail;
        Py_DECREF(a);
        a = temp;
    } else if (dtype_b != dtype) {
        PyObject *temp = convert_array(dtype, b, dtype_b);
        if (temp == 0) goto fail;
        Py_DECREF(b);
        b = temp;
    }

    // Success
    *a_ = a; *b_ = b; *coerced_dtype = dtype;
    return 0;

fail:
    Py_DECREF(a);
    Py_DECREF(b);
    return -1;
}

template <typename T>
PyObject *transpose(PyObject *in_)
{
    assert(Array<T>::check_exact(in_)); Array<T> *in = (Array<T>*)in_;

    int ndim;
    ptrdiff_t hops[max_ndim];
    size_t *shape_in, shape_out[max_ndim], stride = 1;
    in->ndim_shape(&ndim, &shape_in);
    if (ndim == 0) {
        Py_INCREF(in_);
        return in_;
    }
    for (int id = ndim - 1, od = 0; id >= 0; --id, ++od) {
        size_t ext = shape_in[id];
        shape_out[od] = ext;
        hops[od] = stride;
        stride *= ext;
    }
    for (int d = 1; d < ndim; ++d) hops[d - 1] -= hops[d] * shape_out[d];
    Array<T> *out = Array<T>::make(ndim, shape_out);
    if (!out) return 0;
    T *src = in->data(), *dest = out->data();

    int d = 0;
    size_t i[max_ndim];
    --ndim;
    i[0] = shape_out[0];
    while (true) {
        if (i[d]) {
            --i[d];
            if (d == ndim) {
                *dest++ = *src;
                src += hops[d];
            } else {
                ++d;
                i[d] = shape_out[d];
            }
        } else {
            if (d == 0) return (PyObject*)out;
            --d;
            src += hops[d];
        }
    }
}

template <typename T>
Array<T> *Array<T>::make(int ndim, size_t size)
{
    Py_ssize_t ob_size = size;
    assert(ndim != 0 || size == 1);
    if (ndim > 1)
        ob_size += (ndim * sizeof(size_t) + sizeof(T) - 1) / sizeof(T);
    Array *result = PyObject_NewVar(Array<T>, &Array<T>::pytype, ob_size);
    if (!result) return 0;
    if (ndim > 1)
        result->ob_base.ob_size = -ndim;
    else if (ndim == 0)
        result->ob_base.ob_size = -1;
    return result;
}

template <typename T>
Array<T> *Array<T>::make(int ndim, const size_t *shape, size_t *sizep)
{
    // Check shape and calculate size, the total number of elements.
    size_t size = 1;
    // `reserve' allows to detect overflow.
    size_t reserve = PY_SSIZE_T_MAX;
    for (int d = 0; d < ndim; ++d) {
        size_t elem = shape[d];
        if (elem > reserve) {
            PyErr_SetString(PyExc_ValueError, "Array would be too big.");
            return 0;
        }
        size *= elem;
        if (elem) reserve /= elem;
    }

    Array *result = Array<T>::make(ndim, size);
    if (!result) return 0;
    size_t *result_shape;
    result->ndim_shape(0, &result_shape);
    for (int d = 0; d < ndim; ++d) result_shape[d] = shape[d];

    if (sizep) *sizep = size;
    return result;
}

// **************** Type object ****************

template <>
const char *Array<long>::pyname = "tinyarray.ndarray_int";
template <>
const char *Array<double>::pyname = "tinyarray.ndarray_float";
template <>
const char *Array<Complex>::pyname = "tinyarray.ndarray_complex";

template <typename T>
PySequenceMethods Array<T>::as_sequence = {
    (lenfunc)len,                // sq_length
    0,                           // sq_concat
    0,                           // sq_repeat
    (ssizeargfunc)seq_getitem<T> // sq_item
};

template <typename T>
PyMappingMethods Array<T>::as_mapping = {
    (lenfunc)len,               // mp_length
    (binaryfunc)getitem<T>      // mp_subscript
};

template <typename T>
PyBufferProcs Array<T>::as_buffer = {
    // We only support the new buffer protocol.
    0,                          // bf_getreadbuffer
    0,                          // bf_getwritebuffer
    0,                          // bf_getsegcount
    0,                          // bf_getcharbuffer
    (getbufferproc)getbuffer<T> // bf_getbuffer
};

template <typename T>
PyMethodDef Array<T>::methods[] = {
    {"transpose", (PyCFunction)transpose<T>, METH_NOARGS},
    {"conjugate", (PyCFunction)apply_unary_ufunc<Conjugate<T> >, METH_NOARGS},
    {0, 0}                      // Sentinel
};

template <typename T>
PyTypeObject Array<T>::pytype = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    pyname,
    sizeof(Array<T>) - sizeof(T),   // tp_basicsize
    sizeof(T),                      // tp_itemsize
    (destructor)PyObject_Del,       // tp_dealloc
    0,                              // tp_print
    0,                              // tp_getattr
    0,                              // tp_setattr
    0,                              // tp_compare
    (reprfunc)repr<T>,              // tp_repr
    &as_number,                     // tp_as_number
    &as_sequence,                   // tp_as_sequence
    &as_mapping,                    // tp_as_mapping
    (hashfunc)hash<T>,              // tp_hash
    0,                              // tp_call
    (reprfunc)str<T>,               // tp_str
    PyObject_GenericGetAttr,        // tp_getattro
    0,                              // tp_setattro
    &as_buffer,                     // tp_as_buffer
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_HAVE_NEWBUFFER |
    Py_TPFLAGS_CHECKTYPES,          // tp_flags
    doc,                            // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear
    (richcmpfunc)richcompare,       // tp_richcompare
    0,                              // tp_weaklistoffset
    (getiterfunc)Array_iter<T>::make, // tp_iter
    0,                              // tp_iternext
    methods,                        // tp_methods
    0,                              // tp_members
    getset,                         // tp_getset
    0,                              // tp_base
    0,                              // tp_dict
    0,                              // tp_descr_get
    0,                              // tp_descr_set
    0,                              // tp_dictoffset
    0,                              // tp_init
    0,                              // tp_alloc
    0,                              // tp_new
    PyObject_Del                    // tp_free
};

// **************** Explicit instantiations ****************
template class Array<long>;
template class Array<double>;
template class Array<Complex>;

template PyObject *transpose<long>(PyObject*);
template PyObject *transpose<double>(PyObject*);
template PyObject *transpose<Complex>(PyObject*);
