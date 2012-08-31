#include <Python.h>
#include <cstddef>
#include <sstream>
#include <limits>
#include "array.hh"
#include "functions.hh"
#include "conversion.hh"

template <>
const char *Array<long>::pyformat = "l";
template <>
const char *Array<double>::pyformat = "d";
template <>
const char *Array<Complex>::pyformat = "Zd";

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
PyObject *make_and_readin_array(PyObject *src, int ndim,
                                const size_t *shape, PyObject **seqs,
                                bool exact)
{
    Array<T> *result = Array<T>::make(ndim, shape);
    if (result == 0) return 0;
    if (readin_arraylike<T>(result->data(), ndim, shape, src, seqs, exact)
        == -1) {
        Py_DECREF(result);
        return 0;
    }
    return (PyObject*)result;
}

PyObject *(*make_and_readin_array_dtable[])(
    PyObject*, int, const size_t*, PyObject**, bool) =
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

// The hash functions are modelled on Python's.  It is important that a == b =>
// hash(a) == hash(b), otherwise there will be problems with dictionaries etc.

long hash(long x)
{
    return x;
}

long hash(double x)
{
    double intpart, fractpart;
    fractpart = modf(x, &intpart);
    if (fractpart == 0 &&
        intpart >= std::numeric_limits<long>::min() &&
        intpart <= std::numeric_limits<long>::max()) {
        // This must return the same hash as an equal long.
        return long(intpart);
    }
    // Can't represent the number as a long: interpret its bits as hash!
    static_assert(sizeof(double) >= sizeof(size_t),
                  "hash(double) has to be adopted for this machine.");
    return long(*reinterpret_cast<size_t*>(&x));
}

long hash(Complex x)
{
    // x.imag == 0  =>  hash(x.imag) == 0  =>  hash(x) == hash(x.real)
    return hash(x.real()) + 1000003 * hash(x.imag());
}

template <typename T>
long hash(Array<T> *self)
{
    long mult = 1000003, r = 0x345678;
    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    Py_ssize_t size = calc_size(ndim, shape);
    T *p = self->data();
    while (--size >= 0) {
        r = (r ^ hash(*p++)) * mult;
        mult += long(82520 + size + size);
    }
    r += 97531;
    if (r == -1) r = -2;
    return r;
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

// If *dtype == Dtype::NONE the simplest fitting dtype for the array will be
// used and written back to *dtype.  Any other value of *dtype requests an
// array of a given dtype.
PyObject *array_from_arraylike(PyObject *src, Dtype *dtype)
{
    int ndim;
    size_t shape[max_ndim];
    PyObject *seqs[max_ndim];
    if (*dtype == Dtype::NONE) {
        // No specific dtype has been requested.  It will be determined by the
        // input.
        if (examine_arraylike(src, &ndim, shape, seqs, dtype) == -1)
            return 0;
        if (*dtype == Dtype::NONE) {
            assert(shape[ndim - 1] == 0);
            *dtype = default_dtype;
        }
        while (true) {
            PyObject *result = make_and_readin_array_dtable[int(*dtype)](
                src, ndim, shape, seqs, true);
            if (result) return result;
            PyErr_Clear();
            *dtype = Dtype(int(*dtype) + 1);
            if (*dtype == Dtype::NONE) {
                PyErr_SetString(PyExc_TypeError, "Expecting a number.");
                return 0;
            }
            // We have to re-execute examine_arraylike again to rebuild seqs
            // which have been closed.
#ifdef NDEBUG
            examine_arraylike(src, &ndim, shape, seqs, 0);
#else
            assert(examine_arraylike(src, &ndim, shape, seqs, 0) == 0);
#endif
        }
    } else {
        // A specific dtype has been requested.
        if (examine_arraylike(src, &ndim, shape, seqs, 0) == -1)
            return 0;
        return make_and_readin_array_dtable[int(*dtype)](
            src, ndim, shape, seqs, false);
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
    0/*&as_number*/,                     // tp_as_number

    &as_sequence,                   // tp_as_sequence
    &as_mapping,                    // tp_as_mapping

    (hashfunc)hash<T>,              // tp_hash

    0,                              // tp_call
    (reprfunc)str<T>,               // tp_str
    PyObject_GenericGetAttr,        // tp_getattro
    0,                              // tp_setattro
    &as_buffer,           // tp_as_buffer
    Py_TPFLAGS_DEFAULT
    | Py_TPFLAGS_HAVE_NEWBUFFER,    // tp_flags
    doc,                            // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear

    // richcompare,                    // tp_richcompare
    0,                              // tp_richcompare

    0,                              // tp_weaklistoffset

    (getiterfunc)Array_iter<T>::make, // tp_iter

    0,                              // tp_iternext
    0,                              // tp_methods
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
