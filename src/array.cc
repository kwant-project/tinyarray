#include <Python.h>
#include <limits>
#include <cassert>
#include "array.hh"

int dtype_converter(const PyObject *ob, Dtype *dtype)
{
    if (ob == Py_None) {
        *dtype = default_dtype;
    } else if (ob == (PyObject *)(&PyFloat_Type)) {
        *dtype = Dtype::DOUBLE;
    } else if (ob == (PyObject *)(&PyComplex_Type)) {
        *dtype = Dtype::COMPLEX;
    } else if (ob == (PyObject *)(&PyInt_Type) ||
               ob == (PyObject *)(&PyLong_Type)) {
        *dtype = Dtype::LONG;
    } else {
        PyErr_SetString(PyExc_TypeError, "Invalid dtype.");
        return 0;
    }
    return 1;
}

static Py_ssize_t load_seq_as_long(PyObject *obj, long *out, int maxlen)
{
    assert(maxlen >= 1);
    Py_ssize_t len;
    if (PySequence_Check(obj)) {
        obj = PySequence_Fast(obj, "Bug in tinyarray, load_seq_as_long");
        if (!obj) return -1;
        len = PySequence_Fast_GET_SIZE(obj);
        if (len > maxlen) {
            PyErr_Format(PyExc_ValueError, "Sequence too long."
                         " Maximum length is %d.", maxlen);
            Py_DECREF(obj);
            return -1;
        }
        for (PyObject **p = PySequence_Fast_ITEMS(obj), **e = p + len; p < e;
             ++p, ++out) {
            *out = PyInt_AsLong(*p);
            if (*out == -1 and PyErr_Occurred()) {
                Py_DECREF(obj);
                return -1;
            }
        }
    } else {
        len = 1;
        *out = PyInt_AsLong(obj);
        if (*out == -1 and PyErr_Occurred()) return -1;
    }
    return len;
}

Array *array_make(PyObject *shape_obj, Dtype dtype)
{
    // Read shape.
    long shape[max_ndim];
    int ndim = load_seq_as_long(shape_obj, shape, max_ndim);
    if (ndim == -1) return 0;

    // Check shape and calculate ob_size, the total number of elements.
    Py_ssize_t ob_size = 1;
#ifndef NO_OVERFLOW_DANGER
    // `reserve' allows to detect overflow.
    Py_ssize_t reserve = PY_SSIZE_T_MAX;
#endif
    for (int d = 0; d < ndim; ++d) {
        long elem = shape[d];
        if (elem < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "Negative dimensions are not allowed.");
            return 0;
        }
        if (elem > std::numeric_limits<Shape_t>::max()) {
            PyErr_Format(PyExc_ValueError, "shape[%d] is too big.", d);
            return 0;
        }
#ifndef NO_OVERFLOW_DANGER
        if (elem > reserve) {
            PyErr_SetString(PyExc_ValueError, "Array would be too big.");
            return 0;
        }
#endif
        ob_size *= elem;
#ifndef NO_OVERFLOW_DANGER
        if (elem) reserve /= elem;
#endif
    }

    if (dtype != Dtype::LONG) {
        PyErr_SetString(PyExc_ValueError, "dtype must be int for now");
        return 0;
    }

    Array *result = PyObject_NewVar(Array, &Array_type, ob_size);
    if (!result) return 0;
    result->buffer_internal = 0;
    result->ndim = ndim;
    for (int d = 0; d < ndim; ++d) result->shape[d] = shape[d];

    return result;
}

static void dealloc(Array *self)
{
    if (self->buffer_internal) PyMem_Free(self->buffer_internal);
    PyObject_Del(self);
}

static PyObject *repr(Array *self)
{
    Py_ssize_t n = Py_SIZE(self);
    char buf[30 * n + 200];
    char *s = buf;
    s += sprintf(s, "< ");
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (i)
            s += sprintf(s, ", %ld", self->ob_item[i]);
        else
            s += sprintf(s, "%ld", self->ob_item[i]);
    }
    s += sprintf(s, " shape=(");
    for (int d = 0; d < self->ndim; ++d)
        if (d)
            s += sprintf(s, ", %d", self->shape[d]);
        else
            s += sprintf(s, "%d", self->shape[d]);
    s += sprintf(s, ") >");
    return PyString_FromString(buf);
}

PyDoc_STRVAR(doc, "array docstring: to be written\n");

static Py_ssize_t len(Array *self)
{
    if (self->ndim == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object.");
        return -1;
    }
    return self->shape[0];
}

static Py_ssize_t index_from_key(Array *self, PyObject *key)
{
    long indices[max_ndim];
    int ndim = load_seq_as_long(key, indices, max_ndim);
    if (ndim == -1) {
        PyErr_SetString(PyExc_IndexError,
                        "Invalid index.");
        return -1;
    }
    if (ndim != self->ndim) {
        PyErr_SetString(PyExc_IndexError, "Number of indices "
                        "must be equal to number of dimensions.");
        return -1;
    }

    Py_ssize_t s = self->shape[0];
    Py_ssize_t index = indices[0];
    if (index < 0) index += s;
    if (index < 0 || index >= s) {
        PyErr_Format(PyExc_IndexError, "Index %ld out of range "
                     "(-%ld <= index < %ld) in dimension 0.",
                     indices[0], s, s);
        return -1;
    }
    for (int d = 1; d < ndim; ++d) {
        s = self->shape[d];
        Py_ssize_t i = indices[d];
        if (i < 0) i += s;
        if (i < 0 || i >= s) {
            PyErr_Format(PyExc_IndexError, "Index %ld out of range "
                         "(-%ld <= index < %ld) in dimension %d.",
                         indices[d], s, s, d);
            return -1;
        }
        index *= s;
        index += i;
    }
    return index;
}

static PyObject *getitem(Array *self, PyObject *key)
{
    if (PySlice_Check(key)) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "slices are not implemented");
        return 0;
    } else {
        Py_ssize_t index = index_from_key(self, key);
        if (index == -1) return 0;
        return PyInt_FromLong(self->ob_item[index]);
    }
}

// The memoryview implementation is broken in Python <3.3.  See
// http://bugs.python.org/issue7433 and http://bugs.python.org/issue10181.
// This means that the function bf_releasebuffer is useless for releasing
// ressources as it always gets the same copy of the view, even when called
// several times.  To work around this, we allocate the needed memory in
// getbuffer and store a pointer to it in the array data structure.  The memory
// gets freed only when the array is deleted.
//
// Once we require Python 3.3 we can return to a sane way of handling things,
// restore bf_releasebuffer and get rid of the buffer_internal field in struct
// Array.
int getbuffer(Array *self, Py_buffer *view, int flags)
{
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

    view->buf = self->ob_item;
    view->itemsize = sizeof(long);
    view->len = Py_SIZE(self) * view->itemsize;
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
        view->format = (char*)"l";
    else
        view->format = 0;
    if ((flags & PyBUF_ND) == PyBUF_ND) {
        // From the documentation it's not clear whether it is allowed not to
        // set strides to NULL (for C continuous arrays), but it works!
        // However, there is a bug in current numpy
        // (http://projects.scipy.org/numpy/ticket/2197) which requires strides
        // if view->len == 0.  Once this bug is fixed, we can remove the code
        // which sets strides.
        Py_ssize_t *shape, *strides;
        if (!self->buffer_internal) {
            if (view->len == 0 && (flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
                self->buffer_internal = shape = (Py_ssize_t*)
                    PyMem_Malloc(2 * self->ndim * sizeof(Py_ssize_t));
                strides = shape + self->ndim;
            } else {
                self->buffer_internal = shape =
                    (Py_ssize_t*)PyMem_Malloc(self->ndim * sizeof(Py_ssize_t));
                strides = 0;
            }
            if (!shape) {
                PyErr_SetNone(PyExc_MemoryError);
                goto fail;
            }

            for (int d = 0; d < self->ndim; ++d) shape[d] = self->shape[d];
            if (strides) {
                Py_ssize_t l = view->len;
                for (int d = 0; d < self->ndim; ++d) {
                    if (shape[d]) l /= shape[d];
                    strides[d] = l;
                }
            }
        } else {
            shape = self->buffer_internal;
            if (view->len == 0 && (flags & PyBUF_STRIDES) == PyBUF_STRIDES)
                strides = shape + self->ndim;
            else
                strides = 0;
        }
        view->ndim = self->ndim;
        view->shape = shape;
        view->strides = strides;
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

// This is modelled on Python's tuple hash function.
static long hash(Array *self)
{
    long mult = 1000003, r = 0x345678;
    Py_ssize_t len = Py_SIZE(self);
    long *p = self->ob_item;
    while (--len >= 0) {
        r = (r ^ *p++) * mult;
        mult += long(82520 + len + len);
    }
    r += 97531;
    if (r == -1) r = -2;
    return r;
}

static PyMappingMethods as_mapping = {
    (lenfunc)len,
    (binaryfunc)getitem,
    0
};

static PyBufferProcs as_buffer = {
    // We only support the new buffer protocol.
    0,                        // bf_getreadbuffer
    0,                        // bf_getwritebuffer
    0,                        // bf_getsegcount
    0,                        // bf_getcharbuffer
    (getbufferproc)getbuffer, // bf_getbuffer
    0                         // bf_releasebuffer
};

PyTypeObject Array_type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "tinyarray.ndarray",
    sizeof(Array) - sizeof(long),   // tp_basicsize
    sizeof(long),                   // tp_itemsize
    (destructor)dealloc,            // tp_dealloc
    0,                              // tp_print
    0,                              // tp_getattr
    0,                              // tp_setattr
    0,                              // tp_compare
    (reprfunc)repr,                 // tp_repr
    0,                              // tp_as_number

    0,                              // tp_as_sequence
    &as_mapping,                    // tp_as_mapping

    (hashfunc)hash,                 // tp_hash

    0,                              // tp_call
    0,                              // tp_str
    PyObject_GenericGetAttr,        // tp_getattro
    0,                              // tp_setattro
    &as_buffer,                     // tp_as_buffer
    Py_TPFLAGS_DEFAULT
    | Py_TPFLAGS_HAVE_NEWBUFFER,    // tp_flags
    doc,                            // tp_doc
    0,                              // tp_traverse
    0,                              // tp_clear

    // richcompare,                    // tp_richcompare
    0,                              // tp_richcompare

    0,                              // tp_weaklistoffset

    // iter,                          // tp_iter
    0,                              // tp_iter

    0,                              // tp_iternext
    0,                              // tp_methods
    0,                              // tp_members
    0,                              // tp_getset
    0,                              // tp_base
    0,                              // tp_dict
    0,                              // tp_descr_get
    0,                              // tp_descr_set
    0,                              // tp_dictoffset
    0,                              // tp_init
    0,                              // tp_alloc
    0,                              // tp_new
    PyObject_Del,                   // tp_free
};
