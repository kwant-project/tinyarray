Dtype dtype_of_buffer(Py_buffer *view)
{
    char *fmt = view->format;


        fmt++;
        fmt++;
        fmt++;
        }
        fmt++;
    }


    return dtype;
}

int examine_buffer(PyObject *in, Py_buffer *view, Dtype *dtype)
{
    memset(view, 0, sizeof(Py_buffer));
}

template<typename T>
T (*get_buffer_converter_complex(const char *fmt))(const void *);

template<>
{
    PyErr_Format(PyExc_TypeError, "Complex cannot be cast to int.");

    return 0;
}

template<>
{
    // complex can only be cast to complex
    PyErr_Format(PyExc_TypeError, "Complex cannot be cast to float.");

    return 0;
}

template<>
Complex (*get_buffer_converter_complex(const char *fmt))(const void *)
{
    switch(*(fmt + 1)){
    case 'f':
        return number_from_ptr<Complex,  std::complex<float> >;
    case 'd':
        return number_from_ptr<Complex, std::complex<double> >;
    case 'g':
        return number_from_ptr<Complex, std::complex<long double> >;
    }

    return 0;
}

template<typename T>
T (*get_buffer_converter(Py_buffer *view))(const void *)
{
    // currently, we only understand native endianness and alignment
    char *fmt = view->format;

        fmt++;
    }

    switch(*fmt) {
    case 'c':
        return number_from_ptr<T, char>;
    case 'b':
        return number_from_ptr<T, signed char>;
    case 'B':
        return number_from_ptr<T, unsigned char>;
    case '?':
        return number_from_ptr<T, bool>;
    case 'h':
        return number_from_ptr<T, short>;
    case 'H':
        return number_from_ptr<T, unsigned short>;
    case 'i':
        return number_from_ptr<T, int>;
    case 'I':
        return number_from_ptr<T, unsigned int>;
    case 'l':
        return number_from_ptr<T, long>;
    case 'L':
        return number_from_ptr<T, unsigned long>;
    case 'q':
        return number_from_ptr<T, long long>;
    case 'Q':
        return number_from_ptr<T, unsigned long long>;
    case 'f':
        return number_from_ptr<T, float>;
    case 'd':
        return number_from_ptr<T, double>;
    case 'g':
        return number_from_ptr<T, long double>;
    case 'Z':
        return get_buffer_converter_complex<T>(fmt);
    }

    return 0;
}

template<typename T>
{
    T (*number_from_ptr)(const void *) = get_buffer_converter<T>(view);

        *dest = (*number_from_ptr)(view->buf);
        else return 0;
    }

        indices[i] = 0;
    }

        while(indices[0] < view->shape[0]) {
            char *pointer = (char*)view->buf;
            for (int i = 0; i < view->ndim; i++) {
                pointer += view->strides[i] * indices[i];
                if (view->suboffsets[i] >=0 ) {
                    pointer = *((char**)pointer) + view->suboffsets[i];
                }
            }

            *dest++ = (*number_from_ptr)(pointer);

            indices[view->ndim-1]++;

                indices[i-1]++;
                indices[i] = 0;
            }
        }
        char *ptr = (char *)view->buf;

        while(indices[0] < view->shape[0]) {
            *dest++ = (*number_from_ptr)(ptr);

            indices[view->ndim-1] ++;
            ptr += view->strides[view->ndim-1];

                indices[i-1]++;
                ptr += view->strides[i-1];
                indices[i] = 0;
                ptr -= view->strides[i] * view->shape[i];
            }
        }
        char *end = (char *)view->buf + view->len;
        char *p = (char *)view->buf;
        while(p < end) {
            *dest++ = (*number_from_ptr)(p);

            p += view->itemsize;
        }
    }

    return 0;
}

template <typename T>
PyObject *make_and_readin_buffer(Py_buffer *view, int ndim_out,
{
    Array<T> *result = Array<T>::make(ndim_out, shape_out);
#ifndef NDEBUG
    for (int d = 0, e = ndim_out - view->ndim; d < e; ++d)
        assert(shape_out[d] == 1);
#endif
    if (result == 0) return 0;
        Py_DECREF(result);
        return 0;
    }
    return (PyObject*)result;
}

PyObject *(*make_and_readin_buffer_dtable[])(

PyObject *repr(PyObject *obj)
    Array<T> *self = reinterpret_cast<Array<T> *>(obj);
PyObject *str(PyObject *obj)
    Array<T> *self = reinterpret_cast<Array<T> *>(obj);
    int res = load_index_seq_as_long(key, indices, max_ndim);
    if (res != ndim) {
PyObject *getitem(PyObject *obj, PyObject *key)
    Array<T> *self = reinterpret_cast<Array<T> *>(obj);

PyObject *seq_getitem(PyObject *obj, Py_ssize_t index)
    Array<T> *self = reinterpret_cast<Array<T> *>(obj);
int getbuffer(PyObject *obj, Py_buffer *view, int flags)
    Array<T> *self = reinterpret_cast<Array<T> *>(obj);
long hash(PyObject *obj)
    Array<T> *self = reinterpret_cast<Array<T> *>(obj);
template PyObject *repr<long>(PyObject*);
template PyObject *repr<double>(PyObject*);
template PyObject *repr<Complex>(PyObject*);
template PyObject *str<long>(PyObject*);
template PyObject *str<double>(PyObject*);
template PyObject *str<Complex>(PyObject*);
template long hash<long>(PyObject*);
template long hash<double>(PyObject*);
template long hash<Complex>(PyObject*);
template int getbuffer<long>(PyObject*, Py_buffer*, int);
template int getbuffer<double>(PyObject*, Py_buffer*, int);
template int getbuffer<Complex>(PyObject*, Py_buffer*, int);
template PyObject *getitem<long>(PyObject*, PyObject*);
template PyObject *getitem<double>(PyObject*, PyObject*);
template PyObject *getitem<Complex>(PyObject*, PyObject*);
template PyObject *seq_getitem<long>(PyObject*, Py_ssize_t);
template PyObject *seq_getitem<double>(PyObject*, Py_ssize_t);
template PyObject *seq_getitem<Complex>(PyObject*, Py_ssize_t);
int load_index_seq_as_long(PyObject *obj, long *out, int maxlen)
    int len;
        Py_ssize_t long_len = PySequence_Fast_GET_SIZE(obj);
        if (long_len > maxlen) {
                         " Maximum length is %d.", maxlen);
        len = static_cast<int>(long_len);
int load_index_seq_as_ulong(PyObject *obj, unsigned long *uout,
                            int maxlen, const char *errmsg)
    int len = load_index_seq_as_long(obj, out, maxlen);
    for (int i = 0; i < len; ++i)

        *dtype = dt;
        return result;
    } else {

        // Try if buffer interface is supported
        Py_buffer view;
                shape[i] = view.shape[i];
            PyBuffer_Release(&view);

            *dtype = dt;
            return result;

        if (examine_arraylike(in, &ndim, shape, seqs,
                              find_type ? &dt : 0) == 0) {
                PyObject *seqs_copy[max_ndim];
                for (int d = 0; d < ndim; ++d)
                    Py_INCREF(seqs_copy[d] = seqs[d]);
                    assert(shape[ndim - 1] == 0);
                    dt = default_dtype;
                }
                if (int(dt) < int(dtype_min)) dt = dtype_min;
                while (true) {
                    result = make_and_readin_array_dtable[int(dt)](
                        in, ndim, ndim, shape, seqs, true);
                    if (result) break;
                    dt = Dtype(int(dt) + 1);
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
                    in, ndim, ndim, shape, seqs, false);

            *dtype = dt;
            return result;

    return 0;

        *dtype = dt;
        return result;


        // Try if buffer interface is supported
        Py_buffer view;
                shape[i] = view.shape[i];
            if (view.ndim != 2) {
                if (view.ndim > 2) {
                    PyErr_SetString(PyExc_ValueError,
                                    "Matrix must be 2-dimensional.");
                    return 0;
                }
                shape[1] = (view.ndim == 0) ? 1 : shape[0];
                shape[0] = 1;
            }
            PyBuffer_Release(&view);

            *dtype = dt;
            return result;

        if (examine_arraylike(in, &ndim, shape, seqs,
            if (ndim != 2) {
                if (ndim > 2) {
                    PyErr_SetString(PyExc_ValueError,
                                    "Matrix must be 2-dimensional.");
                    return 0;
                }
                shape[1] = (ndim == 0) ? 1 : shape[0];
                shape[0] = 1;
            if (find_type) {
                // No specific dtype has been requested.  It will be
                // determined by the input.
                PyObject *seqs_copy[max_ndim];
                for (int d = 0; d < ndim; ++d)
                    Py_INCREF(seqs_copy[d] = seqs[d]);
                    assert(shape[1] == 0);
                    dt = default_dtype;
                if (int(dt) < int(dtype_min)) dt = dtype_min;
                while (true) {
                    result = make_and_readin_array_dtable[int(dt)](
                        in, ndim, 2, shape, seqs, true);
                    if (result) break;
                    dt = Dtype(int(dt) + 1);
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

            *dtype = dt;
            return result;

    return 0;
PyObject *transpose(PyObject *in_, PyObject *)
PyObject *conjugate(PyObject *in_, PyObject *)
{
  return apply_unary_ufunc<Conjugate<T> >(in_);
}

template <typename T>
template <typename T>
PyObject *reduce(PyObject *self_, PyObject*)
{

    int ndim;
    size_t *shape;
    self->ndim_shape(&ndim, &shape);
    size_t size_in_bytes = calc_size(ndim, shape) * sizeof(T);

    for (int i=0; i < ndim; ++i)

}

    seq_getitem<T> // sq_item
    getitem<T>      // mp_subscript
    getbuffer<T> // bf_getbuffer
    {"conjugate", (PyCFunction)conjugate<T>, METH_NOARGS},
    repr<T>,                        // tp_repr
    hash<T>,                        // tp_hash
    str<T>,                         // tp_str
template PyObject *transpose<long>(PyObject*, PyObject*);
template PyObject *transpose<double>(PyObject*, PyObject*);
template PyObject *transpose<Complex>(PyObject*, PyObject*);

template PyObject *conjugate<long>(PyObject*, PyObject*);
template PyObject *conjugate<double>(PyObject*, PyObject*);
template PyObject *conjugate<Complex>(PyObject*, PyObject*);

template PyObject *reduce<long>(PyObject*, PyObject*);
template PyObject *reduce<double>(PyObject*, PyObject*);
template PyObject *reduce<Complex>(PyObject*, PyObject*);
