            if (ndim) *ndim = static_cast<int>(-ob_size);
int load_index_seq_as_long(PyObject *obj, long *out, int maxlen);
int load_index_seq_as_ulong(PyObject *obj, unsigned long *uout,
                            int maxlen, const char *errmsg = 0);
template <typename T> PyObject *transpose(PyObject *in, PyObject *dummy);
