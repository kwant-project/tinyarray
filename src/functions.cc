{
    PyObject *pyshape;
    Dtype dtype = default_dtype;
    if (!PyArg_ParseTuple(args, "O|O&", &pyshape, dtype_converter, &dtype))
        return 0;

    unsigned long shape_as_ulong[max_ndim];
    int ndim = load_index_seq_as_ulong(pyshape, shape_as_ulong, max_ndim,
                                       "Negative dimensions are not allowed.");
    if (ndim == -1) return 0;

    size_t shape[max_ndim];
    for (int d = 0; d < ndim; ++d) shape[d] = shape_as_ulong[d];
}

PyObject *(*transpose_dtable[])(PyObject*, PyObject *) =
  DTYPE_DISPATCH(transpose);
    return transpose_dtable[int(dtype)](a, 0);
