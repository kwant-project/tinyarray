PyObject *(*transpose_dtable[])(PyObject*, PyObject *) =
  DTYPE_DISPATCH(transpose);
    return transpose_dtable[int(dtype)](a, 0);
