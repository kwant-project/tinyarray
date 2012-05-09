#include <Python.h>
#include "array.hh"
#include "functions.hh"

extern "C"
void inittinyarray()
{
    if (PyType_Ready(&Array_type) < 0) return;

    PyObject* m = Py_InitModule("tinyarray", functions);

    Py_INCREF(&Array_type);
    PyModule_AddObject(m, "ndarray", (PyObject *)&Array_type);
}
