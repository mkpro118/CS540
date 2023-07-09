#include <Python.h>

#ifndef _PY_LONG_CHECK_
#define _PY_LONG_CHECK_(x, t) {\
    if (!PyLong_Check(x)) { \
        free(t); \
        PyErr_SetString(PyExc_TypeError, "Expected tuple of integers"); \
        return NULL; \
    } \
}
#endif

static PyObject* build_bigram(PyObject* self, PyObject* args) {
    PyObject* idx_tuple;
    int size;

    if (!PyArg_ParseTuple(args, "O!i", &PyTuple_Type, &idx_tuple, &size)) {
        return NULL;
    }

    Py_ssize_t tuple_size = PyTuple_Size(idx_tuple);

    int** table = malloc(sizeof(int*) * size);
    for (int i = 0; i < size; ++i){
        table[i] = calloc(size, sizeof(int));
    }

    // Start from 1, since we cannot create
    // a transition probability matrix for the first
    // letter, since there are no preceeding letters

    int prev;
    int curr;
    PyObject* item;

    item = PyTuple_GetItem(idx_tuple, 0);
    _PY_LONG_CHECK_(item, table);

    prev = (int) PyLong_AsLong(item);

    for (Py_ssize_t i = 1; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_(item, table);
        curr = (int) PyLong_AsLong(item);

        table[prev][curr] += 1;
        prev = curr;
    }

    Py_DECREF(idx_tuple);

    PyObject* ret_tuple = PyTuple_New(size);

    for (int i = 0; i < size; i++) {
        PyObject* inner_tuple = PyTuple_New(size);
        for (int j = 0; j < size; j++) {
            PyObject* value = PyLong_FromLong(table[i][j]);
            PyTuple_SetItem(inner_tuple, j, value);
        }
        PyTuple_SetItem(ret_tuple, i, inner_tuple);
    }

    Py_INCREF(ret_tuple);
    return ret_tuple;
}

static PyObject* build_trigram(PyObject* self, PyObject* args) {
    PyObject* idx_tuple;
    int size;

    if (!PyArg_ParseTuple(args, "O!i", &PyTuple_Type, &idx_tuple, &size)) {
        return NULL;
    }

    Py_ssize_t tuple_size = PyTuple_Size(idx_tuple);

    int*** table = malloc(sizeof(int**) * size);
    for (int i = 0; i < size; ++i){
        table[i] = malloc(size, sizeof(int*));
        for (int j = 0; j < size; j++) {
            table[i][j] = calloc(size, sizeof(int));
        }
    }

    // Start from 1, since we cannot create
    // a transition probability matrix for the first
    // letter, since there are no preceeding letters

    int prev1, prev2;
    int curr;
    PyObject* item;

    item = PyTuple_GetItem(idx_tuple, 0);
    _PY_LONG_CHECK_(item, table);

    prev1 = (int) PyLong_AsLong(item);

    item = PyTuple_GetItem(idx_tuple, 1);
    _PY_LONG_CHECK_(item, table);
    prev2 = (int) PyLong_AsLong(item);

    for (Py_ssize_t i = 2; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_(item, table);
        curr = (int) PyLong_AsLong(item);

        table[prev1][prev2][curr] += 1;
        prev1 = prev2;
        prev2 = curr;
    }

    Py_DECREF(idx_tuple);

    PyObject* ret_tuple = PyTuple_New(size);

    for (int i = 0; i < size; i++) {
        PyObject* inner_tuple1 = PyTuple_New(size);

        for (int j = 0; j < size; j++) {
            PyObject* inner_tuple2 = PyTuple_New(size);

            for (int k = 0; k < size; k++) {
                PyObject* value = PyLong_FromLong(table[i][j][k]);
                PyTuple_SetItem(inner_tuple2, k, value);
            }

            PyTuple_SetItem(inner_tuple1, j, inner_tuple2);
        }

        PyTuple_SetItem(ret_tuple, i, inner_tuple1);
    }

    Py_INCREF(ret_tuple);
    return ret_tuple;
}

// tpt -> Transition Probability Table
static PyMethodDef tpt_methods[] = {
    {"build_bigram", build_bigram, METH_VARARGS, "Build a bigram transition probability table (2D-Tuple)"},
    {"build_trigram", build_trigram, METH_VARARGS, "Build a trigram transition probability table (3D-Tuple)"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// tpt -> Transition Probability Table
static struct PyModuleDef tpt = {
    PyModuleDef_HEAD_INIT,
    "transition_probability_table",
    "Provides functions to build bigram and trigram transition probability tables",
    -1,
    tpt_methods
};

PyMODINIT_FUNC PyInit_transition_probability_table(void) {
    return PyModule_Create(&tpt);
}
