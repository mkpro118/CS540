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
#ifndef _INDEX_CHECK_
#define _INDEX_CHECK_(x, y, z) {\
    if (x >= y) { \
        free(z); \
        PyErr_SetString(PyExc_ValueError, "Index is larger than size of table"); \
        return NULL; \
    } \
}
#endif

static PyObject* build_unigram(PyObject* self, PyObject* args) {
    PyObject* idx_tuple;
    int size;

    if (!PyArg_ParseTuple(args, "O!i", &PyTuple_Type, &idx_tuple, &size)) {
        return NULL;
    }

    Py_ssize_t tuple_size = PyTuple_Size(idx_tuple);

    int* table = calloc(size + 1, sizeof(int));

    PyObject* item;
    int curr;

    for (Py_ssize_t i = 0; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_(item, table);
        curr = (int) PyLong_AsLong(item);

        _INDEX_CHECK_(curr, size, table);

        table[curr]++;
        table[size]++;
    }

    Py_DECREF(idx_tuple);

    PyObject* ret_tuple = PyTuple_New(size);

    for (int i = 0; i < size; i++) {
        item = PyFloat_FromDouble(((double) table[i]) / table[size]);
        PyTuple_SetItem(ret_tuple, i, item);
    }

    Py_INCREF(ret_tuple);
    return ret_tuple;
}

static PyObject* build_bigram(PyObject* self, PyObject* args) {
    PyObject* idx_tuple;
    int size;

    if (!PyArg_ParseTuple(args, "O!i", &PyTuple_Type, &idx_tuple, &size)) {
        return NULL;
    }

    Py_ssize_t tuple_size = PyTuple_Size(idx_tuple);

    int** table = malloc(sizeof(int*) * size);
    for (int i = 0; i < size; ++i){
        table[i] = calloc(size + 1, sizeof(int));
    }

    int prev;
    int curr;
    PyObject* item;

    item = PyTuple_GetItem(idx_tuple, 0);
    _PY_LONG_CHECK_(item, table);

    prev = (int) PyLong_AsLong(item);
    _INDEX_CHECK_(prev, size, table);

    // Start from 1, since we cannot create
    // a transition probability matrix for the first
    // letter, since there are no preceeding letters
    for (Py_ssize_t i = 1; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_(item, table);
        curr = (int) PyLong_AsLong(item);
        _INDEX_CHECK_(curr, size, table);

        table[prev][curr]++;
        table[prev][size]++;
        prev = curr;
    }

    Py_DECREF(idx_tuple);

    PyObject* ret_tuple = PyTuple_New(size);

    for (int i = 0; i < size; i++) {
        PyObject* inner_tuple = PyTuple_New(size);
        for (int j = 0; j < size; j++) {
            item = PyFloat_FromDouble(((double) table[i][j]) / table[i][size]);
            PyTuple_SetItem(inner_tuple, j, item);
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
        table[i] = malloc(sizeof(int*) * size);
        for (int j = 0; j < size; j++) {
            table[i][j] = calloc(size + 1, sizeof(int));
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
    _INDEX_CHECK_(prev1, size, table);

    item = PyTuple_GetItem(idx_tuple, 1);
    _PY_LONG_CHECK_(item, table);

    prev2 = (int) PyLong_AsLong(item);
    _INDEX_CHECK_(prev2, size, table);

    for (Py_ssize_t i = 2; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_(item, table);
        curr = (int) PyLong_AsLong(item);

        _INDEX_CHECK_(curr, size, table);

        table[prev1][prev2][curr]++;
        table[prev1][prev2][size]++;
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
                PyObject* value = PyFloat_FromDouble(((double) table[i][j][k]) / table[i][j][size]);
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
    {"build_unigram", build_unigram, METH_VARARGS, "Build a unigram transition probability table (1D-Tuple)"},
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
