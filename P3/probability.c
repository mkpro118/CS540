#include <Python.h>

#ifndef _PY_LONG_CHECK_TPT_
#define _PY_LONG_CHECK_TPT_(x, t) if (!PyLong_Check(x)) { \
    free(t); \
    PyErr_SetString(PyExc_TypeError, "Expected tuple of integers"); \
    return NULL; \
}
#endif

#ifndef _PY_LONG_CHECK_
#define _PY_LONG_CHECK_(x) if (!PyLong_Check(x)) { \
    PyErr_SetString(PyExc_TypeError, "Expected integer!"); \
    return NULL; \
}
#endif

#ifndef _INDEX_CHECK_TPT_
#define _INDEX_CHECK_TPT_(x, y, z) {\
    if (x >= y) { \
        free(z); \
        PyErr_SetString(PyExc_ValueError, "Index is larger than size of table"); \
        return NULL; \
    } \
}
#endif

#ifndef _ARG_PARSE_TPT_
#define _ARG_PARSE_TPT_ PyObject* idx_tuple; \
    int size; \
    int smoothing = 0; \
    static char* kwlist[] = {"x", "size", "smoothing", NULL}; \
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!i|p", kwlist, \
         &PyTuple_Type, &idx_tuple, &size, &smoothing)) { \
        return NULL; \
    } \
    Py_ssize_t tuple_size = PyTuple_Size(idx_tuple); \
    if (tuple_size == -1) { \
        PyErr_SetString(PyExc_ValueError, "Error occurred while getting the length of the sequence"); \
        return NULL; \
    }
#endif

#ifndef _PY_DECREF_ALL_
#define _PY_DECREF_ALL_(...) do { \
    _py_decref_all_(__VA_ARGS__); \
} while(0);
#endif


void _py_decref_all_(PyObject* arg, ...) {
    Py_DECREF(arg); // Decrement reference count for the first argument

    va_list args;
    va_start(args, arg);

    PyObject* next_arg;
    while ((next_arg = va_arg(args, PyObject*))) {
        Py_DECREF(next_arg); // Decrement reference count for each subsequent argument
    }

    va_end(args);
}

static PyObject* build_unigram(PyObject* self, PyObject* args, PyObject* kwargs) {
    _ARG_PARSE_TPT_;

    int* table = calloc(size + 1, sizeof(int));

    PyObject* item;
    int curr;

    for (Py_ssize_t i = 0; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_TPT_(item, table);
        curr = (int) PyLong_AsLong(item);

        _INDEX_CHECK_TPT_(curr, size, table);

        table[curr]++;
        table[size]++;
    }

    Py_DECREF(idx_tuple);

    PyObject* ret_tuple = PyTuple_New(size);

    for (int i = 0; i < size; i++) {
        if (smoothing) {
            item = PyFloat_FromDouble(((float) table[i] + 1) / (table[size] + size));
        } else {
            item = PyFloat_FromDouble(((float) table[i]) / table[size]);
        }
        PyTuple_SetItem(ret_tuple, i, item);
    }

    Py_INCREF(ret_tuple);
    return ret_tuple;
}


static PyObject* build_bigram(PyObject* self, PyObject* args, PyObject* kwargs) {
    _ARG_PARSE_TPT_;

    int** table = malloc(sizeof(int*) * size);
    for (int i = 0; i < size; ++i){
        table[i] = calloc(size + 1, sizeof(int));
    }

    int prev;
    int curr;
    PyObject* item;

    item = PyTuple_GetItem(idx_tuple, 0);
    _PY_LONG_CHECK_TPT_(item, table);

    prev = (int) PyLong_AsLong(item);
    _INDEX_CHECK_TPT_(prev, size, table);

    // Start from 1, since we cannot create
    // a transition probability matrix for the first
    // letter, since there are no preceeding letters
    for (Py_ssize_t i = 1; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_TPT_(item, table);
        curr = (int) PyLong_AsLong(item);
        _INDEX_CHECK_TPT_(curr, size, table);

        table[prev][curr]++;
        table[prev][size]++;
        prev = curr;
    }

    Py_DECREF(idx_tuple);

    PyObject* ret_tuple = PyTuple_New(size);

    for (int i = 0; i < size; i++) {
        PyObject* inner_tuple = PyTuple_New(size);
        for (int j = 0; j < size; j++) {
            if (smoothing) {
                item = PyFloat_FromDouble(((float) table[i][j] + 1) / (table[i][size] + size));
            } else {
                item = PyFloat_FromDouble(((float) table[i][j]) / table[i][size]);
            }
            PyTuple_SetItem(inner_tuple, j, item);
        }
        PyTuple_SetItem(ret_tuple, i, inner_tuple);
    }

    Py_INCREF(ret_tuple);
    return ret_tuple;
}


static PyObject* build_trigram(PyObject* self, PyObject* args, PyObject* kwargs) {
    _ARG_PARSE_TPT_;

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
    _PY_LONG_CHECK_TPT_(item, table);

    prev1 = (int) PyLong_AsLong(item);
    _INDEX_CHECK_TPT_(prev1, size, table);

    item = PyTuple_GetItem(idx_tuple, 1);
    _PY_LONG_CHECK_TPT_(item, table);

    prev2 = (int) PyLong_AsLong(item);
    _INDEX_CHECK_TPT_(prev2, size, table);

    for (Py_ssize_t i = 2; i < tuple_size; i++) {
        item = PyTuple_GetItem(idx_tuple, i);
        _PY_LONG_CHECK_TPT_(item, table);
        curr = (int) PyLong_AsLong(item);

        _INDEX_CHECK_TPT_(curr, size, table);

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
                if (smoothing) {
                    item = PyFloat_FromDouble(((float) table[i][j][k] + 1) / (table[i][j][size] + size));
                } else {
                    item = PyFloat_FromDouble(((float) table[i][j][k]) / table[i][j][size]);
                }
                PyTuple_SetItem(inner_tuple2, k, item);
            }

            PyTuple_SetItem(inner_tuple1, j, inner_tuple2);
        }

        PyTuple_SetItem(ret_tuple, i, inner_tuple1);
    }

    Py_INCREF(ret_tuple);
    return ret_tuple;
}


static PyObject* compute_priors(PyObject* self, PyObject* args) {
    PyObject* labels;

    if (!PyArg_ParseTuple(args, "O", &labels)) {
        PyErr_SetString(PyExc_ValueError, "Expected an indexable sequence!");
        return NULL;
    }

    if (!PySequence_Check(labels)) {
        PyErr_SetString(PyExc_TypeError, "Expected an indexable sequence!");
        return NULL;
    }

    Py_ssize_t list_size = PySequence_Length(labels);

    if (list_size == -1) {
        PyErr_SetString(PyExc_ValueError, "Error occurred while getting the length of the sequence");
        return NULL;
    }

    PyObject* counts = PyDict_New();

    if (NULL == counts) {
        Py_DECREF(labels);
        PyErr_SetString(PyExc_ValueError, "Unexpected error!");
        return NULL;
    }

    PyObject* label;
    PyObject* value;

    unsigned int unique = 0;

    for (Py_ssize_t i = 0; i < list_size; i++) {
        label = PySequence_GetItem(labels, i);
        if (NULL == label) {
            PyErr_SetString(PyExc_ValueError, "Unexpected error while getting labels!");
        }

        value = PyDict_GetItem(counts, label);

        if (NULL == value) {
            value = PyLong_FromLong(1);
            unique++;
        } else {
            value = PyLong_FromLong(PyLong_AsLong(value) + 1);
        }

        PyDict_SetItem(counts, label, value);
    }

    Py_DECREF(label);

    PyObject* keys = PyDict_Keys(counts);

    if (NULL == label) {
        PyErr_SetString(PyExc_ValueError, "Unexpected error while getting dict keys!");
    }

    for (unsigned int i = 0; i < unique; i++) {
        label = PyList_GetItem(keys, (Py_ssize_t) i);
        value = PyDict_GetItem(counts, label);

        _PY_LONG_CHECK_(value);

        value = PyFloat_FromDouble(((double) PyLong_AsLong(value)) / list_size);

        PyDict_SetItem(counts, label, value);
    }

    Py_INCREF(counts);

    return counts;
}

// TODO: Rewrite from python to C
static PyObject* compute_posteriors(PyObject* self, PyObject* args) {
    PyObject* features;
    PyObject* labels;

    if (!PyArg_ParseTuple(args, "OO", &features, &labels)) {
        PyErr_SetString(PyExc_ValueError, "Expected an indexable sequence!");
        return NULL;
    }

    if (!PySequence_Check(labels) || !PySequence_Check(features)) {
        PyErr_SetString(PyExc_TypeError, "Expected an indexable sequence!");
        return NULL;
    }

    Py_ssize_t features_size = PySequence_Length(features);
    Py_ssize_t labels_size = PySequence_Length(labels);

    if (features_size != labels_size) {
        PyErr_Format(PyExc_ValueError, "Unequal Number of features samples and labels, (%i != %i)", (int) features_size, (int) labels_size);
        return NULL;
    }

    PyObject* feature_counts = PyDict_New();
    PyObject* label_counts = PyDict_New();

    Py_ssize_t length;
    PyObject* label;
    PyObject* feature_vec;
    PyObject* feature;
    PyObject* inner_dict;
    PyObject* ctr;

    // Compute the counts of features given each label
    for (Py_ssize_t i = 0; i < features_size; i++) {
        label = PySequence_GetItem(labels, i);
        feature_vec = PySequence_GetItem(features, i);

        if (!PySequence_Check(feature_vec)) {
            PyErr_SetString(PyExc_ValueError, "Expected a sequence of features");
            return NULL;
        }

        length = PySequence_Length(feature_vec);
        for (Py_ssize_t j = 0; j < length; j++) {
            feature = PySequence_GetItem(feature_vec, j);

            inner_dict = PyDict_GetItem(feature_counts, label);

            if (NULL == inner_dict) {
                inner_dict = PyDict_New();
                PyDict_SetItem(feature_counts, label, inner_dict);
            }

            ctr = PyDict_GetItem(inner_dict, feature);
            ctr = (NULL == ctr) ? PyLong_FromLong(1) : PyLong_FromLong(PyLong_AsLong(ctr) + 1);
            PyDict_SetItem(inner_dict, feature, ctr);

            ctr = PyDict_GetItem(label_counts, label);
            ctr = (NULL == ctr) ? PyLong_FromLong(1) : PyLong_FromLong(PyLong_AsLong(ctr) + 1);
            PyDict_SetItem(label_counts, label, ctr);
        }
    }

    // Compute the posterior probabilities of features given each label
    PyObject* posteriors = PyDict_New();
    PyObject* total_count_obj;
    PyObject* feat_count_obj;
    PyObject* feature_counts_for_label;
    PyObject* feat_ratio;

    double total_count;
    double feature_count;

    PyObject* unique_labels = PyDict_Keys(label_counts); // label_counts.keys()
    Py_ssize_t unique_labels_size = PyList_Size(unique_labels);

    PyObject* unique_feature_counts_for_label;
    Py_ssize_t unique_feature_counts_for_label_size;

    for (Py_ssize_t i = 0; i < unique_labels_size; i++) {
        label = PyList_GetItem(unique_labels, i);

        total_count_obj = PyDict_GetItem(label_counts, label);
        total_count = (double) PyLong_AsLong(total_count_obj);

        feature_counts_for_label = PyDict_GetItem(feature_counts, label); // feature_counts[label]
        unique_feature_counts_for_label = PyDict_Keys(feature_counts_for_label); // feature_counts[label]

        unique_feature_counts_for_label_size = PyList_Size(unique_feature_counts_for_label);

        inner_dict = PyDict_New();

        PyDict_SetItem(posteriors, label, inner_dict);

        for (Py_ssize_t j = 0; j < unique_feature_counts_for_label_size; j++) {
            feature = PyList_GetItem(unique_feature_counts_for_label, j);

            feat_count_obj = PyDict_GetItem(feature_counts_for_label, feature);
            feature_count = (double) PyLong_AsLong(feat_count_obj);

            feat_ratio = PyFloat_FromDouble(feature_count / total_count);
            PyDict_SetItem(inner_dict, feature, feat_ratio);
        }

        for (Py_ssize_t j = 0; j < unique_feature_counts_for_label_size; j++) {
            feature = PyList_GetItem(unique_feature_counts_for_label, j);
            feat_ratio = PyDict_GetItem(inner_dict, feature);
            feat_ratio = PyFloat_FromDouble(PyFloat_AsDouble(feat_ratio) / sum_prob);
            PyDict_SetItem(inner_dict, feature, feat_ratio);
        }
    }

    Py_INCREF(posteriors);
    return posteriors;
}


static PyMethodDef probability_methods[] = {
    {
        "build_unigram", (PyCFunction) build_unigram, METH_VARARGS | METH_KEYWORDS,
        "Build a unigram transition probability table (1D-Tuple)"
    },
    {
        "build_bigram", (PyCFunction) build_bigram, METH_VARARGS | METH_KEYWORDS,
        "Build a bigram transition probability table (2D-Tuple)"
    },
    {
        "build_trigram", (PyCFunction) build_trigram, METH_VARARGS | METH_KEYWORDS,
        "Build a trigram transition probability table (3D-Tuple)"
    },
    {
        "compute_priors", (PyCFunction) compute_priors, METH_VARARGS,
        "Compute prior probabilities of the items in a given sequence"
    },
    {
        "compute_posteriors", (PyCFunction) compute_posteriors, METH_VARARGS,
        "Compute posterior probabilities of the features in samples given the labels"
    },
    {NULL, NULL, 0, NULL}  // Sentinel, marks the end of the array
};


static struct PyModuleDef probability = {
    PyModuleDef_HEAD_INIT,
    "probability",
    "Provides functions to build probability tables",
    -1,
    probability_methods
};


PyMODINIT_FUNC PyInit_probability(void) {
    return PyModule_Create(&probability);
}
