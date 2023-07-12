#define PY_SSIZE_T_CLEAN
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
#ifndef _PY_FLOAT_CHECK_
#define _PY_FLOAT_CHECK_(x) if (!PyFloat_Check(x)) { \
    PyErr_SetString(PyExc_TypeError, "Expected float!"); \
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

PyObject* posterior_probability(PyObject* priors, PyObject* priors_keys,
                                Py_ssize_t priors_size, PyObject* likelihoods,
                                PyObject* label_counts, PyObject* label,
                                PyObject* value, PyObject* inner) {
    PyObject* p_label_obj = PyDict_GetItem(priors, label);
    PyObject* label_likelihood_value_obj = PyDict_GetItem(PyDict_GetItemWithError(likelihoods, label), value);
    PyObject* label_count_obj = PyDict_GetItem(label_counts, label);
    _PY_FLOAT_CHECK_(p_label_obj);
    _PY_LONG_CHECK_(label_likelihood_value_obj);
    _PY_LONG_CHECK_(label_count_obj);

    double p_label = PyFloat_AsDouble(p_label_obj);
    double label_likelihood_value = PyLong_AsLong(label_likelihood_value_obj);
    double label_count = PyLong_AsLong(label_count_obj);
    double likelihood = label_likelihood_value / label_count;

    double evidence = 0.0;


    for(Py_ssize_t i = 0; i < priors_size; i++) {
        label = PyList_GetItem(priors_keys, i);
        p_label_obj = PyDict_GetItem(priors, label);
        label_likelihood_value_obj = PyDict_GetItem(PyDict_GetItem(likelihoods, label), value);
        label_count_obj = PyDict_GetItem(label_counts, label);

        _PY_FLOAT_CHECK_(p_label_obj);
        _PY_LONG_CHECK_(label_likelihood_value_obj);
        _PY_LONG_CHECK_(label_count_obj);

        double p_label_ = PyFloat_AsDouble(p_label_obj);
        double likelihood_ = PyLong_AsLong(label_likelihood_value_obj);
        likelihood_ /= PyLong_AsLong(label_count_obj);

        evidence += p_label_ * likelihood_;
    }

    double posterior_prob = (p_label * likelihood) / evidence;

    PyDict_SetItem(inner, value, PyFloat_FromDouble(posterior_prob));

    return inner;
}

static PyObject* compute_posteriors(PyObject* self, PyObject* args) {
    PyObject* features;
    PyObject* labels;
    PyObject* priors;

    if (!PyArg_ParseTuple(args, "OOO", &features, &labels, &priors)) {
        PyErr_SetString(PyExc_ValueError, "Expected an indexable sequence!");
        return NULL;
    }

    if (!PySequence_Check(labels) || !PySequence_Check(features)) {
        PyErr_SetString(PyExc_TypeError, "Expected an indexable sequence!");
        return NULL;
    }

    if (!PyDict_Check(priors)) {
        PyErr_SetString(PyExc_TypeError, "Expected dict of priors!");
        return NULL;
    }

    Py_ssize_t samples_size = PySequence_Length(features);
    Py_ssize_t labels_size = PySequence_Length(labels);
    Py_ssize_t priors_size = PyDict_Size(priors);

    if (samples_size != labels_size) {
        PyErr_Format(
            PyExc_ValueError,
            "Unequal Number of features samples and labels, (%i != %i)",
            (int) samples_size,
            (int) labels_size
        );
        return NULL;
    }

    PyObject* priors_keys = PyDict_Keys(priors);
    PyObject* label;

    PyObject* likelihoods = PyDict_New();
    PyObject* label_counts = PyDict_New();
    PyObject* feature_set = PySet_New(NULL);

    for (Py_ssize_t i = 0; i < priors_size; i++) {
        label = PyList_GetItem(priors_keys, i);

        PyDict_SetItem(likelihoods, label, PyDict_New());
        PyDict_SetItem(label_counts, label, PyLong_FromLong(0));
    }

    for (Py_ssize_t i = 0; i < samples_size; i++) {
        PyObject* feature_ = PySequence_GetItem(features, i);
        PyObject* label_ = PySequence_GetItem(labels, i);

        PyObject* value;
        PyObject* inner;
        PyObject* counts;
        Py_ssize_t size_ = PySequence_Length(feature_);

        for (Py_ssize_t j = 0; j < size_; j++) {
            value = PySequence_GetItem(feature_, j);
            PySet_Add(feature_set, value);

            inner = PyDict_GetItem(likelihoods, label_);
            counts = PyDict_GetItem(inner, value);

            if (NULL == counts) {
                counts = PyLong_FromLong(0);
            }

            _PY_LONG_CHECK_(counts);

            counts = PyLong_FromLong(PyLong_AsLong(counts) + 1);
            PyDict_SetItem(inner, value, counts);

            counts = PyDict_GetItem(label_counts, label_);
            counts = PyLong_FromLong(PyLong_AsLong(counts) + 1);
            PyDict_SetItem(label_counts, label_, counts);
            PyDict_SetItem(likelihoods, label_, inner);
        }
    }

    PyObject* posteriors = PyDict_New();
    for (Py_ssize_t i = 0; i < priors_size; i++) {
        PyObject* label = PyList_GetItem(priors_keys, i);

        PyObject* feature_set_iter = PyObject_GetIter(feature_set);
        PyObject* value;

        PyObject* inner = PyDict_New();
        int ctr = 0;
        while((value = PyIter_Next(feature_set_iter)) != NULL) {
            inner = posterior_probability(
                priors, priors_keys, priors_size, likelihoods,
                label_counts, label,value, inner
            );
        }

        PyDict_SetItem(posteriors, label, inner);
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
