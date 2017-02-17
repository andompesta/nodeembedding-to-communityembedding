import cython
import numpy as np
cimport numpy as np
from cpython cimport PyCapsule_GetPointer
import sys

from libc.math cimport exp, isnan
from libc.string cimport memset
import scipy.linalg.blas as fblas

cdef extern from "math.h":
    double log(double x) nogil

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)


REAL = np.float32
ctypedef np.float32_t REAL_t

DEF MAX_SENTENCE_LEN = 10000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
ctypedef void (*sgemm_ptr) (char *transA, char *transB, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc) nogil
ctypedef void (*dgemm_ptr) (char *transA, char *transB, int *m, int *n, int *k, float *alpha, double *a, int *lda, double *b, int *ldb, float *beta, double *c, int *ldc) nogil


ctypedef unsigned long long (*fast_context_loss_prt) (
    const int negative,
    np.uint32_t *table,
    unsigned long long table_len,
    REAL_t *node_embedding,
    REAL_t *negative_embedding,
    const int size,
    const np.uint32_t word_index,
    const np.uint32_t word2_index,
    const REAL_t _lambda,
    float loss,
    unsigned long long next_random
    ) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x

cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)

cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

cdef sgemm_ptr sgemm=<sgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer) # float x = alpha * (A B) + beta * (x)
cdef dgemm_ptr dgemm=<dgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer) # double x = alpha * (A B) + beta * (x)

cdef fast_context_loss_prt fast_context_loss


DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t NONEF = <REAL_t>-1.0

cdef REAL_t DECF = <REAL_t>0.05
cdef REAL_t NDECF = <REAL_t>-0.05


cdef struct RetType:
    unsigned long long next_random
    float loss



cdef unsigned long long fast_context_loss_0 (
        const int negative,
        np.uint32_t *table,
        unsigned long long table_len,
        REAL_t *node_embedding,
        REAL_t *negative_embedding,
        const int size,
        const np.uint32_t word_index,
        const np.uint32_t word2_index,
        const REAL_t _lambda,
        float loss,
        unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, gl
    cdef np.uint32_t target_index
    cdef int d, i

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = NONEF

        row2 = target_index * size
        f = <REAL_t>dsdot(&size, &node_embedding[row1], &ONE, &negative_embedding[row2], &ONE)
        f *= label

        #sigmoid function
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        loss = loss + log(f)
    loss = loss * _lambda
    return next_random

cdef unsigned long long fast_context_loss_1 (
        const int negative,
        np.uint32_t *table,
        unsigned long long table_len,
        REAL_t *node_embedding,
        REAL_t *negative_embedding,
        const int size,
        const np.uint32_t word_index,
        const np.uint32_t word2_index,
        const REAL_t _lambda,
        float loss,
        unsigned long long next_random) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, gl, f_log
    cdef np.uint32_t target_index
    cdef int d, i

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = NONEF

        row2 = target_index * size
        f = <REAL_t>sdot(&size, &node_embedding[row1], &ONE, &negative_embedding[row2], &ONE)
        f *= label

        #sigmoid function
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        f_log = log(f)

        with nogil:
            print(str(f))
            print(str(f_log))

        loss = loss + log(f)
    loss = loss * _lambda
    return next_random

def o2_loss(py_node_embedding, py_negative_embedding, py_path, py_negative, py_window, py_table,
             py_lambda=1.0, py_size=None):

    cdef REAL_t *node_embedding = <REAL_t *>(np.PyArray_DATA(py_node_embedding))
    cdef REAL_t *negative_embedding = <REAL_t *>(np.PyArray_DATA(py_negative_embedding))
    cdef int size = py_size
    cdef REAL_t _lambda = py_lambda

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int path_len
    cdef int negative
    cdef int window = py_window
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int i, j, k
    cdef long result = 0
    cdef float loss = 0.0

    # For negative sampling
    cdef np.uint32_t *table = <np.uint32_t *>(np.PyArray_DATA(py_table))
    cdef unsigned long long table_len = len(py_table)
    cdef unsigned long long next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)


    path_len = <int>min(MAX_SENTENCE_LEN, len(py_path))
    negative = <int>py_negative

    for i in range(path_len):
        word = py_path[i]
        if word is None:
            codelens[i] = 0
        else:
            if window > 1:
                reduced_windows[i] = np.random.randint(window)
            else:
                reduced_windows[i] = 0
            indexes[i] = word.index
            codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(path_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > path_len:
                k = path_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                else:
                    next_random = fast_context_loss(negative, table, table_len, node_embedding, negative_embedding,
                                                       size, indexes[i], indexes[j], _lambda, loss, next_random)
    return loss


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    global fast_context_loss

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        fast_context_loss = fast_context_loss_0
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_context_loss = fast_context_loss_1
        return 1  # float

FAST_VERSION = init()  # initialize the module
