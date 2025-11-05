# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np
from cpython.unicode cimport PyUnicode_READ_CHAR
from cython.parallel import prange, parallel

ctypedef np.uint8_t bool_t
ctypedef np.int64_t int64_t


# --------------------------
# Change points detection
# --------------------------
def find_change_points(np.ndarray[bool_t, ndim=1] is_new):
    """
    Detect change points in boolean array.
    Thread-safe: main loop releases GIL.
    Returns a list of (flag, start, end).
    """
    cdef Py_ssize_t n = is_new.shape[0]
    if n == 0:
        return []

    cdef np.ndarray[bool_t, ndim=1] change_idx = np.zeros(n, dtype=np.uint8)
    cdef Py_ssize_t i

    for i in prange(1, n, nogil=True, schedule='static'):
        if is_new[i] != is_new[i-1]:
            change_idx[i] = 1

    # --- GIL ---
    cdef list result = []
    cdef Py_ssize_t start = 0
    cdef bool_t current = is_new[0]

    for i in range(1, n):
        if change_idx[i]:
            result.append((bool(current), start, i))
            start = i
            current = is_new[i]
    result.append((bool(current), start, n))

    return result


# --------------------------
# Parallel unicode char check (thread-safe)
# --------------------------
def is_new_char_array(str text):
    cdef Py_ssize_t n = len(text)
    cdef np.ndarray[bool_t, ndim=1] mask = np.empty(n, dtype=np.uint8)
    cdef Py_ssize_t i
    cdef int code

    for i in range(n):
        code = PyUnicode_READ_CHAR(text, i)
        mask[i] = (
            (0xE000 <= code <= 0xF8FF)
            or (0xF0000 <= code <= 0xFFFFD)
            or (0x100000 <= code <= 0x10FFFD)
        )
    return mask