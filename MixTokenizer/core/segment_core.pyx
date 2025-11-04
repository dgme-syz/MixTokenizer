# segment_core.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.unicode cimport PyUnicode_READ_CHAR
from cython cimport boundscheck, wraparound

ctypedef np.uint8_t bool_t
ctypedef np.int64_t int64_t

# --------------------------
# Change points detection
# --------------------------
def find_change_points(np.ndarray[bool_t, ndim=1] is_new):
    """
    Find segments where the boolean array value is constant.
    Returns a list of tuples (flag, start, end)
    """
    cdef Py_ssize_t n = is_new.shape[0]
    if n == 0:
        return []
    cdef Py_ssize_t i, start = 0
    cdef list result = []
    cdef bool_t current = is_new[0]

    for i in range(1, n):
        if is_new[i] != current:
            result.append((bool(current), start, i))
            start = i
            current = is_new[i]
    result.append((bool(current), start, n))
    return result

# --------------------------
# Fast unicode char check
# --------------------------
def is_new_char_array(str text):
    """
    Check for each character if it is in the private Unicode areas.
    Returns np.uint8 array (0/1).
    """
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

# --------------------------
# Cython Trie Implementation
# --------------------------
cdef class TrieNode:
    cdef dict children
    cdef int idx

    def __cinit__(self):
        self.children = {}
        self.idx = -1

cdef class Trie:
    cdef TrieNode root
    cdef int counter  # insertion order

    def __cinit__(self):
        self.root = TrieNode()
        self.counter = 0

    cpdef int insert(self, object sequence):
        """
        Insert a sequence of ints into the Trie.
        Returns the assigned index.
        """
        cdef TrieNode node = self.root
        cdef int token
        for token in sequence:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        if node.idx == -1:
            node.idx = self.counter
            self.counter += 1
        return node.idx

    cpdef int lookup(self, object sequence):
        """
        Lookup sequence in Trie.
        Returns idx if exists, else -1.
        """
        cdef TrieNode node = self.root
        cdef int token
        for token in sequence:
            if token not in node.children:
                return -1
            node = node.children[token]
        return node.idx

    cpdef list batch_lookup(self, object sequences):
        """
        Lookup multiple sequences at once.
        Returns list of idxs (-1 if not found).
        """
        cdef list result = []
        cdef object seq
        for seq in sequences:
            result.append(self.lookup(seq))
        return result
