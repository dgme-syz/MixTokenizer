# cython: language_level=3
cimport cython

ZH_START = 0x4E00
ZH_END = 0x9FFF
cdef set EXTRA_ZH_CHARS = set([
    "。","？","！","【","】","，","、","；","：",
    "「","」","『","』","’","“","”","‘",
    "（","）","〔","〕","…","–","．","—",
    "《","》","〈","〉",
    "·","～","︰","︱",
])

cdef inline int utf8_len(int cp):
    if cp <= 0x7F:
        return 1
    elif cp <= 0x7FF:
        return 2
    elif cp <= 0xFFFF:
        return 3
    else:
        return 4

@cython.boundscheck(False)
@cython.wraparound(False)
def zh_encode(str text):
    cdef int n = len(text)
    cdef list char_flags = [False] * n
    cdef int i, l

    for i in range(n):
        c = text[i]
        cp = ord(c)
        char_flags[i] = (ZH_START <= cp <= ZH_END) or (c in EXTRA_ZH_CHARS)

    cdef list flags = []
    for i in range(n):
        l = utf8_len(ord(text[i]))
        flags.extend([char_flags[i]] * l)

    text_bytes = text.encode("utf-8")
    return tuple(text_bytes), tuple(flags)
