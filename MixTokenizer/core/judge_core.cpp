#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

py::array_t<bool> is_new_char_array(const std::u32string &text) {
    ssize_t n = text.size();
    auto result = py::array_t<bool>(n);
    auto r = result.mutable_data();

    for (ssize_t i = 0; i < n; i++) {
        char32_t cp = text[i];
        r[i] = (cp >= 0xE000 && cp <= 0xF8FF) ||
               (cp >= 0xF0000 && cp <= 0xFFFFD) ||
               (cp >= 0x100000 && cp <= 0x10FFFD);
    }

    return result;
}

PYBIND11_MODULE(judge_core, m) {
    m.def("is_new_char_array", &is_new_char_array,
          py::arg("text"));  
}
