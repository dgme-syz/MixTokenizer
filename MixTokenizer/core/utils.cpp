#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <unordered_set>

namespace py = pybind11;
using Span = std::tuple<bool, size_t, size_t>;

class ChineseSplitter {
public:
    ChineseSplitter(const std::vector<char32_t>& extra_chars = {}) {
        for (auto c : extra_chars) zh_chars_.insert(c);
    }

    bool is_chinese_char(char32_t c) const {
        return zh_chars_.count(c) > 0;
    }

    std::vector<Span> split_zh_nonzh(const std::u32string& text32) const {
        std::vector<Span> result;
        if (text32.empty()) return result;

        bool zh_flag = is_chinese_char(text32[0]);
        size_t start = 0;

        for (size_t i = 0; i < text32.size(); ++i) {
            bool cur_flag = is_chinese_char(text32[i]);
            if (cur_flag != zh_flag) {
                result.emplace_back(zh_flag, start, i);
                start = i;
                zh_flag = cur_flag;
            }
        }
        result.emplace_back(zh_flag, start, text32.size());
        return result;
    }

    std::vector<Span> py_split_zh_nonzh(const std::string& text_utf8) const {
        std::u32string text32;
        size_t i = 0;
        while (i < text_utf8.size()) {
            char32_t cp = 0;
            unsigned char c = text_utf8[i];
            if (c <= 0x7F) {
                cp = c; i += 1;
            } else if ((c & 0xE0) == 0xC0) {
                cp = ((c & 0x1F) << 6) | (text_utf8[i+1] & 0x3F); i += 2;
            } else if ((c & 0xF0) == 0xE0) {
                cp = ((c & 0x0F) << 12) | ((text_utf8[i+1] & 0x3F) << 6) | (text_utf8[i+2] & 0x3F);
                i += 3;
            } else if ((c & 0xF8) == 0xF0) {
                cp = ((c & 0x07) << 18) | ((text_utf8[i+1] & 0x3F) << 12) | ((text_utf8[i+2] & 0x3F) << 6) | (text_utf8[i+3] & 0x3F);
                i += 4;
            } else {
                i += 1; continue;
            }
            text32.push_back(cp);
        }
        return split_zh_nonzh(text32);
    }

    std::vector<uint32_t> get_state() const {
        std::vector<uint32_t> state;
        for (auto c : zh_chars_) state.push_back(static_cast<uint32_t>(c));
        return state;
    }

    void set_state(const std::vector<uint32_t>& state) {
        zh_chars_.clear();
        for (auto c : state) zh_chars_.insert(static_cast<char32_t>(c));
    }

public:
    std::unordered_set<char32_t> zh_chars_;
};

// pybind11 wrapper
PYBIND11_MODULE(utils, m) {
    py::class_<ChineseSplitter>(m, "ChineseSplitter")
        .def(py::init([](py::sequence seq){
            std::vector<char32_t> vec;
            for (auto item : seq) {
                uint32_t val = item.cast<uint32_t>();
                vec.push_back(static_cast<char32_t>(val));
            }
            return new ChineseSplitter(vec);
        }))
        .def("is_chinese_char", &ChineseSplitter::is_chinese_char)
        .def("split_zh_nonzh", &ChineseSplitter::split_zh_nonzh)
        .def("py_split_zh_nonzh", &ChineseSplitter::py_split_zh_nonzh)
        .def(py::pickle(
            // __getstate__
            [](const ChineseSplitter &self) {
                std::vector<uint32_t> chars(self.zh_chars_.begin(), self.zh_chars_.end());
                return py::make_tuple(chars);
            },
            // __setstate__
            [](py::tuple t) {
                std::vector<uint32_t> chars = t[0].cast<std::vector<uint32_t>>();
                ChineseSplitter splitter;   
                splitter.set_state(chars);
                return splitter;           
            }
        ));
}