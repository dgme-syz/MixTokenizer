#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace py = pybind11;

// ---------------- is_new_char_array ----------------
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

// ---------------- find_change_points ----------------
std::vector<std::vector<int>> find_change_points(py::array_t<bool> is_new_array, int level = 1) {
    auto buf = is_new_array.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input array must be 1D");

    size_t n = buf.shape[0];
    if (n == 0) return {};

    if (level <= 0) throw std::invalid_argument("level must be positive");

    bool* ptr = static_cast<bool*>(buf.ptr);

    std::vector<unsigned char> change_idx(n, 0);
    for (size_t i = 1; i < n; ++i) {
        if (ptr[i] != ptr[i-1]) change_idx[i] = 1;
    }

    std::vector<std::vector<int>> result;
    result.push_back({0, 0, 0});
    size_t start = 0;
    int current_flag = ptr[0] ? 1 : 0;
    int x = 0;

    for (size_t i = 1; i < n; ++i) {
        if (change_idx[i]) {
            size_t l = start;
            size_t r = i;
            if (current_flag) {
                x = (i - start) % level;
                result.back()[2] += x;
                l += x;
            }
            if (l < r) result.push_back({current_flag, static_cast<int>(l), static_cast<int>(r)});
            start = i;
            current_flag = ptr[i] ? 1 : 0;
        }
    }

    if (current_flag) {
        x = (n - start) % level;
        if (x > 0) result.back()[2] += x;
        start += x;
    }
    if (start < n) result.push_back({current_flag, static_cast<int>(start), static_cast<int>(n)});

    return result;
}

// ---------------- ComboTrie ----------------
struct TrieNode {
    bool is_end = false;
    std::unordered_map<int, std::unique_ptr<TrieNode>> children;
};

class ComboTrie {
public:
    ComboTrie() : root(std::make_unique<TrieNode>()) {}

    ComboTrie(const std::vector<std::vector<int>>& combo_list) : ComboTrie() {
        for (auto &seq : combo_list) insert(seq);
    }

    void insert(const std::vector<int>& seq) {
        TrieNode* node = root.get();
        for (int t : seq) {
            if (!node->children.count(t)) node->children[t] = std::make_unique<TrieNode>();
            node = node->children[t].get();
        }
        node->is_end = seq.size();
    }

    std::vector<std::vector<int>> find_change_points_plus(const std::vector<int>& token_ids) const {
        size_t n = token_ids.size();
        if (n == 0) return {};

        std::vector<std::vector<int>> result;
        size_t l = 0, r = 0;       
        int last_flag = 0;         
        TrieNode* node = root.get(); 
        size_t combo_last = 0;     

        for (int i = 0; i < n; ++i) {
            int t = token_ids[i];
            // printf("%d %d %d\n", l, r, last_flag);
            if (node && node->children.count(t)) {
                node = node->children.at(t).get();
                if (node->is_end) {
                    if (last_flag == 1) {
                        r = i + 1;
                        node = root.get();
                    } else {
                        // printf("[end]\n");
                        if (l < r) {
                            result.push_back({last_flag, (int)l, (int)r});
                        }
                        node = root.get();
                        l = r, r = i + 1, last_flag = 1;
                    }
                }
            } else {
                // printf("&&&&&&&&&&&&&&&");
                if (last_flag == 1) {
                    if (l < r) result.push_back({last_flag, (int)l, (int)r});
                    l = r, r = i, last_flag = 0;
                } else {
                    r = i;
                }
                if (root.get()->children.count(t)) {
                    node = root.get()->children.at(t).get();
                } else {
                    r += 1, node = root.get();
                }
            }
        }
        
        if (l < n) {
            if (last_flag == 1) {
                if (l < r) result.push_back({1, (int)l, (int)r});
                if (r < n) result.push_back({0, (int)r, (int)n});
            } else {
                result.push_back({0, (int)l, (int)n});
            }
        }
        return result;
    }

public:
    std::unique_ptr<TrieNode> root;
};

// ---------------- Pybind11 Module ----------------
PYBIND11_MODULE(cpp_core, m) {
    m.doc() = "Judge core module with pickle support for ComboTrie";

    m.def("is_new_char_array", &is_new_char_array,
          py::arg("text"),
          "Return numpy bool array indicating private-use characters");

    m.def("find_change_points", &find_change_points,
          py::arg("is_new_array"), py::arg("level") = 1,
          "Find change points in a numpy boolean array");

    py::class_<ComboTrie>(m, "ComboTrie")
    .def(py::init<>())
    .def(py::init<const std::vector<std::vector<int>>&>(), py::arg("combo_list"))
    .def("insert", &ComboTrie::insert, py::arg("seq"))
    .def("find_change_points_plus", &ComboTrie::find_change_points_plus, py::arg("token_ids"))

    // Pickle support
    .def(py::pickle(
        // __getstate__: return combo_list
        [](const ComboTrie &trie) {
            std::vector<std::vector<int>> combo_list;
            std::vector<int> path;
            std::function<void(const TrieNode*, std::vector<int>&)> dfs;
            dfs = [&](const TrieNode* node, std::vector<int>& path) {
                if (node->is_end) combo_list.push_back(path);
                for (const auto& [k, child] : node->children) {
                    path.push_back(k);
                    dfs(child.get(), path);
                    path.pop_back();
                }
            };
            dfs(trie.root.get(), path);
            return py::make_tuple(combo_list);
        },
        [](py::tuple t) {
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");
            auto combo_list = t[0].cast<std::vector<std::vector<int>>>();
            return ComboTrie(combo_list);
        }
    ));
}
