#include <vector>
#include <unordered_set>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <unordered_map>
#include <tuple>

namespace py = pybind11;

class HybridDecoder {
public:
    size_t k;
    uint64_t base1 = 257, mod1 = 1000000007;
    uint64_t base2 = 263, mod2 = 1000000009;

    std::unordered_map<uint64_t, std::unordered_map<uint64_t, int>> values;

    HybridDecoder(size_t k_) : k(k_) {}

    void add_pattern(const std::vector<int>& seq, int val) {
        if (seq.size() != k) return;

        uint64_t h1 = 0, h2 = 0;
        for (size_t i = 0; i < k; ++i) {
            h1 = (h1 * base1 + seq[i]) % mod1;
            h2 = (h2 * base2 + seq[i]) % mod2;
        }
        values[h1][h2] = val;
    }

    std::vector<std::tuple<int,size_t,size_t>> search(const std::vector<int>& nums) const {
        size_t n = nums.size();
        std::vector<std::tuple<int,size_t,size_t>> intervals;
        if (n < k) return {{-1, 0, n}};

        uint64_t h1 = 0, h2 = 0;
        uint64_t base_pow1 = 1, base_pow2 = 1;
        for (size_t i = 0; i < k-1; ++i) {
            base_pow1 = (base_pow1 * base1) % mod1;
            base_pow2 = (base_pow2 * base2) % mod2;
        }

        size_t work = 0;
        for (size_t i = 0; i <= n - k; ++i) {
            if (i == 0) {
                for (size_t j = 0; j < k; ++j) {
                    h1 = (h1 * base1 + nums[j]) % mod1;
                    h2 = (h2 * base2 + nums[j]) % mod2;
                }
            } else {
                h1 = (h1 + mod1 - (nums[i-1] * base_pow1) % mod1) % mod1;
                h1 = (h1 * base1 + nums[i+k-1]) % mod1;

                h2 = (h2 + mod2 - (nums[i-1] * base_pow2) % mod2) % mod2;
                h2 = (h2 * base2 + nums[i+k-1]) % mod2;
            }

            int flag = -1;
            auto it1 = values.find(h1);
            if (it1 != values.end()) {
                auto it2 = it1->second.find(h2);
                if (it2 != it1->second.end()) flag = it2->second;
            }

            if (flag != -1) {
                intervals.push_back({flag, i, i+k});
                work = i+k;
            }
        }

        size_t last_end = 0;
        std::vector<std::tuple<int,size_t,size_t>> result;
        for (auto& inter : intervals) {
            auto [flag,l,r] = inter;
            if (last_end < l) result.push_back({-1,last_end,l});
            result.push_back(inter);
            last_end = r;
        }
        if (last_end < n) result.push_back({-1,last_end,n});

        return result;
    }

    py::tuple __getstate__() const {
        std::vector<std::tuple<uint64_t,uint64_t,int>> val_vec;
        for (const auto& [h1, sub_map] : values) {
            for (const auto& [h2, v] : sub_map) {
                val_vec.emplace_back(h1, h2, v);
            }
        }
        return py::make_tuple(k, val_vec);
    }

    static HybridDecoder __setstate__(py::tuple t) {
        size_t k_ = t[0].cast<size_t>();
        auto val_vec = t[1].cast<std::vector<std::tuple<uint64_t,uint64_t,int>>>();
        HybridDecoder dec(k_);
        for (const auto& item : val_vec) {
            uint64_t h1 = std::get<0>(item);
            uint64_t h2 = std::get<1>(item);
            int v = std::get<2>(item);
            dec.values[h1][h2] = v;
        }
        return dec;
    }
};



class ACAutomaton {
public:
    int idx = 0;
    ACAutomaton() {
        root = new ACNode();
        root->uuid = 0;
    }

    ~ACAutomaton() {
        clear(root);
    }

    void add_pattern(const std::vector<int>& pattern, int val) {
        ACNode* node = root;
        for (int x : pattern) {
            if (!node->children.count(x)) {
                node->children[x] = new ACNode();
                node->children[x]->uuid = ++idx;
                node->children[x]->depth = node->depth + 1;
                node->children[x]->value = x;
                node->children[x]->fail = root;
            }
            node = node->children[x];
        }
        node->end_value = val;
    }

    void build() {
        std::queue<ACNode*> q;
        root->fail = root;
        for (auto& [k, child] : root->children) {
            child->fail = root;
            q.push(child);
        }

        while (!q.empty()) {
            ACNode* node = q.front(); q.pop();
            assert (node->fail != nullptr);
            for (auto& [k, child] : node->children) {
                ACNode* f = node->fail;
                while (f != root && !f->children.count(k))
                    f = f->fail;
                if (f->children.count(k))
                    f = f->children[k];
                child->fail = f;
                q.push(child);
            }
        }
    }

    std::vector<std::tuple<int, size_t, size_t>> search(const std::vector<int>& text) {
        std::vector<std::tuple<int, size_t, size_t>> results;
        size_t n = text.size();
        size_t last_pos = 0;
        ACNode* node = root;

        for (size_t i = 0; i < n; ++i) {
            int x = text[i];
            while (node != root && !node->children.count(x)) {
                if (node->fail == nullptr) {
                    printf("Warning: node fail is null while processing char %d at index %zu\n", x, i);
                    printf("Node: uuid=%d, value=%d\n", node->uuid, node->value);
                    assert (false); // should not happen
                }
                node = node->fail;
            }
            if (node && node->children.count(x))
                node = node->children[x];
            // printf("At text index %zu, char=%d, current node uuid=%d, value=%d, is_root=%d, depth=%zu, fail=%d\n", i, x, node->uuid, node->value, node == root, node->depth, node->fail != nullptr ? node->fail->value : -1);
            ACNode* tmp = node;
            // if (tmp && tmp->link) tmp = tmp->link;
            while (tmp != root && tmp != nullptr) {
                if (tmp->end_value != -1) {
                    node->link = tmp;
                    size_t start = i + 1 - tmp->depth;
                    if (last_pos < start) {
                        results.emplace_back(-1, last_pos, start);
                        // printf("Added non-match interval [%zu, %zu) before match at index %zu\n", last_pos, start, i);
                    }
                    results.emplace_back(tmp->end_value, start, i + 1);
                    // printf("Now info of node at index %zu: uuid=%d, value=%d, depth=%zu, end_value=%d\n", i, tmp->uuid, tmp->value, tmp->depth, tmp->end_value);
                    last_pos = i + 1;
                    node = root;
                    break; 
                }
                tmp = tmp->fail;
            }
            // printf("After processing char %d at index %zu, last_pos=%zu node is null=%d\n", x, i, last_pos, node == nullptr);
        }

        if (last_pos < n)
            results.emplace_back(-1, last_pos, n);

        return results;
    }

    // ===== pickle 支持 =====
    py::tuple __getstate__() const {
        std::vector<int> keys;
        std::vector<int> values;
        std::vector<int> end_values;  // 序列化 end_value
        std::vector<std::vector<int>> children;
        serialize_node(root, keys, values, end_values, children);
        return py::make_tuple(keys, values, end_values, children, idx);
    }

    void __setstate__(py::tuple t) {
        auto keys = t[0].cast<std::vector<int>>(); 
        auto values = t[1].cast<std::vector<int>>(); 
        auto end_values = t[2].cast<std::vector<int>>(); 
        auto children = t[3].cast<std::vector<std::vector<int>>>();
        auto idx_ = t[4].cast<int>();
        deserialize_node(keys, values, end_values, children);
        idx = idx_;
        build();
    }

    struct ACNode {
        std::unordered_map<int, ACNode*> children;
        ACNode* fail = nullptr;
        ACNode* link = nullptr;
        int value = -1, uuid = 0, end_value = -1;
        size_t depth = 0;
    };

    ACNode* root;

    void clear(ACNode* node) {
        for (auto& [k, child] : node->children)
            clear(child);
        delete node;
    }

    // ===== 序列化/反序列化节点 =====
    void serialize_node(ACNode* node, std::vector<int>& keys, std::vector<int>& values, std::vector<int>& end_values, std::vector<std::vector<int>>& children) const {
        if (keys.size() < node->uuid + 1) {
            keys.resize(node->uuid + 1);
            values.resize(node->uuid + 1);
            end_values.resize(node->uuid + 1);  // 为 end_value 分配空间
            children.resize(node->uuid + 1);
        }
        keys[node->uuid] = node->value;
        values[node->uuid] = static_cast<int>(node->depth);
        end_values[node->uuid] = node->end_value;  // 序列化 end_value

        // printf("now serializing node uuid=%d, value=%d, depth=%zu, end_value=%d\n", node->uuid, node->value, node->depth, node->end_value);
        std::vector<int> child_keys;
        for (auto& [k, child] : node->children) {
            assert(k == child->value);
            // printf("Serializing parent uuid=%d to child uuid=%d, from %d to %d\n", node->uuid, child->uuid, node->value, child->value);
            child_keys.push_back(child->uuid);
        }
        children[node->uuid] = child_keys;

        for (auto& [k, child] : node->children) {
            serialize_node(child, keys, values, end_values, children);
        }
    }

    void deserialize_node(const std::vector<int>& keys, const std::vector<int>& values, const std::vector<int>& end_values, const std::vector<std::vector<int>>& children) {
        std::vector<ACNode*> nodes(keys.size(), nullptr);
        for (size_t i = 0; i < keys.size(); ++i) {
            ACNode* node = new ACNode();
            node->value = keys[i];
            node->depth = values[i];
            node->end_value = end_values[i];  
            nodes[i] = node;
            node->uuid = i;
        }

        size_t idx = 0;
        root = nodes[0];
        for (auto& child_keys : children) {
            ACNode* parent = nodes[idx++];
            parent->fail = root; 
            // printf("now deserializing node uuid=%d, value=%d, depth=%zu, end_value=%d\n", parent->uuid, parent->value, parent->depth, parent->end_value);
            for (int key : child_keys) {
                ACNode* child = nodes[key];
                if (child) {
                    // printf("Linking parent uuid=%d to child uuid=%d, from %d to %d\n", parent->uuid, child->uuid, parent->value, child->value);
                    parent->children[child->value] = child;
                }
            }
        }

    }
};





PYBIND11_MODULE(decode, m) {
    py::class_<HybridDecoder>(m, "HybridDecoder")
        .def(py::init<size_t>(), py::arg("k"))
        .def("add_pattern", &HybridDecoder::add_pattern)
        .def("search", &HybridDecoder::search)
        .def(py::pickle(
            [](const HybridDecoder& self) {
                return self.__getstate__();
            },
            [](py::tuple t) {
                auto hd = new HybridDecoder(HybridDecoder::__setstate__(t));
                return hd;
            }
        ));
    py::class_<ACAutomaton>(m, "ACAutomaton")
        .def(py::init<>())
        .def("add_pattern", &ACAutomaton::add_pattern)
        .def("build", &ACAutomaton::build)
        .def("search", &ACAutomaton::search)
        .def(py::pickle(
            [](const ACAutomaton& self) {
                return self.__getstate__();
            },
            [](py::tuple t) {
                auto ac = new ACAutomaton();
                ac->__setstate__(t);
                return ac;
            }
        ));
}