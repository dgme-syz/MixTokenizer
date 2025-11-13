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

    std::unordered_set<uint64_t> map1;
    std::unordered_set<uint64_t> map2;

    HybridDecoder(size_t k_, const std::vector<std::vector<int>>& exist_seq) : k(k_) {
        for (const auto& seq : exist_seq) {
            if (seq.size() != k) continue;
            uint64_t h1 = 0, h2 = 0;
            for (size_t i = 0; i < k; ++i) {
                h1 = (h1 * base1 + seq[i]) % mod1;
                h2 = (h2 * base2 + seq[i]) % mod2;
            }
            map1.insert(h1);
            map2.insert(h2);
            // if (seq == std::vector<int>({186, 195})) {
            //     printf("Inserted special seq {186, 195} with h1=%llu, h2=%llu\n", h1, h2);
            // }
        }
    }

    std::vector<std::vector<size_t>> decode(const std::vector<int>& nums, bool strict=false) const {
        std::vector<std::vector<size_t>> intervals;
        size_t n = nums.size();
        if (n == 0) return intervals;

        if (n < k) {
            // everything is considered non-flag (0), covering [0, n)
            intervals.push_back({0, 0, n});
            return intervals;
        }

        uint64_t h1 = 0, h2 = 0;
        uint64_t base_pow1 = 1, base_pow2 = 1;

        for (size_t i = 0; i < k - 1 && i < n; ++i) {
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
                h1 = (h1 + mod1 - (nums[i - 1] * base_pow1) % mod1) % mod1;
                h1 = (h1 * base1 + nums[i + k - 1]) % mod1;
 
                h2 = (h2 + mod2 - (nums[i - 1] * base_pow2) % mod2) % mod2;
                h2 = (h2 * base2 + nums[i + k - 1]) % mod2;
            }
            if (i < work) continue;
            bool flag = map1.count(h1) && map2.count(h2);
            // printf("At position %zu: h1=%llu, h2=%llu, flag=%d list=%d, %d map1.count=%zu, map2.count=%zu\n", i, h1, h2, flag, nums[i], nums[i+1], map1.count(h1), map2.count(h2));
            if (intervals.empty()) {
                if (flag) intervals.push_back({1, i, i + k}), work = i + k;
                else {
                    intervals.push_back({0, i, i + 1}), work = i + 1;
                }
            } else {
                auto& last = intervals.back();
                if (last[0] == size_t(flag)) {
                    if (flag) last[2] = i + k, work = i + k;
                    else last[2] = i + 1, work = i + 1;
                } else {
                    if (flag) intervals.push_back({1, i, i + k}), work = i + k;
                    else {
                        if (last[2] - last[1] > k || !strict) {
                            intervals.push_back({0, i, i + 1});
                        } else {
                            last[0] = 0, last[2] = i + 1;
                        }
                        work = i + 1;
                    }
                }
            }
        }

        // Handle remaining tokens
        size_t processed = 0;
        if (!intervals.empty()) {
            processed = intervals.back()[2];
        }
        for (size_t i = processed; i < n; ++i) {
            auto& last = intervals.back();
            if (last[0] == 0) {
                last[2] = i + 1;
            } else {
                intervals.push_back({0, i, i + 1});
            }
        }

        return intervals;
    }

    py::tuple __getstate__() const {
        std::vector<uint64_t> keys1(map1.begin(), map1.end());
        std::vector<uint64_t> keys2(map2.begin(), map2.end());
        return py::make_tuple(k, keys1, keys2);
    }

    static HybridDecoder __setstate__(py::tuple t) {
        if (t.size() != 3) throw std::runtime_error("Invalid state!");
        size_t k_ = t[0].cast<size_t>();
        std::vector<uint64_t> keys1 = t[1].cast<std::vector<uint64_t>>();
        std::vector<uint64_t> keys2 = t[2].cast<std::vector<uint64_t>>();
        HybridDecoder dec(k_, {});
        dec.map1.insert(keys1.begin(), keys1.end());
        dec.map2.insert(keys2.begin(), keys2.end());
        return dec;
    }
};


// struct ACNode {
//     std::unordered_map<int, ACNode*> children;
//     ACNode* fail = nullptr;
//     ACNode* link = nullptr; // jump fail link to next output node 
//     int value = -1; // -1 non-leaf
//     int depth = 0;
// };

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
            // printf("At text index %zu, char=%d, current node uuid=%d, value=%d\n", i, x, node->uuid, node->value);
            ACNode* tmp = node;
            // if (tmp && tmp->link) tmp = tmp->link;
            while (tmp != root && tmp != nullptr) {
                if (tmp->end_value != -1) {
                    node->link = tmp;
                    size_t start = i + 1 - tmp->depth;
                    if (last_pos < start) {
                        results.emplace_back(-1, last_pos, start);
                        
                    }
                    results.emplace_back(tmp->end_value, start, i + 1);
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
            node->end_value = end_values[i];  // 反序列化 end_value
            nodes[i] = node;
            node->uuid = i;
        }

        size_t idx = 0;
        for (auto& child_keys : children) {
            ACNode* parent = nodes[idx++];
            // printf("now deserializing node uuid=%d, value=%d, depth=%zu, end_value=%d\n", parent->uuid, parent->value, parent->depth, parent->end_value);
            for (int key : child_keys) {
                ACNode* child = nodes[key];
                if (child) {
                    // printf("Linking parent uuid=%d to child uuid=%d, from %d to %d\n", parent->uuid, child->uuid, parent->value, child->value);
                    parent->children[child->value] = child;
                }
            }
        }

        root = nodes[0];
    }
};





PYBIND11_MODULE(decode, m) {
    py::class_<HybridDecoder, std::shared_ptr<HybridDecoder>>(m, "HybridDecoder")
        .def(py::init<size_t, const std::vector<std::vector<int>>&>())
        .def("decode", &HybridDecoder::decode, py::arg("nums"), py::arg("strict") = false)
        .def(py::pickle(
            [](const HybridDecoder &self) {
                std::vector<uint64_t> keys1(self.map1.begin(), self.map1.end());
                std::vector<uint64_t> keys2(self.map2.begin(), self.map2.end());
                return py::make_tuple(self.k, keys1, keys2);
            },
            [](py::tuple t) {
                size_t k_ = t[0].cast<size_t>();
                std::vector<uint64_t> keys1 = t[1].cast<std::vector<uint64_t>>();
                std::vector<uint64_t> keys2 = t[2].cast<std::vector<uint64_t>>();
                auto dec = std::make_shared<HybridDecoder>(k_, std::vector<std::vector<int>>{});
                dec->map1.insert(keys1.begin(), keys1.end());
                dec->map2.insert(keys2.begin(), keys2.end());
                return dec; 
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