#include <vector>
#include <unordered_set>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

PYBIND11_MODULE(decode, m) {
    py::class_<HybridDecoder>(m, "HybridDecoder")
        .def(py::init<size_t, const std::vector<std::vector<int>>&>())
        .def("decode", &HybridDecoder::decode, py::arg("nums"), py::arg("strict") = false)
        .def(py::pickle(
            [](const HybridDecoder &self) { // __getstate__
                std::vector<uint64_t> keys1(self.map1.begin(), self.map1.end());
                std::vector<uint64_t> keys2(self.map2.begin(), self.map2.end());
                return py::make_tuple(self.k, keys1, keys2);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");
                size_t k_ = t[0].cast<size_t>();
                std::vector<uint64_t> keys1 = t[1].cast<std::vector<uint64_t>>();
                std::vector<uint64_t> keys2 = t[2].cast<std::vector<uint64_t>>();
                auto dec = new HybridDecoder(k_, {});
                dec->map1.insert(keys1.begin(), keys1.end());
                dec->map2.insert(keys2.begin(), keys2.end());
                return dec; 
            }
        ));
}