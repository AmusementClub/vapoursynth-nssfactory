// SPDX-License-Identifier: GPL-2.0-only
// Deterministic block-wide top-K over match keys. A key is a total order on
// (finite first, distance, t, y, x), so the selected set and its order do not
// depend on thread scheduling (D17). Candidates stream through a fixed-size
// shared buffer: the current top-K occupies [0, K) and each chunk fills the
// rest before one bitonic sort.
#pragma once

#include <cstdint>

namespace nss_cuda::detail {

struct MatchKey {
    unsigned long long hi;  // sortable distance (32) | t (32)
    unsigned long long lo;  // y (32) | x (32)
};

__device__ __forceinline__ bool key_less(const MatchKey& a, const MatchKey& b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

__device__ __forceinline__ MatchKey sentinel_key() {
    return MatchKey{~0ull, ~0ull};
}

// CPU match_less: non-finite distances sort after every finite one and among
// themselves by position; +0 and -0 compare equal.
__device__ __forceinline__ unsigned sortable_distance(float d) {
    unsigned bits = __float_as_uint(d);
    if ((bits & 0x7f800000u) == 0x7f800000u) return 0xfffffffeu;
    if (d == 0.f) bits = 0u;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ MatchKey make_key(float dist, int t, int y, int x) {
    return MatchKey{(static_cast<unsigned long long>(sortable_distance(dist)) << 32) | static_cast<unsigned>(t),
                    (static_cast<unsigned long long>(static_cast<unsigned>(y)) << 32) | static_cast<unsigned>(x)};
}

__device__ __forceinline__ int key_t(const MatchKey& k) { return static_cast<int>(k.hi & 0xffffffffu); }
__device__ __forceinline__ int key_y(const MatchKey& k) { return static_cast<int>(k.lo >> 32); }
__device__ __forceinline__ int key_x(const MatchKey& k) { return static_cast<int>(k.lo & 0xffffffffu); }

// Ascending bitonic sort of N (power of two) keys in shared memory.
template <int N>
__device__ void bitonic_sort(MatchKey* keys) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int partner = i ^ j;
                if (partner > i) {
                    const bool ascending = (i & k) == 0;
                    MatchKey a = keys[i];
                    MatchKey b = keys[partner];
                    if (key_less(b, a) == ascending) {
                        keys[i] = b;
                        keys[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Streaming top-K: the current result occupies [0, k) (k <= 64) and each
// chunk of candidates is written to [64, N) before merge(). After the last
// merge, entries [0, filled) are the selected keys in ascending order.
template <int N>
struct BlockTopK {
    static constexpr int kChunk = N - 64;
    MatchKey* keys;
    int k;
    int filled;

    __device__ void init(MatchKey* storage, int wanted) {
        keys = storage;
        k = wanted;
        filled = 0;
        for (int i = threadIdx.x; i < N; i += blockDim.x) keys[i] = sentinel_key();
        __syncthreads();
    }
    __device__ MatchKey& candidate(int i) { return keys[64 + i]; }
    __device__ void merge() {
        bitonic_sort<N>(keys);
        int valid = 0;
        while (valid < k && (keys[valid].hi != ~0ull || keys[valid].lo != ~0ull)) ++valid;
        filled = valid;
        __syncthreads();
        for (int i = threadIdx.x + k; i < N; i += blockDim.x) keys[i] = sentinel_key();
        __syncthreads();
    }
};

__device__ __forceinline__ float key_distance(const MatchKey& key) {
    const unsigned s = static_cast<unsigned>(key.hi >> 32);
    if (s == 0xfffffffeu) return __int_as_float(0x7f800000);  // non-finite sorts as +inf
    const unsigned bits = (s & 0x80000000u) ? (s & 0x7fffffffu) : ~s;
    return __uint_as_float(bits);
}

}  // namespace nss_cuda::detail
