// SPDX-License-Identifier: GPL-2.0-only
// Deterministic block-wide top-K over match keys. A key is a total order on
// (finite first, distance, t, y, x), so the selected set and its order do not
// depend on thread scheduling. Candidates stream through a fixed-size
// shared buffer: the current top-K occupies [0, K) and each chunk fills the
// rest before one sort: a radix sort of the key packed into 64 bits where the
// coordinates allow it (1.28-1.30x the bitonic network on the predictive
// matcher, docs/cuda-tools-all-20261017.md T03), else the bitonic network.
#pragma once

#include <cub/block/block_radix_sort.cuh>

#include <cstddef>
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

// Ascending bitonic sort of N (power of two) keys in shared memory; every
// pass assigns one compare-exchange pair per thread slot.
template <int N>
__device__ void bitonic_sort(MatchKey* keys) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int pair = threadIdx.x; pair < N / 2; pair += blockDim.x) {
                const int i = ((pair & ~(j - 1)) << 1) | (pair & (j - 1));  // j is a power of two
                const int partner = i + j;
                const bool ascending = (i & k) == 0;
                const MatchKey a = keys[i];
                const MatchKey b = keys[partner];
                if (key_less(b, a) == ascending) {
                    keys[i] = b;
                    keys[partner] = a;
                }
            }
            __syncthreads();
        }
    }
}

__host__ __device__ constexpr int pow2_at_least(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// The key packed as distance (32) | t (6) | y (13) | x (13): the same order as
// (hi, lo) while t < 64 and y, x < 8192. The sentinel packs to ~0, which no
// real key reaches (sortable distances end at 0xfffffffe).
inline constexpr int kPackedMaxCoordinate = 8192;
inline constexpr int kPackedMaxT = 64;
__device__ __forceinline__ unsigned long long pack_key(const MatchKey& k) {
    if (k.hi == ~0ull && k.lo == ~0ull) return ~0ull;
    return (k.hi & 0xffffffff00000000ull) | (k.hi & 0x3full) << 26 | (k.lo >> 32 & 0x1fffull) << 13 | (k.lo & 0x1fffull);
}
__device__ __forceinline__ MatchKey unpack_key(unsigned long long v) {
    if (v == ~0ull) return sentinel_key();
    return MatchKey{(v & 0xffffffff00000000ull) | (v >> 26 & 0x3full), (v >> 13 & 0x1fffull) << 32 | (v & 0x1fffull)};
}
// Whether every key of a search packs (the frame count: 1 for spatial).
__host__ __device__ constexpr bool keys_pack(int width, int height, int frames) {
    return width <= kPackedMaxCoordinate && height <= kPackedMaxCoordinate && frames <= kPackedMaxT;
}

// Streaming top-K over a shared buffer of N keys: the current result occupies
// the front (pow2_at_least(k) slots, k < N) and each chunk of candidates is
// written behind it before merge(). After the last merge, entries
// [0, filled) are the selected keys in ascending order. Blocks have Threads
// threads (the radix sort's shape).
template <int N, int Threads = 128>
struct BlockTopK {
    static_assert(N % Threads == 0, "whole keys a thread");
    using RadixSort = cub::BlockRadixSort<unsigned long long, Threads, N / Threads>;
    // Static shared memory merge() adds to its kernel.
    static constexpr std::size_t kSortBytes = sizeof(typename RadixSort::TempStorage);
    MatchKey* keys;
    int k;
    int front;
    int chunk;
    int filled;
    bool packed;

    __device__ void init(MatchKey* storage, int wanted, bool pack) {
        keys = storage;
        k = wanted;
        front = pow2_at_least(wanted > 0 ? wanted : 1);
        chunk = N - front;
        filled = 0;
        packed = pack;
        for (int i = threadIdx.x; i < N; i += blockDim.x) keys[i] = sentinel_key();
        __syncthreads();
    }
    __device__ MatchKey& candidate(int i) { return keys[front + i]; }
    __device__ void merge() {
        __shared__ typename RadixSort::TempStorage sort_storage;
        if (packed) {
            constexpr int kItems = N / Threads;
            unsigned long long items[kItems];
#pragma unroll
            for (int i = 0; i < kItems; ++i) items[i] = pack_key(keys[threadIdx.x * kItems + i]);
            RadixSort(sort_storage).Sort(items);
#pragma unroll
            for (int i = 0; i < kItems; ++i) keys[threadIdx.x * kItems + i] = unpack_key(items[i]);
            __syncthreads();
        } else {
            bitonic_sort<N>(keys);
        }
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
