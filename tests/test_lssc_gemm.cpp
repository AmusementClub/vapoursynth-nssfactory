#include "nss/cpu_lssc.hpp"
#include "cpu/lssc/gemm.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
constexpr float canary = -9876.5f;
struct Guarded {
    float* data;
    std::vector<float> fallback;
#if defined(__unix__) || defined(__APPLE__)
    void* mapping;
    std::size_t bytes;
    explicit Guarded(int count) {
        const std::size_t page = sysconf(_SC_PAGESIZE);
        const std::size_t active = ((std::size_t(count) + 16) * sizeof(float) + page - 1) / page * page;
        bytes = active + page;
        mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mapping == MAP_FAILED || mprotect(static_cast<char*>(mapping) + active, page, PROT_NONE)) std::abort();
        data = reinterpret_cast<float*>(static_cast<char*>(mapping) + active) - count;
        std::fill(data - 16, data + count, canary);
    }
    ~Guarded() { munmap(mapping, bytes); }
#else
    explicit Guarded(int count) : fallback(count + 16, canary) { data = fallback.data() + 16; }
#endif
    bool prefix_ok() const { return std::all_of(data - 16, data, [](float v) { return v == canary; }); }
};

int extent(int rows, int columns, int stride) { return columns ? (columns - 1) * stride + rows : 0; }
float sample(unsigned value) {
    value = value * 1664525u + 1013904223u;
    value ^= value >> 13;
    return float(int(value % 100003u) - 50001) / 50003.f;
}

bool product(int m, int n, int k, float scale, bool short_work, int& sme, int& exact) {
    const int lda = m + 3, ldb = k + 5, ldc = m + 7;
    Guarded a(extent(m, k, lda)), b(extent(k, n, ldb));
    Guarded c(extent(m, n, ldc)), reference(extent(m, n, ldc));
    const int need = nss::lssc_gemm_pack_work_floats(m, n, k);
    Guarded scratch(std::max(1, need));
    for (int col = 0; col < k; ++col) for (int row = 0; row < m; ++row)
        a.data[col * lda + row] = scale * sample(row + 37 * col);
    for (int col = 0; col < n; ++col) for (int row = 0; row < k; ++row)
        b.data[col * ldb + row] = sample(row + 19 * col + 7);
    const int capacity = short_work ? std::max(0, need - 1) : need;
    const bool selected = nss::lssc_gemm_nn(m, n, k, k ? a.data : nullptr, lda,
        k ? b.data : nullptr, ldb, c.data, ldc, scratch.data, capacity);
    if (selected != (need != 0 && !short_work)) return false;
    sme += selected;
    nss::gemm_nn_hwy(m, n, k, k ? a.data : nullptr, lda, k ? b.data : nullptr, ldb,
                     reference.data, ldc);
    exact += std::memcmp(c.data, reference.data, extent(m, n, ldc) * sizeof(float)) == 0;
    const double unit = std::numeric_limits<float>::epsilon();
    const double gamma = 2 * k * unit / (1 - 2 * k * unit);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double expected = 0, absolute = 0;
            for (int t = 0; t < k; ++t) {
                const double term = double(a.data[t * lda + row]) * b.data[col * ldb + t];
                expected += term; absolute += std::abs(term);
            }
            if (!std::isfinite(c.data[col * ldc + row]) ||
                std::abs(double(c.data[col * ldc + row]) - expected) > gamma * absolute + 1e-30) return false;
        }
        if (col + 1 < n) for (int row = m; row < ldc; ++row)
            if (c.data[col * ldc + row] != canary) return false;
    }
    return a.prefix_ok() && b.prefix_ok() && c.prefix_ok() && scratch.prefix_ok();
}
}

int main() {
    // Exercise first-use capability initialization and independent ZA/scratch
    // state from concurrent ordinary C++ callers.
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) workers.emplace_back([&, i] {
        int selected = 0, exact = 0;
        for (int call = 0; call < 3; ++call)
            if (!product(64, 129, 64, float(i + 1), false, selected, exact)) ++errors;
    });
    for (auto& worker : workers) worker.join();
    if (errors) return 1;
    int packs = 0;
    for (int rows : {0, 1, 3, 4, 7, 16, 65, 256}) for (int columns : {0, 1, 3, 4, 9, 33, 129}) {
        const int stride = rows + 3;
        Guarded input(extent(rows, columns, stride)), packed(rows * columns);
        for (int col = 0; col < columns; ++col) for (int row = 0; row < rows; ++row)
            input.data[col * stride + row] = sample(row + col * 13);
        nss::lssc_pack_columns_hwy(input.data, rows, columns, stride, packed.data);
        for (int row = 0; row < rows; ++row) for (int col = 0; col < columns; ++col)
            if (packed.data[row * columns + col] != input.data[col * stride + row]) return 2;
        if (!input.prefix_ok() || !packed.prefix_ok()) return 3;
        ++packs;
    }
    int cases = 0, sme = 0, exact = 0;
    const int shapes[][3] = {{16, 129, 64}, {64, 1, 256}, {64, 127, 64}, {64, 128, 64},
        {64, 129, 256}, {256, 257, 64}, {64, 947, 256}, {65, 129, 64}, {64, 129, 65}, {64, 129, 0}};
    for (const auto& shape : shapes) for (float scale : {1e-8f, 1.f, 1e8f}) {
        if (!product(shape[0], shape[1], shape[2], scale, false, sme, exact)) return 4;
        ++cases;
    }
    if (!product(64, 129, 256, 1.f, true, sme, exact)) return 5;
    ++cases;
    if (nss::lssc_sme_available() && sme == 0) return 6;
    std::printf("pack_cases=%d product_cases=%d byte_exact=%d sme_products=%d SME_available=%d concurrent_calls=24\n",
                 packs, cases, exact, sme, int(nss::lssc_sme_available()));
    return 0;
}
