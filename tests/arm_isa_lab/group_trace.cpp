#include "group_trace.hpp"
#include "nss/cpu_ncsr.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
thread_local int bm_depth = 0;
thread_local int twsc_depth = 0;
struct BmScope { BmScope() { ++bm_depth; } ~BmScope() { --bm_depth; } };
struct TwscScope { TwscScope() { ++twsc_depth; } ~TwscScope() { --twsc_depth; } };
struct State {
    std::ofstream output;
    std::mutex mutex;
    State() { if (const char* p = std::getenv("NSS_GROUP_RECORD")) { output.open(p, std::ios::binary); if (!output) throw std::runtime_error("group trace open failed"); } }
};
State& state() { static State s; return s; }
template<class T> void write(std::ofstream& out, const T* p, std::size_t count) {
    out.write(reinterpret_cast<const char*>(p), count * sizeof(T));
    if (!out) throw std::runtime_error("group trace write failed");
}
std::vector<float> pack(const float* p, int m, int n, int ld, int padded_n) {
    std::vector<float> result(m * padded_n, 0.f);
    if (p) for (int col = 0; col < n; ++col) std::copy_n(p + col * ld, m, result.data() + col * m);
    return result;
}
struct Snapshot { std::vector<float> input, reference; };
}

int nss_trace_bm_groups(nss::Bm3dFilterBatchItem* items, int count,
                        int (*run)(nss::Bm3dFilterBatchItem*, int)) {
    auto& s = state();
    if (!s.output.is_open() || count <= 0) return run(items, count);
    std::lock_guard lock(s.mutex);
    std::vector<Snapshot> snapshots(count);
    for (int i = 0; i < count; ++i) {
        const auto& item = items[i]; const int m = item.block * item.block;
        snapshots[i].input = pack(item.patches, m, item.k, item.lda, item.group);
        snapshots[i].reference = pack(item.ref_patches, m, item.k, item.lda, item.group);
    }
    int status;
    { BmScope scope; status = run(items, count); }
    const std::array<int, 4> header{0x4e534747, 1, count, status}; write(s.output, header.data(), header.size());
    for (int i = 0; i < count; ++i) {
        const auto& item = items[i]; const int m = item.block * item.block;
        const std::array<int, 5> shape{m, item.group, item.k, item.lda, int(item.wiener)};
        const std::array<float, 2> scalars{item.sigma, *item.weight};
        write(s.output, shape.data(), shape.size()); write(s.output, scalars.data(), scalars.size());
        write(s.output, snapshots[i].input.data(), m * item.group);
        write(s.output, snapshots[i].reference.data(), m * item.group);
        const auto result = pack(item.lda == m && item.k == item.group ? item.patches : item.work,
                                 m, item.group, m, item.group);
        write(s.output, result.data(), result.size());
    }
    return status;
}

void nss_trace_bm_call(float* patches, int lda, int group, int k, int block, float sigma, bool wiener,
                       const float* reference, float* weight, float* work, const std::function<void()>& run) {
    auto& s = state();
    if (!s.output.is_open() || bm_depth) { run(); return; }
    std::lock_guard lock(s.mutex);
    const int m = block * block;
    const auto input = pack(patches, m, k, lda, group), ref = pack(reference, m, k, lda, group);
    run();
    const auto result = pack(lda == m && k == group ? patches : work, m, group, m, group);
    const std::array<int, 4> header{0x4e534747, 1, 1, 0};
    const std::array<int, 5> shape{m, group, k, lda, int(wiener)};
    const std::array<float, 2> scalars{sigma, *weight};
    write(s.output, header.data(), header.size()); write(s.output, shape.data(), shape.size());
    write(s.output, scalars.data(), scalars.size());
    for (const auto* values : {&input, &ref, &result}) write(s.output, values->data(), values->size());
}

int nss_trace_ncsr_groups(nss::NcsrFilterBatchItem* items, int count,
                          int (*run)(nss::NcsrFilterBatchItem*, int)) {
    auto& s = state();
    if (!s.output.is_open() || count <= 0) return run(items, count);
    std::lock_guard lock(s.mutex);
    std::vector<Snapshot> snapshots(count);
    for (int i = 0; i < count; ++i) {
        const auto& item = items[i];
        snapshots[i].input = pack(item.group, item.m, item.n, item.lda, item.n);
        snapshots[i].reference = std::vector<float>(item.n, 0.f);
        if (item.col_dist) std::copy_n(item.col_dist, item.n, snapshots[i].reference.data());
    }
    const int status = run(items, count);
    const std::array<int, 4> header{0x4e534747, 5, count, status}; write(s.output, header.data(), header.size());
    for (int i = 0; i < count; ++i) {
        const auto& item = items[i];
        const std::array<int, 5> shape{item.m, item.n, item.n, item.lda, int(item.col_dist != nullptr)};
        const std::array<float, 2> scalars{item.sigma, 0.f};
        write(s.output, shape.data(), shape.size()); write(s.output, scalars.data(), scalars.size());
        write(s.output, snapshots[i].input.data(), item.m * item.n);
        write(s.output, snapshots[i].reference.data(), item.n);
        const auto result = pack(item.group, item.m, item.n, item.lda, item.n);
        write(s.output, result.data(), result.size());
        write(s.output, item.work, item.m * item.n + item.n);  // Actual U/S from single or batch route.
        std::vector<float> weights(item.n);
        if (!item.col_dist) throw std::runtime_error("NCSR weight trace requires explicit distances");
        const float h = std::max(2.f * float(item.m) * item.sigma * item.sigma, 1e-12f);
        nss::ncsr_group_weights(item.col_dist, nullptr, item.m, item.n, item.lda, h, weights.data());
        write(s.output, weights.data(), weights.size());
    }
    return status;
}

int nss_trace_mc_groups(nss::McwnnmFilterBatchItem* items, int count,
                        int (*run)(nss::McwnnmFilterBatchItem*, int)) {
    auto& s = state();
    if (!s.output.is_open() || count <= 0) return run(items, count);
    std::lock_guard lock(s.mutex);
    std::vector<std::vector<float>> input(count);
    for (int i = 0; i < count; ++i) input[i] = pack(items[i].group, items[i].m, items[i].n, items[i].lda, items[i].n);
    const int status = run(items, count);
    const std::array<int, 4> header{0x4e534747, 3, count, status}; write(s.output, header.data(), header.size());
    for (int i = 0; i < count; ++i) {
        const auto& a = items[i];
        const std::array<int, 8> shape{a.m, a.n, a.lda, a.nch, a.admm_iter, a.residual, a.adaptive, int(a.avx2_gemm)};
        const std::array<float, 3> values{a.rho, a.mu, a.adaptive_weight ? *a.adaptive_weight : 1.f};
        write(s.output, shape.data(), shape.size()); write(s.output, values.data(), values.size());
        write(s.output, a.sigma, a.nch); write(s.output, input[i].data(), input[i].size());
        const auto output = pack(a.group, a.m, a.n, a.lda, a.n);
        write(s.output, output.data(), output.size());
    }
    return status;
}

int nss_trace_twsc_groups(LegacyTwscTraceItem* items, int count,
                          int (*run)(LegacyTwscTraceItem*, int)) {
    auto& s = state();
    if (!s.output.is_open() || count <= 0) return run(items, count);
    std::lock_guard lock(s.mutex);
    std::vector<std::vector<float>> input(count);
    for (int i = 0; i < count; ++i) input[i] = pack(items[i].group, items[i].m, items[i].n, items[i].lda, items[i].n);
    int status;
    { TwscScope scope; status = run(items, count); }
    const std::array<int, 4> header{0x4e534747, 4, count, status}; write(s.output, header.data(), header.size());
    for (int i = 0; i < count; ++i) {
        const auto& a = items[i];
        const std::array<int, 5> shape{a.m, a.n, a.n, a.lda, int(a.row_weight != nullptr)};
        const std::array<float, 2> values{a.sigma, 0.f};
        const std::vector<float> cols = a.col_sigma ? std::vector<float>(a.col_sigma, a.col_sigma + a.n) : std::vector<float>(a.n, a.sigma);
        const std::vector<float> rows = a.row_weight ? std::vector<float>(a.row_weight, a.row_weight + a.m) : std::vector<float>(a.m, 1.f);
        write(s.output, shape.data(), shape.size()); write(s.output, values.data(), values.size());
        write(s.output, input[i].data(), input[i].size());
        write(s.output, cols.data(), cols.size()); write(s.output, rows.data(), rows.size());
        const auto output = pack(a.group, a.m, a.n, a.lda, a.n);
        write(s.output, output.data(), output.size());
        write(s.output, a.work, a.m * a.n + a.n);
    }
    return status;
}

int nss_trace_twsc_call(float* group, int m, int n, int lda, float sigma, float* work, int work_floats,
                         const float* col_sigma, float* col_weight, const float* row_weight,
                         const std::function<int()>& run) {
    auto& s = state();
    if (!s.output.is_open() || twsc_depth) return run();
    std::lock_guard lock(s.mutex);
    const auto input = pack(group, m, n, lda, n);
    const std::vector<float> cols = col_sigma ? std::vector<float>(col_sigma,col_sigma+n) : std::vector<float>(n,sigma);
    const std::vector<float> rows = row_weight ? std::vector<float>(row_weight,row_weight+m) : std::vector<float>(m,1.f);
    const int status = run();
    const auto output = pack(group,m,n,lda,n);
    const std::array<int,4> header{0x4e534747,4,1,status};
    const std::array<int,5> shape{m,n,n,lda,int(row_weight!=nullptr)};
    const std::array<float,2> values{sigma,0.f};
    write(s.output,header.data(),header.size());write(s.output,shape.data(),shape.size());write(s.output,values.data(),values.size());
    for (const auto* data : {&input,&cols,&rows,&output}) write(s.output,data->data(),data->size());
    write(s.output,work,m*n+n);
    return status;
}
