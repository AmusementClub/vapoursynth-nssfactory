#include "nss/cpu_batch.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_nlh.hpp"
#include "nss/cpu_twsc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

bool close_float(float a, float b, float tol = 3e-5f) {
    const float scale = std::max({1.f, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= tol * scale;
}

bool compare_array(const float* a, const float* b, std::size_t n, float tol = 3e-5f) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!close_float(a[i], b[i], tol)) {
            std::fprintf(stderr, "batch value mismatch at %zu: %.9g != %.9g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

bool compare_matches(const nss::Match* a, const nss::Match* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].t != b[i].t || a[i].ordinal != b[i].ordinal ||
            !close_float(a[i].dist, b[i].dist, 2e-5f)) {
            std::fprintf(stderr, "batch match mismatch at %d\n", i);
            return false;
        }
    }
    return true;
}

double svd_reconstruction_error(const std::vector<float>& a, int m, int lda, const std::vector<float>& u, int ldu,
                                const std::vector<float>& s, const std::vector<float>& vt, int ldvt) {
    double error2 = 0.0;
    double norm2 = 0.0;
    for (int col = 0; col < 8; ++col) {
        for (int row = 0; row < m; ++row) {
            double value = 0.0;
            for (int k = 0; k < 8; ++k) {
                value += static_cast<double>(u[static_cast<std::size_t>(row + k * ldu)]) * s[k] *
                         vt[static_cast<std::size_t>(k + col * ldvt)];
            }
            const double original = a[static_cast<std::size_t>(row + col * lda)];
            const double delta = value - original;
            error2 += delta * delta;
            norm2 += original * original;
        }
    }
    return std::sqrt(error2) / (std::sqrt(norm2) + 1e-30);
}

bool check_svd_lane_batches() {
    struct Buffers {
        int m = 0;
        int lda = 0;
        int ldu = 0;
        int ldvt = 0;
        std::vector<float> a;
        std::vector<float> scalar_u;
        std::vector<float> scalar_s;
        std::vector<float> scalar_vt;
        std::vector<float> batch_u;
        std::vector<float> batch_s;
        std::vector<float> batch_vt;
        std::vector<float> repeat_u;
        std::vector<float> repeat_s;
        std::vector<float> repeat_vt;
        std::vector<float> work;
        int status = 0;
    };

    constexpr int counts[] = {1, 2, 7, 8, 9, 15, 16, 17, 31, 32};
    constexpr int rows[] = {8, 16, 48, 64};
    for (int m : rows) {
        for (int count : counts) {
            std::vector<Buffers> matrices(static_cast<std::size_t>(count));
            std::vector<nss::SvdBatchItem> items(static_cast<std::size_t>(count));
            std::vector<nss::SvdBatchItem> repeats(static_cast<std::size_t>(count));
            for (int index = 0; index < count; ++index) {
                auto& b = matrices[static_cast<std::size_t>(index)];
                b.m = m;
                b.lda = m + index % 3;
                b.ldu = m + index % 2;
                b.ldvt = 8 + index % 3;
                b.a.assign(static_cast<std::size_t>(b.lda * 8), 0.f);
                for (int col = 0; col < 8; ++col) {
                    for (int row = 0; row < m; ++row) {
                        float value = std::sin(static_cast<float>((index + 3) * (row + 1) * (col + 2)) * 0.071f);
                        if (index % 5 == 1) {
                            value = static_cast<float>(row + 1) * static_cast<float>((col % 3) + 1) * 0.01f;
                        } else if (index % 5 == 2) {
                            value = row == col ? std::pow(10.f, -static_cast<float>(col)) : 0.f;
                        } else if (index % 5 == 3) {
                            value = (row == col ? 1.f : 0.f) + (row == 0 && col == 2 ? 1.f : 0.f);
                        }
                        b.a[static_cast<std::size_t>(row + col * b.lda)] = value;
                    }
                }
                b.scalar_u.assign(static_cast<std::size_t>(b.ldu * 8), 0.f);
                b.scalar_s.assign(8, 0.f);
                b.scalar_vt.assign(static_cast<std::size_t>(b.ldvt * 8), 0.f);
                b.batch_u.assign(static_cast<std::size_t>(b.ldu * 8), 0.f);
                b.batch_s.assign(8, 0.f);
                b.batch_vt.assign(static_cast<std::size_t>(b.ldvt * 8), 0.f);
                b.repeat_u.assign(static_cast<std::size_t>(b.ldu * 8), 0.f);
                b.repeat_s.assign(8, 0.f);
                b.repeat_vt.assign(static_cast<std::size_t>(b.ldvt * 8), 0.f);
                b.work.assign(static_cast<std::size_t>(m * 8 * 6 + 8 * 8 * 8 + 8 + 256), 0.f);
                if (nss::svd_economy(m, 8, b.a.data(), b.lda, b.scalar_u.data(), b.ldu, b.scalar_s.data(),
                                     b.scalar_vt.data(), b.ldvt, b.work.data(), static_cast<int>(b.work.size())) != 0) {
                    std::fprintf(stderr, "scalar SVD fixture failed for m=%d count=%d item=%d\n", m, count, index);
                    return false;
                }
                items[static_cast<std::size_t>(index)] =
                    nss::SvdBatchItem{m, 8, b.a.data(), b.lda, b.batch_u.data(), b.ldu, b.batch_s.data(),
                                      b.batch_vt.data(), b.ldvt, b.work.data(), static_cast<int>(b.work.size()), &b.status};
                repeats[static_cast<std::size_t>(index)] =
                    nss::SvdBatchItem{m, 8, b.a.data(), b.lda, b.repeat_u.data(), b.ldu, b.repeat_s.data(),
                                      b.repeat_vt.data(), b.ldvt, b.work.data(), static_cast<int>(b.work.size()), nullptr};
            }

            if (nss::svd_economy_batch(items.data(), count) != 0 || nss::svd_economy_batch(repeats.data(), count) != 0) {
                std::fprintf(stderr, "lane SVD batch failed for m=%d count=%d\n", m, count);
                return false;
            }
            for (int index = 0; index < count; ++index) {
                const auto& b = matrices[static_cast<std::size_t>(index)];
                if (b.status != 1 || !compare_array(b.scalar_s.data(), b.batch_s.data(), 8, 8e-4f)) {
                    std::fprintf(stderr, "lane SVD singular mismatch for m=%d count=%d item=%d\n", m, count, index);
                    return false;
                }
                const double residual =
                    svd_reconstruction_error(b.a, m, b.lda, b.batch_u, b.ldu, b.batch_s, b.batch_vt, b.ldvt);
                if (!(residual < 8e-4)) {
                    std::fprintf(stderr, "lane SVD residual %.9g for m=%d count=%d item=%d\n", residual, m, count,
                                 index);
                    return false;
                }
                if (std::memcmp(b.batch_u.data(), b.repeat_u.data(), b.batch_u.size() * sizeof(float)) != 0 ||
                    std::memcmp(b.batch_s.data(), b.repeat_s.data(), b.batch_s.size() * sizeof(float)) != 0 ||
                    std::memcmp(b.batch_vt.data(), b.repeat_vt.data(), b.batch_vt.size() * sizeof(float)) != 0) {
                    std::fprintf(stderr, "lane SVD is not deterministic for m=%d count=%d item=%d\n", m, count,
                                 index);
                    return false;
                }
            }
        }
    }
    return true;
}

bool check_wnnm_lane_batches() {
    struct Buffers {
        int m = 0;
        int lda = 0;
        std::vector<float> scalar;
        std::vector<float> batch;
        std::vector<float> repeat;
        std::vector<float> scalar_work;
        std::vector<float> batch_work;
        std::vector<float> repeat_work;
        float scalar_weight = 1.f;
        float batch_weight = 1.f;
        float repeat_weight = 1.f;
        int status = 0;
    };

    constexpr int counts[] = {2, 7, 8, 9, 16, 17, 32};
    for (int m : {16, 64}) {
        for (int residual : {0, 1}) {
            for (int count : counts) {
                std::vector<Buffers> groups(static_cast<std::size_t>(count));
                std::vector<nss::WnnmShrinkBatchItem> items(static_cast<std::size_t>(count));
                std::vector<nss::WnnmShrinkBatchItem> repeats(static_cast<std::size_t>(count));
                for (int index = 0; index < count; ++index) {
                    auto& b = groups[static_cast<std::size_t>(index)];
                    b.m = m;
                    b.lda = m + index % 3;
                    b.scalar.assign(static_cast<std::size_t>(b.lda * 8), 0.f);
                    for (int col = 0; col < 8; ++col) {
                        for (int row = 0; row < m; ++row) {
                            b.scalar[static_cast<std::size_t>(row + col * b.lda)] =
                                0.25f * std::sin(static_cast<float>((index + 2) * (row + 3) * (col + 1)) * 0.037f) +
                                (row == col ? 0.2f : 0.f);
                        }
                    }
                    b.batch = b.scalar;
                    b.repeat = b.scalar;
                    const int work_floats = nss::wnnm_shrink_work_floats(m, 8);
                    b.scalar_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                    b.batch_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                    b.repeat_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                    const float sigma = 0.01f + static_cast<float>(index % 4) * 0.001f;
                    if (nss::wnnm_shrink(b.scalar.data(), m, 8, b.lda, sigma, residual, 1, &b.scalar_weight,
                                         b.scalar_work.data(), work_floats) != 0) {
                        std::fprintf(stderr, "scalar WNNM fixture failed for m=%d count=%d item=%d\n", m, count,
                                     index);
                        return false;
                    }
                    items[static_cast<std::size_t>(index)] = nss::WnnmShrinkBatchItem{
                        b.batch.data(), m, 8, b.lda, sigma, residual, 1, &b.batch_weight, b.batch_work.data(),
                        work_floats, &b.status};
                    repeats[static_cast<std::size_t>(index)] = nss::WnnmShrinkBatchItem{
                        b.repeat.data(), m, 8, b.lda, sigma, residual, 1, &b.repeat_weight, b.repeat_work.data(),
                        work_floats, nullptr};
                }
                if (nss::wnnm_shrink_batch(items.data(), count) != 0 ||
                    nss::wnnm_shrink_batch(repeats.data(), count) != 0) {
                    std::fprintf(stderr, "lane WNNM batch failed for m=%d count=%d residual=%d\n", m, count,
                                 residual);
                    return false;
                }
                for (int index = 0; index < count; ++index) {
                    const auto& b = groups[static_cast<std::size_t>(index)];
                    if (b.status != 1 || !compare_array(b.scalar.data(), b.batch.data(), b.scalar.size(), 2e-3f) ||
                        !close_float(b.scalar_weight, b.batch_weight, 2e-3f)) {
                        std::fprintf(stderr, "lane WNNM mismatch for m=%d count=%d residual=%d item=%d\n", m,
                                     count, residual, index);
                        return false;
                    }
                    if (std::memcmp(b.batch.data(), b.repeat.data(), b.batch.size() * sizeof(float)) != 0 ||
                        b.batch_weight != b.repeat_weight) {
                        std::fprintf(stderr, "lane WNNM is not deterministic for m=%d count=%d residual=%d item=%d\n",
                                     m, count, residual, index);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool check_twsc_lane_batches() {
    struct Buffers {
        int m = 0;
        int lda = 0;
        std::vector<float> scalar;
        std::vector<float> batch;
        std::vector<float> repeat;
        std::vector<float> scalar_work;
        std::vector<float> batch_work;
        std::vector<float> repeat_work;
        std::array<float, 8> col_sigma{};
        std::array<float, 8> scalar_weight{};
        std::array<float, 8> batch_weight{};
        std::array<float, 8> repeat_weight{};
        std::vector<float> row_weight;
        int status = 0;
    };

    constexpr int counts[] = {2, 7, 8, 9, 16, 17, 32};
    for (int m : {16, 64, 65}) {
        for (int count : counts) {
            std::vector<Buffers> groups(static_cast<std::size_t>(count));
            std::vector<nss::TwscPcaBatchItem> items(static_cast<std::size_t>(count));
            std::vector<nss::TwscPcaBatchItem> repeats(static_cast<std::size_t>(count));
            for (int index = 0; index < count; ++index) {
                auto& b = groups[static_cast<std::size_t>(index)];
                b.m = m;
                b.lda = m + index % 3;
                b.scalar.assign(static_cast<std::size_t>(b.lda * 8), 0.f);
                for (int col = 0; col < 8; ++col) {
                    b.col_sigma[static_cast<std::size_t>(col)] = 0.008f + 0.001f * static_cast<float>((col + index) % 4);
                    for (int row = 0; row < m; ++row) {
                        b.scalar[static_cast<std::size_t>(row + col * b.lda)] =
                            0.2f * std::cos(static_cast<float>((index + 1) * (row + 2) * (col + 3)) * 0.029f) +
                            (row == col ? 0.15f : 0.f);
                    }
                }
                b.batch = b.scalar;
                b.repeat = b.scalar;
                b.row_weight.resize(static_cast<std::size_t>(m));
                for (int row = 0; row < m; ++row) {
                    b.row_weight[static_cast<std::size_t>(row)] = 0.75f + 0.05f * static_cast<float>(row % 7);
                }
                const int work_floats = nss::twsc_pca_soft_work_floats(m, 8);
                b.scalar_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                b.batch_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                b.repeat_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                const float* row_weight = index % 2 == 0 ? b.row_weight.data() : nullptr;
                if (nss::twsc_pca_soft(b.scalar.data(), m, 8, b.lda, 0.01f, b.scalar_work.data(), work_floats,
                                       b.col_sigma.data(), b.scalar_weight.data(), row_weight) != 0) {
                    std::fprintf(stderr, "scalar TWSC fixture failed for m=%d count=%d item=%d\n", m, count, index);
                    return false;
                }
                items[static_cast<std::size_t>(index)] = nss::TwscPcaBatchItem{
                    b.batch.data(), m, 8, b.lda, 0.01f, b.batch_work.data(), work_floats, b.col_sigma.data(),
                    b.batch_weight.data(), row_weight, &b.status};
                repeats[static_cast<std::size_t>(index)] = nss::TwscPcaBatchItem{
                    b.repeat.data(), m, 8, b.lda, 0.01f, b.repeat_work.data(), work_floats, b.col_sigma.data(),
                    b.repeat_weight.data(), row_weight, nullptr};
            }
            if (nss::twsc_pca_soft_batch(items.data(), count) != 0 ||
                nss::twsc_pca_soft_batch(repeats.data(), count) != 0) {
                std::fprintf(stderr, "lane TWSC batch failed for m=%d count=%d\n", m, count);
                return false;
            }
            for (int index = 0; index < count; ++index) {
                const auto& b = groups[static_cast<std::size_t>(index)];
                if (b.status != 1 || !compare_array(b.scalar.data(), b.batch.data(), b.scalar.size(), 3e-3f) ||
                    !compare_array(b.scalar_weight.data(), b.batch_weight.data(), 8, 3e-5f)) {
                    std::fprintf(stderr, "lane TWSC mismatch for m=%d count=%d item=%d\n", m, count, index);
                    return false;
                }
                if (std::memcmp(b.batch.data(), b.repeat.data(), b.batch.size() * sizeof(float)) != 0 ||
                    std::memcmp(b.batch_weight.data(), b.repeat_weight.data(), 8 * sizeof(float)) != 0) {
                    std::fprintf(stderr, "lane TWSC is not deterministic for m=%d count=%d item=%d\n", m, count,
                                 index);
                    return false;
                }
            }
        }
    }
    return true;
}

bool check_ncsr_lane_batches() {
    struct Buffers {
        int m = 0;
        int lda = 0;
        std::vector<float> scalar;
        std::vector<float> batch;
        std::vector<float> repeat;
        std::vector<float> scalar_work;
        std::vector<float> batch_work;
        std::vector<float> repeat_work;
        std::array<float, 8> distances{};
        int status = 0;
    };

    constexpr int counts[] = {2, 7, 8, 9, 16, 17, 32};
    for (int m : {16, 64}) {
        for (int count : counts) {
            std::vector<Buffers> groups(static_cast<std::size_t>(count));
            std::vector<nss::NcsrFilterBatchItem> items(static_cast<std::size_t>(count));
            std::vector<nss::NcsrFilterBatchItem> repeats(static_cast<std::size_t>(count));
            for (int index = 0; index < count; ++index) {
                auto& b = groups[static_cast<std::size_t>(index)];
                b.m = m;
                b.lda = m + index % 3;
                b.scalar.assign(static_cast<std::size_t>(b.lda * 8), 0.f);
                for (int col = 0; col < 8; ++col) {
                    b.distances[static_cast<std::size_t>(col)] = 0.03f * static_cast<float>((col + index) % 5);
                    for (int row = 0; row < m; ++row) {
                        b.scalar[static_cast<std::size_t>(row + col * b.lda)] =
                            0.22f * std::sin(static_cast<float>((index + 4) * (row + 1) * (col + 2)) * 0.023f) +
                            (row == col ? 0.12f : 0.f);
                    }
                }
                b.batch = b.scalar;
                b.repeat = b.scalar;
                const int work_floats = nss::ncsr_filter_work_floats(m, 8);
                b.scalar_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                b.batch_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                b.repeat_work.assign(static_cast<std::size_t>(work_floats), 0.f);
                const float* distances = index % 2 == 0 ? b.distances.data() : nullptr;
                if (nss::ncsr_filter_group(b.scalar.data(), m, 8, b.lda, 0.01f, distances, b.scalar_work.data(),
                                           work_floats) != 0) {
                    std::fprintf(stderr, "scalar NCSR fixture failed for m=%d count=%d item=%d\n", m, count, index);
                    return false;
                }
                items[static_cast<std::size_t>(index)] = nss::NcsrFilterBatchItem{
                    b.batch.data(), m, 8, b.lda, 0.01f, distances, b.batch_work.data(), work_floats, &b.status};
                repeats[static_cast<std::size_t>(index)] = nss::NcsrFilterBatchItem{
                    b.repeat.data(), m, 8, b.lda, 0.01f, distances, b.repeat_work.data(), work_floats, nullptr};
            }
            if (nss::ncsr_filter_group_batch(items.data(), count) != 0 ||
                nss::ncsr_filter_group_batch(repeats.data(), count) != 0) {
                std::fprintf(stderr, "lane NCSR batch failed for m=%d count=%d\n", m, count);
                return false;
            }
            for (int index = 0; index < count; ++index) {
                const auto& b = groups[static_cast<std::size_t>(index)];
                if (b.status != 1 || !compare_array(b.scalar.data(), b.batch.data(), b.scalar.size(), 4e-3f)) {
                    std::fprintf(stderr, "lane NCSR mismatch for m=%d count=%d item=%d\n", m, count, index);
                    return false;
                }
                if (std::memcmp(b.batch.data(), b.repeat.data(), b.batch.size() * sizeof(float)) != 0) {
                    std::fprintf(stderr, "lane NCSR is not deterministic for m=%d count=%d item=%d\n", m, count,
                                 index);
                    return false;
                }
            }
        }
    }
    return true;
}

bool check_ordered_queue() {
    nss::GroupKey a{64, 8, 1, nss::GroupAlgorithm::WNNM, false, false};
    nss::GroupKey b{64, 8, 1, nss::GroupAlgorithm::BM3D, true, false};
    std::vector<nss::GroupJob> jobs{{3, 3, 0, 0, a}, {1, 1, 0, 0, b}, {2, 2, 0, 0, a}};
    nss::bucket_group_jobs(jobs);
    if (jobs[0].key != b || jobs[1].key != a || jobs[2].ordinal != 3) {
        std::fprintf(stderr, "group bucketing is not key-stable\n");
        return false;
    }

    nss::OrderedCommitQueue<int> queue(0, 2);
    queue.push(2, 20);
    queue.push(0, 0);
    std::vector<int> committed;
    queue.drain([&](int value) { committed.push_back(value); });
    if (committed != std::vector<int>({0}) || queue.next_ordinal() != 1) {
        std::fprintf(stderr, "ordered drain violated raster order\n");
        return false;
    }
    queue.push(1, 10);
    queue.drain([&](int value) { committed.push_back(value); });
    if (committed != std::vector<int>({0, 10, 20}) || queue.next_ordinal() != 3) {
        std::fprintf(stderr, "ordered gap fill violated raster order\n");
        return false;
    }
    queue.push(3, 30);
    queue.push(4, 40);
    queue.finish([&](int value) { committed.push_back(value); });
    if (committed != std::vector<int>({0, 10, 20, 30, 40}) || queue.pending() != 0) {
        std::fprintf(stderr, "ordered finish violated raster order\n");
        return false;
    }

    nss::OrderedCommitQueue<int> bounded(10, 2);
    if (!bounded.push(12, 12) || bounded.push(12, 99) || bounded.push(9, 9) || bounded.push(13, 13)) {
        std::fprintf(stderr, "ordered queue did not reject duplicate/old/out-of-window ordinal\n");
        return false;
    }
    bounded.finish([&](int) {});
    if (bounded.complete() || bounded.next_ordinal() != 10 || bounded.pending() != 1) {
        std::fprintf(stderr, "ordered finish crossed a missing ordinal\n");
        return false;
    }
    if (!bounded.push(10, 10) || bounded.drain([&](int value) { committed.push_back(value); }) != 1 ||
        !bounded.push(11, 11) || bounded.drain([&](int) {}) != 2 || !bounded.complete()) {
        std::fprintf(stderr, "ordered queue did not recover after filling a gap\n");
        return false;
    }
    return true;
}

bool check_match_batches() {
    constexpr int width = 23;
    constexpr int height = 19;
    constexpr int stride = 32;
    constexpr int frames = 3;
    std::mt19937 rng(20260831u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<std::vector<float>> planes(static_cast<std::size_t>(frames),
                                           std::vector<float>(static_cast<std::size_t>(stride * height)));
    for (auto& plane : planes) {
        for (float& value : plane) {
            value = dist(rng);
        }
    }
    const float* refs[frames] = {planes[0].data(), planes[1].data(), planes[2].data()};
    int strides[frames] = {stride, stride, stride};
    const nss::MatchBatchItem items[]{{0, 0, 1, 5, 1}, {4, 3, 4, 4, 4}, {7, 7, 8, 7, 8},
                                      {19, 15, 2, 3, 2}, {8, 7, 8, 6, 8}};
    constexpr int count = static_cast<int>(std::size(items));
    std::array<nss::Match, count * nss::kBmMaxGroup> batch{};
    std::array<int, count> counts{};
    if (nss::spatial_match_batch(refs[0], stride, width, height, items, count, batch.data(), nss::kBmMaxGroup,
                                 counts.data()) != 0) {
        std::fprintf(stderr, "spatial match batch returned an error\n");
        return false;
    }
    for (int i = 0; i < count; ++i) {
        nss::Match scalar[nss::kBmMaxGroup]{};
        const int n = nss::spatial_match(refs[0], stride, width, height, items[i].bx, items[i].by, items[i].block,
                                         items[i].bm_range, items[i].group, scalar);
        if (counts[i] != n || !compare_matches(batch.data() + static_cast<std::size_t>(i) * nss::kBmMaxGroup, scalar, n)) {
            std::fprintf(stderr, "spatial item %d mismatch batch_n=%d scalar_n=%d\n", i, counts[i], n);
            return false;
        }
    }

    nss::SearchConfig cfg;
    cfg.block = 4;
    cfg.step = 2;
    cfg.group = 8;
    cfg.bm_range = 4;
    cfg.radius = 1;
    cfg.ps_num = 2;
    cfg.ps_range = 3;
    const nss::MatchBatchItem temporal_items[]{{2, 2, 4, 4, 8}, {11, 7, 4, 3, 4}, {16, 11, 2, 3, 2}};
    constexpr int temporal_count = static_cast<int>(std::size(temporal_items));
    std::array<nss::Match, temporal_count * nss::kBmMaxGroup> temporal_batch{};
    std::array<int, temporal_count> temporal_counts{};
    if (nss::predictive_match_batch(refs, strides, frames, width, height, 1, cfg, temporal_items, temporal_count,
                                    temporal_batch.data(), nss::kBmMaxGroup, temporal_counts.data()) != 0) {
        std::fprintf(stderr, "predictive match batch returned an error\n");
        return false;
    }
    for (int i = 0; i < temporal_count; ++i) {
        nss::Match scalar[nss::kBmMaxGroup]{};
        nss::SearchConfig local = cfg;
        local.block = temporal_items[i].block;
        local.group = temporal_items[i].group;
        local.bm_range = temporal_items[i].bm_range;
        const int n = nss::predictive_match(refs, strides, frames, width, height, temporal_items[i].bx,
                                             temporal_items[i].by, 1, local, scalar);
        if (temporal_counts[i] != n ||
            !compare_matches(temporal_batch.data() + static_cast<std::size_t>(i) * nss::kBmMaxGroup, scalar, n)) {
            std::fprintf(stderr, "temporal item %d mismatch batch_n=%d scalar_n=%d\n", i, temporal_counts[i], n);
            return false;
        }
    }
    return true;
}

template <typename Fill>
std::vector<float> make_values(std::size_t n, Fill&& fill) {
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<float>(fill(i));
    }
    return values;
}

bool check_filter_batches() {
    std::mt19937 rng(911u);
    std::uniform_real_distribution<float> random(-0.6f, 0.6f);
    auto fill_random = [&](std::vector<float>& values) {
        for (float& value : values) {
            value = random(rng);
        }
    };

    {
        const int block = 4, area = 16, group = 4, k = 3, lda = 20;
        const int work_n = nss::bm3d_filter_work_floats(group, block);
        std::vector<float> scalar(static_cast<std::size_t>(lda * group));
        std::vector<float> batch = scalar;
        std::vector<float> ref(static_cast<std::size_t>(lda * group));
        fill_random(scalar);
        fill_random(ref);
        batch = scalar;
        std::vector<float> sw(static_cast<std::size_t>(work_n));
        std::vector<float> bw(static_cast<std::size_t>(work_n));
        float sweight = 1.f, bweight = 1.f;
        nss::bm3d_filter_group(scalar.data(), lda, group, k, block, 0.025f, true, ref.data(), &sweight, sw.data());
        nss::Bm3dFilterBatchItem item{batch.data(), lda, group, k, block, 0.025f, true, ref.data(), &bweight,
                                      bw.data()};
        if (nss::bm3d_filter_group_batch(&item, 1) != 0 || !compare_array(scalar.data(), batch.data(), scalar.size()) ||
            !close_float(sweight, bweight)) {
            std::fprintf(stderr, "BM3D scalar/batch mismatch\n");
            return false;
        }
        (void)area;
    }

    {
        const int m = 16, n = 4, lda = 20;
        const int work_n = nss::wnnm_shrink_work_floats(m, n);
        std::vector<float> scalar(static_cast<std::size_t>(lda * n));
        fill_random(scalar);
        std::vector<float> batch = scalar;
        std::vector<float> sw(static_cast<std::size_t>(work_n));
        std::vector<float> bw(static_cast<std::size_t>(work_n));
        float sweight = 1.f, bweight = 1.f;
        nss::wnnm_shrink(scalar.data(), m, n, lda, 0.03f, 1, 1, &sweight, sw.data(), work_n);
        nss::WnnmShrinkBatchItem item{batch.data(), m, n, lda, 0.03f, 1, 1, &bweight, bw.data(), work_n};
        if (nss::wnnm_shrink_batch(&item, 1) != 0 || !compare_array(scalar.data(), batch.data(), scalar.size()) ||
            !close_float(sweight, bweight)) {
            std::fprintf(stderr, "WNNM scalar/batch mismatch\n");
            return false;
        }
    }

    {
        const int m = 48, n = 4, lda = 64, nch = 3;
        const int work_n = nss::mcwnnm_filter_work_floats(m, n);
        const float sigma[3] = {0.02f, 0.025f, 0.03f};
        std::vector<float> scalar(static_cast<std::size_t>(lda * n));
        fill_random(scalar);
        std::vector<float> batch = scalar;
        std::vector<float> sw(static_cast<std::size_t>(work_n));
        std::vector<float> bw(static_cast<std::size_t>(work_n));
        float sweight = 1.f, bweight = 1.f;
        nss::mcwnnm_filter_group(scalar.data(), m, n, lda, nch, sigma, 3, 1.1f, 1.01f, 1, 1, &sweight, sw.data(),
                                 work_n);
        nss::McwnnmFilterBatchItem item{batch.data(), m, n, lda, nch, sigma, 3, 1.1f, 1.01f, 1, 1, &bweight,
                                        bw.data(), work_n};
        if (nss::mcwnnm_filter_group_batch(&item, 1) != 0 || !compare_array(scalar.data(), batch.data(), scalar.size(), 2e-4f) ||
            !close_float(sweight, bweight, 2e-4f)) {
            std::fprintf(stderr, "MCWNNM scalar/batch mismatch\n");
            return false;
        }
    }

    {
        const int m = 16, n = 4, lda = 20;
        const int work_n = nss::twsc_pca_soft_work_floats(m, n);
        const float col_sigma[4] = {0.02f, 0.03f, 0.025f, 0.04f};
        const float row_w[16] = {1.f, 1.f, 0.9f, 0.8f, 1.1f, 1.f, 0.7f, 1.f,
                                 1.f, 0.9f, 1.f, 1.1f, 0.8f, 1.f, 1.f, 0.95f};
        std::vector<float> scalar(static_cast<std::size_t>(lda * n));
        fill_random(scalar);
        std::vector<float> batch = scalar;
        std::vector<float> sw(static_cast<std::size_t>(work_n));
        std::vector<float> bw(static_cast<std::size_t>(work_n));
        std::array<float, 4> swgt{}, bwgt{};
        nss::twsc_pca_soft(scalar.data(), m, n, lda, 0.02f, sw.data(), work_n, col_sigma, swgt.data(), row_w);
        nss::TwscPcaBatchItem item{batch.data(), m, n, lda, 0.02f, bw.data(), work_n, col_sigma, bwgt.data(), row_w};
        if (nss::twsc_pca_soft_batch(&item, 1) != 0 || !compare_array(scalar.data(), batch.data(), scalar.size(), 2e-4f) ||
            !compare_array(swgt.data(), bwgt.data(), swgt.size(), 2e-4f)) {
            std::fprintf(stderr, "TWSC scalar/batch mismatch\n");
            return false;
        }
    }

    {
        const int m = 16, n = 4, lda = 20;
        const int work_n = nss::ncsr_filter_work_floats(m, n);
        const float distances[4] = {0.f, 0.2f, 0.4f, 0.1f};
        std::vector<float> scalar(static_cast<std::size_t>(lda * n));
        fill_random(scalar);
        std::vector<float> batch = scalar;
        std::vector<float> sw(static_cast<std::size_t>(work_n));
        std::vector<float> bw(static_cast<std::size_t>(work_n));
        nss::ncsr_filter_group(scalar.data(), m, n, lda, 0.025f, distances, sw.data(), work_n);
        nss::NcsrFilterBatchItem item{batch.data(), m, n, lda, 0.025f, distances, bw.data(), work_n};
        if (nss::ncsr_filter_group_batch(&item, 1) != 0 || !compare_array(scalar.data(), batch.data(), scalar.size(), 2e-4f)) {
            std::fprintf(stderr, "NCSR scalar/batch mismatch\n");
            return false;
        }
    }

    {
        const int m = 16, n = 4, lda = 20, q = 4;
        const int work_n = nss::nlh_filter_work_floats(m, n, q, lda);
        std::vector<float> scalar(static_cast<std::size_t>(lda * n));
        std::vector<float> ref(static_cast<std::size_t>(lda * n));
        fill_random(scalar);
        fill_random(ref);
        std::vector<float> batch = scalar;
        std::vector<float> sw(static_cast<std::size_t>(work_n));
        std::vector<float> bw(static_cast<std::size_t>(work_n));
        float sweight = 1.f, bweight = 1.f;
        nss::nlh_filter_group(scalar.data(), m, n, lda, q, 0.025f, true, ref.data(), &sweight, sw.data(), work_n);
        nss::NlhFilterBatchItem item{batch.data(), m, n, lda, q, 0.025f, true, ref.data(), &bweight, bw.data(), work_n};
        if (nss::nlh_filter_group_batch(&item, 1) != 0 || !compare_array(scalar.data(), batch.data(), scalar.size(), 2e-4f) ||
            !close_float(sweight, bweight, 2e-4f)) {
            std::fprintf(stderr, "NLH scalar/batch mismatch\n");
            return false;
        }
    }
    return true;
}

bool check_batch_failure_continues() {
    // A malformed item must be reported without preventing a later bucket from
    // completing. This is the contract host filters use for short/invalid
    // groups discovered in one matcher window.
    constexpr int m = 4;
    constexpr int n = 2;
    constexpr int lda = 4;
    std::vector<float> good(static_cast<std::size_t>(lda * n));
    for (std::size_t i = 0; i < good.size(); ++i) {
        good[i] = 0.1f + static_cast<float>(i) * 0.03f;
    }
    const std::vector<float> before = good;
    std::vector<float> work(static_cast<std::size_t>(nss::wnnm_shrink_work_floats(m, n)));
    int bad_status = -1;
    int good_status = -1;
    nss::WnnmShrinkBatchItem items[2]{};
    items[0] = nss::WnnmShrinkBatchItem{nullptr, m, n, lda, 0.03f, 0, 0, nullptr, work.data(),
                                       static_cast<int>(work.size()), &bad_status};
    items[1] = nss::WnnmShrinkBatchItem{good.data(), m, n, lda, 0.03f, 0, 0, nullptr, work.data(),
                                       static_cast<int>(work.size()), &good_status};
    const int rc = nss::wnnm_shrink_batch(items, 2);
    if (rc != 1 || bad_status != 0 || good_status != 1) {
        std::fprintf(stderr, "batch failure did not preserve per-item progress (rc=%d bad=%d good=%d)\n", rc,
                     bad_status, good_status);
        return false;
    }
    for (float value : good) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "successful batch item produced a non-finite value\n");
            return false;
        }
    }
    if (good == before) {
        std::fprintf(stderr, "successful batch item was not processed after a failure\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!check_ordered_queue()) {
        return 1;
    }
    if (!check_match_batches()) {
        return 1;
    }
    if (!check_svd_lane_batches()) {
        return 1;
    }
    if (!check_wnnm_lane_batches()) {
        return 1;
    }
    if (!check_twsc_lane_batches()) {
        return 1;
    }
    if (!check_ncsr_lane_batches()) {
        return 1;
    }
    if (!check_filter_batches()) {
        return 1;
    }
    if (!check_batch_failure_continues()) {
        return 1;
    }

    constexpr int width = 12;
    constexpr int height = 12;
    constexpr int stride = 16;
    std::vector<float> plane(static_cast<std::size_t>(stride * height), 0.f);
    nss::MatchBatchItem items[2]{{0, 0, 2, 2, 4}, {8, 8, 4, 2, 4}};
    nss::Match matches[2 * 4]{};
    int counts[2]{};
    if (nss::spatial_match_batch(plane.data(), stride, width, height, items, 2, matches, 4, counts) != 0 ||
        counts[0] != 4 || counts[1] != 4) {
        std::fprintf(stderr, "batched matcher failed\n");
        return 1;
    }

    float a0[8] = {1.f, 0.f, 0.f, 1.f, 2.f, 0.f, 0.f, 2.f};
    float a1[8] = {2.f, 0.f, 0.f, 2.f, 1.f, 0.f, 0.f, 1.f};
    float u0[8]{}, u1[8]{}, s0[2]{}, s1[2]{}, vt0[4]{}, vt1[4]{};
    nss::SvdBatchItem svd_items[2]{{4, 2, a0, 4, u0, 4, s0, vt0, 2, nullptr, 0},
                                   {4, 2, a1, 4, u1, 4, s1, vt1, 2, nullptr, 0}};
    if (nss::svd_economy_batch(svd_items, 2) != 0 || !(s0[0] >= s0[1]) || !(s1[0] >= s1[1])) {
        std::fprintf(stderr, "batched SVD failed\n");
        return 1;
    }
    float g0[4]{}, g1[4]{};
    nss::GemmNNBatchItem gemm_items[2]{{2, 2, 2, a0, 2, a0, 2, g0, 2},
                                       {2, 2, 2, a1, 2, a1, 2, g1, 2}};
    nss::gemm_nn_batch(gemm_items, 2);
    if (!std::isfinite(g0[0]) || !std::isfinite(g1[0])) {
        std::fprintf(stderr, "batched GEMM failed\n");
        return 1;
    }
    std::printf("batch scheduler ordering checks complete\n");
    return 0;
}
