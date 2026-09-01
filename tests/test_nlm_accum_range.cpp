#include "nss/cpu_api.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr float kGuard = -12345.25f;

struct Case {
    int width;
    int height;
    int stride;
    int ox;
    int oy;
    int y0;
    int y1;
};

int clampi(int value, int lo, int hi) {
    return std::max(lo, std::min(value, hi));
}

using Planes = std::array<std::vector<float>, 3>;
using ConstPtrs = std::array<const float*, 3>;
using MutPtrs = std::array<float*, 3>;

void call_full(int nch, float* weight, const MutPtrs& wdst, float* maxw, const ConstPtrs& sb, const ConstPtrs& sf,
               const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride) {
    if (nch == 1) {
        nss::nlm_accum_ch1(weight, wdst[0], maxw, sb[0], sf[0], temp1, temp2, ox, oy, w, h, stride);
    } else if (nch == 2) {
        nss::nlm_accum_ch2(weight, wdst[0], wdst[1], maxw, sb[0], sb[1], sf[0], sf[1], temp1, temp2, ox, oy, w,
                           h, stride);
    } else {
        nss::nlm_accum_ch3(weight, wdst[0], wdst[1], wdst[2], maxw, sb[0], sb[1], sb[2], sf[0], sf[1], sf[2], temp1,
                           temp2, ox, oy, w, h, stride);
    }
}

void call_range(int nch, float* weight, const MutPtrs& wdst, float* maxw, const ConstPtrs& sb, const ConstPtrs& sf,
                const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride, int y0, int y1) {
    if (nch == 1) {
        nss::nlm_accum_ch1_range(weight, wdst[0], maxw, sb[0], sf[0], temp1, temp2, ox, oy, w, h, stride, y0, y1);
    } else if (nch == 2) {
        nss::nlm_accum_ch2_range(weight, wdst[0], wdst[1], maxw, sb[0], sb[1], sf[0], sf[1], temp1, temp2, ox, oy,
                                 w, h, stride, y0, y1);
    } else {
        nss::nlm_accum_ch3_range(weight, wdst[0], wdst[1], wdst[2], maxw, sb[0], sb[1], sb[2], sf[0], sf[1], sf[2],
                                 temp1, temp2, ox, oy, w, h, stride, y0, y1);
    }
}

void call_core_range(int nch, float* weight, const MutPtrs& wdst, float* maxw, const ConstPtrs& sb, const ConstPtrs& sf,
                     const float* temp1_core, const float* temp2, int ox, int oy, int w, int h, int stride, int y0,
                     int y1, int temp2_base_y = 0) {
    if (nch == 1) {
        nss::nlm_accum_ch1_core_range(weight, wdst[0], maxw, sb[0], sf[0], temp1_core, temp2, ox, oy, w, h, stride,
                                      y0, y1, temp2_base_y);
    } else if (nch == 2) {
        nss::nlm_accum_ch2_core_range(weight, wdst[0], wdst[1], maxw, sb[0], sb[1], sf[0], sf[1], temp1_core, temp2,
                                      ox, oy, w, h, stride, y0, y1, temp2_base_y);
    } else {
        nss::nlm_accum_ch3_core_range(weight, wdst[0], wdst[1], wdst[2], maxw, sb[0], sb[1], sb[2], sf[0], sf[1],
                                      sf[2], temp1_core, temp2, ox, oy, w, h, stride, y0, y1, temp2_base_y);
    }
}

void scalar_accum(int nch, float* weight, const MutPtrs& wdst, float* maxw, const ConstPtrs& sb, const ConstPtrs& sf,
                  const float* temp1, const float* temp2, int ox, int oy, int w, int h, int stride, int y0, int y1,
                  int temp1_base_y) {
    for (int y = y0; y < y1; ++y) {
        const int out_y = y - y0;
        const int temp_y = y - temp1_base_y;
        for (int x = 0; x < w; ++x) {
            const int out = out_y * stride + x;
            const int temp = temp_y * stride + x;
            const int mq = clampi(y - oy, 0, h - 1) * stride + clampi(x - ox, 0, w - 1);
            const int pq = clampi(y + oy, 0, h - 1) * stride + clampi(x + ox, 0, w - 1);
            const float u = temp1[temp];
            const float um = temp2[mq];
            weight[out] += u + um;
            maxw[out] = std::max(maxw[out], std::max(u, um));
            for (int c = 0; c < nch; ++c) {
                wdst[c][out] += u * sb[c][pq] + um * sf[c][mq];
            }
        }
    }
}

struct Compact {
    std::vector<float> storage;
    int rows = 0;
    int stride = 0;

    Compact(int row_count, int row_stride) : storage(static_cast<std::size_t>(row_count + 2) * row_stride, kGuard),
                                              rows(row_count), stride(row_stride) {}

    float* data() { return storage.data() + stride; }
    const float* data() const { return storage.data() + stride; }

    void copy_rows(const std::vector<float>& full, int first_row) {
        for (int y = 0; y < rows; ++y) {
            std::copy_n(full.data() + static_cast<std::size_t>(first_row + y) * stride, stride,
                        data() + static_cast<std::size_t>(y) * stride);
        }
    }

    bool guards_intact() const {
        return std::all_of(storage.begin(), storage.begin() + stride, [](float v) { return v == kGuard; }) &&
               std::all_of(storage.end() - stride, storage.end(), [](float v) { return v == kGuard; });
    }
};

bool close_enough(float got, float want) {
    const float scale = 1.0f + std::max(std::fabs(got), std::fabs(want));
    return std::fabs(got - want) <= 4e-5f * scale;
}

bool compare_compact(const Compact& weight, const Compact& maxw, const std::array<const Compact*, 3>& wdst,
                     const std::vector<float>& want_w, const std::vector<float>& want_max, const Planes& want_dst,
                     int y0, int nch, int stride) {
    if (!weight.guards_intact() || !maxw.guards_intact()) {
        return false;
    }
    for (int c = 0; c < nch; ++c) {
        if (!wdst[c]->guards_intact()) {
            return false;
        }
    }
    for (int y = 0; y < weight.rows; ++y) {
        for (int x = 0; x < stride; ++x) {
            const std::size_t compact_idx = static_cast<std::size_t>(y) * stride + x;
            const std::size_t full_idx = static_cast<std::size_t>(y0 + y) * stride + x;
            if (!close_enough(weight.data()[compact_idx], want_w[full_idx]) ||
                !close_enough(maxw.data()[compact_idx], want_max[full_idx])) {
                std::fprintf(stderr, "state mismatch y=%d x=%d got(w=%g,m=%g) want(w=%g,m=%g)\n", y, x,
                             weight.data()[compact_idx], maxw.data()[compact_idx], want_w[full_idx], want_max[full_idx]);
                return false;
            }
            for (int c = 0; c < nch; ++c) {
                if (!close_enough(wdst[c]->data()[compact_idx], want_dst[c][full_idx])) {
                    std::fprintf(stderr, "dst mismatch y=%d x=%d c=%d got=%g want=%g\n", y, x, c,
                                 wdst[c]->data()[compact_idx], want_dst[c][full_idx]);
                    return false;
                }
            }
        }
    }
    return true;
}

bool run_case(int nch, const Case& tc, std::mt19937& rng) {
    const std::size_t size = static_cast<std::size_t>(tc.height) * tc.stride;
    const int core_rows = tc.y1 - tc.y0;
    std::uniform_real_distribution<float> value(-0.75f, 0.75f);

    Planes sb{};
    Planes sf{};
    Planes base_dst{};
    std::vector<float> temp1(size);
    std::vector<float> temp2(size);
    std::vector<float> base_weight(size);
    std::vector<float> base_max(size);
    for (int c = 0; c < 3; ++c) {
        sb[c].resize(size);
        sf[c].resize(size);
        base_dst[c].resize(size);
        for (std::size_t i = 0; i < size; ++i) {
            sb[c][i] = value(rng);
            sf[c][i] = value(rng);
            base_dst[c][i] = value(rng);
        }
    }
    for (std::size_t i = 0; i < size; ++i) {
        temp1[i] = value(rng);
        temp2[i] = value(rng);
        base_weight[i] = value(rng);
        base_max[i] = value(rng);
    }

    ConstPtrs sbp{sb[0].data(), sb[1].data(), sb[2].data()};
    ConstPtrs sfp{sf[0].data(), sf[1].data(), sf[2].data()};

    std::vector<float> expected_weight = base_weight;
    std::vector<float> expected_max = base_max;
    Planes expected_dst = base_dst;
    MutPtrs expected_range_dstp{expected_dst[0].data() + static_cast<std::size_t>(tc.y0) * tc.stride,
                               expected_dst[1].data() + static_cast<std::size_t>(tc.y0) * tc.stride,
                               expected_dst[2].data() + static_cast<std::size_t>(tc.y0) * tc.stride};
    scalar_accum(nch, expected_weight.data() + static_cast<std::size_t>(tc.y0) * tc.stride, expected_range_dstp,
                 expected_max.data() + static_cast<std::size_t>(tc.y0) * tc.stride, sbp, sfp, temp1.data(), temp2.data(),
                 tc.ox, tc.oy, tc.width, tc.height, tc.stride, tc.y0, tc.y1, 0);

    std::vector<float> full_expected_weight = base_weight;
    std::vector<float> full_expected_max = base_max;
    Planes full_expected_dst = base_dst;
    MutPtrs full_expected_dstp{full_expected_dst[0].data(), full_expected_dst[1].data(), full_expected_dst[2].data()};
    scalar_accum(nch, full_expected_weight.data(), full_expected_dstp, full_expected_max.data(), sbp, sfp, temp1.data(),
                 temp2.data(), tc.ox, tc.oy, tc.width, tc.height, tc.stride, 0, tc.height, 0);

    std::vector<float> full_weight = base_weight;
    std::vector<float> full_max = base_max;
    Planes full_dst = base_dst;
    MutPtrs full_dstp{full_dst[0].data(), full_dst[1].data(), full_dst[2].data()};
    call_full(nch, full_weight.data(), full_dstp, full_max.data(), sbp, sfp, temp1.data(), temp2.data(), tc.ox, tc.oy,
              tc.width, tc.height, tc.stride);
    for (int y = 0; y < tc.height; ++y) {
        for (int x = 0; x < tc.stride; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * tc.stride + x;
            if (!close_enough(full_weight[i], full_expected_weight[i]) || !close_enough(full_max[i], full_expected_max[i])) {
                std::fprintf(stderr, "full mismatch nch=%d w=%d h=%d ox=%d oy=%d y=[%d,%d) at %d,%d\n", nch,
                             tc.width, tc.height, tc.ox, tc.oy, tc.y0, tc.y1, y, x);
                return false;
            }
            for (int c = 0; c < nch; ++c) {
                if (!close_enough(full_dst[c][i], full_expected_dst[c][i])) {
                    std::fprintf(stderr, "full channel mismatch nch=%d c=%d at %d,%d\n", nch, c, y, x);
                    return false;
                }
            }
        }
    }

    Compact range_weight(core_rows, tc.stride);
    Compact range_max(core_rows, tc.stride);
    std::array<Compact, 3> range_dst{Compact(core_rows, tc.stride), Compact(core_rows, tc.stride),
                                     Compact(core_rows, tc.stride)};
    range_weight.copy_rows(base_weight, tc.y0);
    range_max.copy_rows(base_max, tc.y0);
    for (int c = 0; c < 3; ++c) {
        range_dst[c].copy_rows(base_dst[c], tc.y0);
    }
    MutPtrs rangep{range_dst[0].data(), range_dst[1].data(), range_dst[2].data()};
    call_range(nch, range_weight.data(), rangep, range_max.data(), sbp, sfp, temp1.data(), temp2.data(), tc.ox, tc.oy,
               tc.width, tc.height, tc.stride, tc.y0, tc.y1);
    std::array<const Compact*, 3> range_dst_const{&range_dst[0], &range_dst[1], &range_dst[2]};
    if (!compare_compact(range_weight, range_max, range_dst_const, expected_weight, expected_max, expected_dst, tc.y0,
                         nch, tc.stride)) {
        std::fprintf(stderr, "range mismatch nch=%d w=%d h=%d ox=%d oy=%d y=[%d,%d)\n", nch, tc.width, tc.height,
                     tc.ox, tc.oy, tc.y0, tc.y1);
        return false;
    }

    Compact core_weight(core_rows, tc.stride);
    Compact core_max(core_rows, tc.stride);
    std::array<Compact, 3> core_dst{Compact(core_rows, tc.stride), Compact(core_rows, tc.stride),
                                    Compact(core_rows, tc.stride)};
    core_weight.copy_rows(base_weight, tc.y0);
    core_max.copy_rows(base_max, tc.y0);
    for (int c = 0; c < 3; ++c) {
        core_dst[c].copy_rows(base_dst[c], tc.y0);
    }
    std::vector<float> temp1_core(static_cast<std::size_t>(core_rows) * tc.stride);
    for (int y = 0; y < core_rows; ++y) {
        std::copy_n(temp1.data() + static_cast<std::size_t>(tc.y0 + y) * tc.stride, tc.stride,
                    temp1_core.data() + static_cast<std::size_t>(y) * tc.stride);
    }
    MutPtrs corep{core_dst[0].data(), core_dst[1].data(), core_dst[2].data()};
    call_core_range(nch, core_weight.data(), corep, core_max.data(), sbp, sfp, temp1_core.data(), temp2.data(), tc.ox,
                    tc.oy, tc.width, tc.height, tc.stride, tc.y0, tc.y1);
    std::array<const Compact*, 3> core_dst_const{&core_dst[0], &core_dst[1], &core_dst[2]};
    if (!compare_compact(core_weight, core_max, core_dst_const, expected_weight, expected_max, expected_dst, tc.y0, nch,
                         tc.stride)) {
        std::fprintf(stderr, "core range mismatch nch=%d w=%d h=%d ox=%d oy=%d y=[%d,%d)\n", nch, tc.width,
                     tc.height, tc.ox, tc.oy, tc.y0, tc.y1);
        return false;
    }

    const int temp2_base_y = clampi(tc.y0 - tc.oy, 0, tc.height - 1);
    const int temp2_end_y = clampi(tc.y1 - 1 - tc.oy, 0, tc.height - 1) + 1;
    std::vector<float> temp2_core(static_cast<std::size_t>(temp2_end_y - temp2_base_y) * tc.stride);
    for (int y = temp2_base_y; y < temp2_end_y; ++y) {
        std::copy_n(temp2.data() + static_cast<std::size_t>(y) * tc.stride, tc.stride,
                    temp2_core.data() + static_cast<std::size_t>(y - temp2_base_y) * tc.stride);
    }
    Compact compact2_weight(core_rows, tc.stride);
    Compact compact2_max(core_rows, tc.stride);
    std::array<Compact, 3> compact2_dst{Compact(core_rows, tc.stride), Compact(core_rows, tc.stride),
                                        Compact(core_rows, tc.stride)};
    compact2_weight.copy_rows(base_weight, tc.y0);
    compact2_max.copy_rows(base_max, tc.y0);
    for (int c = 0; c < 3; ++c) {
        compact2_dst[c].copy_rows(base_dst[c], tc.y0);
    }
    MutPtrs compact2p{compact2_dst[0].data(), compact2_dst[1].data(), compact2_dst[2].data()};
    call_core_range(nch, compact2_weight.data(), compact2p, compact2_max.data(), sbp, sfp, temp1_core.data(),
                    temp2_core.data(), tc.ox, tc.oy, tc.width, tc.height, tc.stride, tc.y0, tc.y1, temp2_base_y);
    std::array<const Compact*, 3> compact2_dst_const{&compact2_dst[0], &compact2_dst[1], &compact2_dst[2]};
    if (!compare_compact(compact2_weight, compact2_max, compact2_dst_const, expected_weight, expected_max,
                         expected_dst, tc.y0, nch, tc.stride)) {
        std::fprintf(stderr, "compact temp2 mismatch nch=%d w=%d h=%d ox=%d oy=%d y=[%d,%d) temp2=[%d,%d)\n", nch,
                     tc.width, tc.height, tc.ox, tc.oy, tc.y0, tc.y1, temp2_base_y, temp2_end_y);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const std::array<Case, 9> cases{{
        {1, 1, 4, 0, 0, 0, 1},
        {2, 2, 7, -1, 1, 0, 1},
        {3, 2, 8, 2, -3, 1, 2},
        {5, 5, 9, -4, 4, 1, 4},
        {7, 11, 13, 3, -7, 3, 8},
        {11, 4, 17, -1, 2, 2, 3},
        {19, 9, 24, 0, -5, 0, 9},
        {19, 9, 24, 7, 1, 4, 5},
        {31, 3, 37, -9, 6, 1, 3},
    }};
    std::mt19937 rng(0x4e4c4d52u);
    for (int nch = 1; nch <= 3; ++nch) {
        for (const Case& tc : cases) {
            if (!run_case(nch, tc, rng)) {
                return 1;
            }
        }
    }
    std::puts("nlm accumulation range differential tests passed");
    return 0;
}
