#pragma once

#include "nss/cpu_api.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace nss::detail {

inline bool finite_distance(float value) noexcept {
    // Keep classification valid in the -ffast-math TUs as well.  Calling
    // std::isfinite there is permitted to fold to true and would violate the
    // matcher contract for NaN/Inf distances.
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

// Match ordering is part of the denoiser contract: equal distances must not
// depend on the implementation's container or SIMD lane order.
inline bool match_less(const Match& a, const Match& b) noexcept {
    const bool a_finite = finite_distance(a.dist);
    const bool b_finite = finite_distance(b.dist);
    if (a_finite != b_finite) {
        return a_finite;
    }
    if (a_finite && a.dist != b.dist) {
        return a.dist < b.dist;
    }
    // The reference patch is part of every group and remains first even when
    // another patch has the same distance (the historical BM contract).
    const bool a_self = a.ordinal == 0;
    const bool b_self = b.ordinal == 0;
    if (a_self != b_self) {
        return a_self;
    }
    if (a.t != b.t) {
        return a.t < b.t;
    }
    if (a.y != b.y) {
        return a.y < b.y;
    }
    if (a.x != b.x) {
        return a.x < b.x;
    }
    return a.ordinal < b.ordinal;
}

class StableTopK {
public:
    StableTopK(Match* storage, int capacity) : storage_(storage), capacity_(capacity) {}

    void add(Match candidate) noexcept {
        if (!storage_ || capacity_ <= 0) {
            return;
        }
        if (size_ < capacity_) {
            storage_[size_++] = candidate;
            return;
        }
        int worst = 0;
        for (int i = 1; i < size_; ++i) {
            if (match_less(storage_[worst], storage_[i])) {
                worst = i;
            }
        }
        if (match_less(candidate, storage_[worst])) {
            storage_[worst] = candidate;
        }
    }

    int finish() noexcept {
        if (size_ > 1) {
            std::sort(storage_, storage_ + size_, match_less);
        }
        return size_;
    }

    int size() const noexcept { return size_; }

private:
    Match* storage_ = nullptr;
    int capacity_ = 0;
    int size_ = 0;
};

template <typename DistanceFn>
int collect_spatial_coords(int width, int height, int bx, int by, int block, int bm_range, int group, Match* out,
                           DistanceFn&& distance) {
    if (!out || width < 1 || height < 1 || block < 1 || width < block || height < block || group < 1) {
        return 0;
    }
    const int wanted = std::min(group, kBmMaxGroup);
    const int max_x = width - block;
    const int max_y = height - block;
    const int cx = std::clamp(bx, 0, max_x);
    const int cy = std::clamp(by, 0, max_y);
    const int range = std::max(bm_range, 0);
    const int top = std::max(cy - range, 0);
    const int bottom = std::min(cy + range, max_y);
    const int left = std::max(cx - range, 0);
    const int right = std::min(cx + range, max_x);

    out[0] = Match{cx, cy, 0, 0.f, 0};
    if (wanted == 1) {
        return 1;
    }
    StableTopK topk(out + 1, wanted - 1);
    std::uint32_t ordinal = 1;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x, ++ordinal) {
            if (x == cx && y == cy) {
                continue;
            }
            const float dist = distance(cx, cy, x, y);
            topk.add(Match{x, y, 0, dist, ordinal});
        }
    }
    return 1 + topk.finish();
}

template <typename DistanceFn>
int collect_spatial(const float* ref, int stride, int width, int height, int bx, int by, int block, int bm_range,
                    int group, Match* out, DistanceFn&& distance) {
    if (!ref || stride < width || width < 1 || height < 1) {
        return 0;
    }
    return collect_spatial_coords(width, height, bx, by, block, bm_range, group, out,
                                  [&](int cx, int cy, int x, int y) {
                                      return distance(ref + cy * stride + cx, ref + y * stride + x, stride, block);
                                  });
}

inline void assign_temporal_order(Match* matches, int count, int t, int reference_t,
                                  std::uint32_t& next_ordinal) noexcept {
    // Preserve each spatial matcher ordinal while adding a frame component.
    // Time and coordinates remain earlier keys; the ordinal only resolves an
    // otherwise identical candidate deterministically.
    constexpr std::uint32_t kFrameStride = 1u << 20;
    for (int i = 0; i < count; ++i) {
        const std::uint32_t local = matches[i].ordinal;
        matches[i].t = t;
        if (t == reference_t && local == 0) {
            matches[i].ordinal = 0;
        } else {
            const std::uint32_t frame = static_cast<std::uint32_t>(std::max(t, 0) + 1);
            matches[i].ordinal = frame * kFrameStride + local;
        }
        next_ordinal = std::max(next_ordinal, matches[i].ordinal + 1u);
    }
}

}  // namespace nss::detail
