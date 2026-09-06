#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace nss::host_detail {

// Subtract the available support before adding an offset, avoiding overflow
// for legal clips whose final frame is near INT_MAX.
inline int temporal_first(int center, int radius) {
    return center - std::min(center, radius);
}

inline int temporal_last(int center, int radius, int frames) {
    return center + std::min(radius, frames - 1 - center);
}

inline int temporal_slot_frame(int center, int slot, int radius, int frames) {
    return center + std::clamp(slot - radius, -center, frames - 1 - center);
}

inline void temporal_identity(float* dst, int ds, const float* src, int ss, int w, int h, int radius) {
    for (int t = 0; t < 2 * radius + 1; ++t) {
        for (int y = 0; y < h; ++y) {
            float* num = dst + (static_cast<std::size_t>(2 * t) * h + y) * ds;
            float* den = num + static_cast<std::size_t>(h) * ds;
            if (t == radius) {
                std::memcpy(num, src + y * ss, w * sizeof(float));
            } else {
                std::fill_n(num, w, 0.f);
            }
            std::fill_n(den, w, t == radius ? 1.f : 0.f);
        }
    }
}

} // namespace nss::host_detail
