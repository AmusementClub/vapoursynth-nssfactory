// SPDX-License-Identifier: GPL-2.0-only
// The fixed-point scale of deterministic aggregation (D17), shared by the GPU
// backends (D27): sums are int64 cells of value * 2^32, so integer atomics
// give the same sums in any order.
#pragma once

namespace nss::gpuhost {

// 2^32: a float converts without rounding unless it carries bits below
// 2^-32, and sums up to 2^31 in magnitude fit.
inline constexpr float kFixedScale = 4294967296.f;

}  // namespace nss::gpuhost
