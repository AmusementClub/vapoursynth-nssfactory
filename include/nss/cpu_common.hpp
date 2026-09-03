#pragma once

#include <bit>
#include <cstdint>

namespace nss {

// Finite classification used by correctness guards. This remains reliable in
// translation units compiled with aggressive floating-point optimizations.
inline bool is_finite_bits(float value) noexcept {
    constexpr std::uint32_t kExponentMask = 0x7f800000u;
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & kExponentMask) != kExponentMask;
}

// Row means of an m×n column-major group. mean[i] = avg_j group[i + j*lda].
void group_center_sub(float* group, int m, int n, int lda, float* mean);
void group_center_add(float* group, int m, int n, int lda, const float* mean);

// ClosedWNNM / WNNM SV update: S[k] = (s + sqrt(s² - C))/2 while s² > C.
// start_k=1 leaves S[0] untouched. Returns first unkept index (kept count).
int sv_shrink(float* S, int n, float constant, int start_k);

void soft_threshold(float* x, int n, float tau);
// Per-element τ: x[i] = sign(x[i]) * max(|x[i]| − tau[i], 0). tau[i] clamped to ≥ 0.
void soft_threshold_var(float* x, const float* tau, int n);

// x += delta * (y - x)
void iter_regularize(float* x, const float* y, int n, float delta);

// W_c = min(σ)/σ_c on each channel's rows; writes w2[c*area + i] = W_c².
// Returns min positive finite sigma, or 0 if none.
float channel_weight_diag(float* w2, int m, int nch, const float* sigma);

// X = (W²Y + ρ/2 (Z − A/ρ)) / (W² + ρ/2). Column-major, X/Z/A packed lda_x=m.
void admm_weighted_x(float* X, const float* Y, int ldy, const float* Z, const float* A, const float* w2, int m, int n,
                     float rho);

float dot_n(const float* a, const float* b, int n);
float ssd_vec(const float* a, const float* b, int n);
void axpy_n(float* y, const float* x, float a, int n);
void scale_n(float* y, float s, int n);

// Channel-major concat: col[c * block * block + py * block + px]. lda is unused inside a column
// (matches pack_patch); callers pass the start of column i at patches + i * lda.
void pack_patch_nch(float* col, int lda, const float* const* srcs, const int* strides, int nch, int x, int y,
                    int block, int width, int height);
void unpack_patch_nch(float* const* nums, float* const* dens, const int* strides, int nch, int x, int y,
                      const float* col, int block, int width, int height, float w);

}  // namespace nss
