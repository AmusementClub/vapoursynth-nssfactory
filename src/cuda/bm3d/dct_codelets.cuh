// SPDX-License-Identifier: GPL-2.0-only
// The FFTW genfft DCT codelets (lengths 12, 16, 32, 64; orthonormal DCT-II
// and DCT-III) as per-thread scalar code. The generated files are the ones
// the CPU plugin compiles through Highway (same arithmetic); this adapter
// maps their vector macros to float. Element k of a line lives at index 8 * k
// of the input and output arrays (the generated input stride).
#pragma once

#include <cuda_runtime.h>

namespace nss_cuda::codelet {

#define R float
#define V float
#define HWY_INLINE __device__ __forceinline__
#define DVK(name, value) const float name = static_cast<float>(value)
#define LDK(name) (name)
#define LD(ptr, vstride, alignment) (*(ptr))
#define ST(ptr, value, vstride, alignment) (*(ptr) = (value))
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMUL(a, b) ((a) * (b))
#define VFMA(a, b, c) fmaf((a), (b), (c))
#define VFMS(a, b, c) fmaf((a), (b), -(c))
#define VFNMS(a, b, c) fmaf(-(a), (b), (c))
#define VNEG(a) (-(a))
#define VLEAVE() ((void)0)

#include "cuda/bm3d/dct_codelet_fwd_n12_is8.hpp"
#include "cuda/bm3d/dct_codelet_fwd_n16_is8.hpp"
#include "cuda/bm3d/dct_codelet_fwd_n32_is8.hpp"
#include "cuda/bm3d/dct_codelet_fwd_n64_is8.hpp"
#include "cuda/bm3d/dct_codelet_inv_n12_is8.hpp"
#include "cuda/bm3d/dct_codelet_inv_n16_is8.hpp"
#include "cuda/bm3d/dct_codelet_inv_n32_is8.hpp"
#include "cuda/bm3d/dct_codelet_inv_n64_is8.hpp"

#undef R
#undef V
#undef HWY_INLINE
#undef DVK
#undef LDK
#undef LD
#undef ST
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMA
#undef VFMS
#undef VFNMS
#undef VNEG
#undef VLEAVE

}  // namespace nss_cuda::codelet
