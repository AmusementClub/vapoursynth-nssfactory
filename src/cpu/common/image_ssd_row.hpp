// SPDX-License-Identifier: GPL-2.0-only
#pragma once

// Row-of-candidates block SSD kernel shared by the image-level matchers
// (NLH/TWSC spatial window scans in cpu/common/image.cpp). The kernel
// reproduces ssd_block's per-candidate reduction tree exactly, so distances
// are bit-identical to per-candidate ssd_block calls; only dispatch and
// anchor reloads are hoisted out of the candidate loop.
//
// Contract: anchor/cand/out must be non-null and count >= 1 — callers
// prevalidate, and the kernel returns without touching out otherwise (there
// is no sentinel buffer to write). If the view itself is invalid
// (block < 1 or sa/sb < block), out is filled with the same finite sentinel
// (numeric_limits<float>::max()) that ssd_block returns for invalid views.
namespace nss {
using ImageSsdRowKernel = void (*)(const float* anchor, int sa, const float* cand, int sb, int block,
                                   int count, float* out);
// Resolves the runtime Highway target once; call sites keep the pointer.
ImageSsdRowKernel image_ssd_row_kernel();
} // namespace nss
