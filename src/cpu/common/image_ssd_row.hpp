#pragma once

// Row-of-candidates block SSD kernel shared by the image-level matchers
// (NLH/TWSC spatial window scans in cpu/common/image.cpp). The kernel
// reproduces ssd_block's per-candidate reduction tree exactly, so distances
// are bit-identical to per-candidate ssd_block calls; only dispatch and
// anchor reloads are hoisted out of the candidate loop.
namespace nss {
using ImageSsdRowKernel = void (*)(const float* anchor, int sa, const float* cand, int sb, int block,
                                   int count, float* out);
// Resolves the runtime Highway target once; call sites keep the pointer.
ImageSsdRowKernel image_ssd_row_kernel();
} // namespace nss
