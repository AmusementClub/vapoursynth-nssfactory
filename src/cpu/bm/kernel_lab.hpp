#pragma once

// Opt-in microkernel laboratory. These entry points are absent from ordinary
// builds and return the actual target-local kernels to exclude dispatch cost.
namespace nss::detail {
using SsdRowKernel = void (*)(const float*, const float*, int, int, float*);
using Dct16Kernel = bool (*)(float*, int, bool);
SsdRowKernel ssd12_lab_kernel(int variant);
SsdRowKernel ssd16_lab_kernel(int variant);
Dct16Kernel dct16_lab_kernel(int variant);
}  // namespace nss::detail
