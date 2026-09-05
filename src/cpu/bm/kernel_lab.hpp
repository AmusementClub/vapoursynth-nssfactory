#pragma once

#include "nss/cpu_api.hpp"

// Opt-in microkernel laboratory. These entry points are absent from ordinary
// builds and return the actual target-local kernels to exclude dispatch cost.
namespace nss::detail {
using SsdRowKernel = void (*)(const float*, const float*, int, int, float*);
using DctKernel = bool (*)(float*, int, bool);
using Dct16Kernel = DctKernel;  // Preserve the original lab interface.
DctKernel dct_lab_kernel(int block, int variant);
DctKernel dct12_lab_kernel(int variant);
struct MatchReplayItem { int x, y; float distance; };
using Match8Kernel = int (*)(const float*, int, int, int, int, int, int, Match*);
using Match8Capture = int (*)(const float*, int, int, int, int, int, int, Match*, MatchReplayItem*);
// Replay returns -1 when production would recompute via its non-finite fallback.
using Match8Replay = int (*)(const MatchReplayItem*, int, int, int, int, int, int, Match*);
Match8Kernel match8_lab_kernel(int variant);
Match8Capture match8_capture_kernel(int variant = 0);
Match8Replay match8_replay_kernel(int variant);
SsdRowKernel ssd12_lab_kernel(int variant);
SsdRowKernel ssd16_lab_kernel(int variant);
Dct16Kernel dct16_lab_kernel(int variant);
}  // namespace nss::detail
