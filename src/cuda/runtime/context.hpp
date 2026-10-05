// SPDX-License-Identifier: GPL-2.0-only
// Filter-creation helpers shared by every nss_cuda filter: the D14 backend
// argument tail, device acquisition with the D13 support policy, and the
// per-thread device guard used inside getFrame.
#pragma once

#include "cuda/runtime/device.hpp"
#include "nss/params/temporal.hpp"

#include <VapourSynth4.h>

#include <string>

namespace nss_cuda {

// Appended to every shared k<Filter>Signature (D14).
inline constexpr const char* kBackendSignature = "device_id:int:opt;num_streams:int:opt;";
// Three slots hide the per-frame host staging copies behind device work
// (1080p BM3D: 286 fps with 2, 357 with 3, 360 with 4 on an RTX 5080).
inline constexpr int kDefaultStreams = 3;
// temporal_mode when the argument is not given: finished frames, so that only
// they cross back to the host. "legacy" still returns the fat intermediate.
inline constexpr nss::TemporalMode kDefaultTemporalMode = nss::TemporalMode::Rolling;
inline constexpr int kMaxStreams = 16;

std::string signature(const char* shared);

struct BackendArgs {
    int device_id = 0;
    int num_streams = kDefaultStreams;
    bool streams_explicit = false;  // default may be lowered to fit memory_limit_mb
};
// Throws std::invalid_argument("nss_cuda.<filter>: ...") on invalid values.
BackendArgs parse_backend_args(const VSAPI* vsapi, const VSMap* in, const char* filter);

// Resolves device_id: unsupported devices are a creation error, JIT-only
// devices log one warning per device and process (D13). Throws
// std::invalid_argument with the "nss_cuda.<filter>: " prefix.
DeviceInfo acquire_device(int device_id, const char* filter, VSCore* core, const VSAPI* vsapi);

// Makes `device` current for the calling (VapourSynth worker) thread: the
// device's primary context, shared with every other CUDA user in the process.
// The thread's previous device is not restored. cudaSetDevice initializes the
// device it selects, so switching back would create a primary context on a
// device this filter never uses (the thread's default device is 0).
class DeviceGuard {
public:
    explicit DeviceGuard(int device);
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

}  // namespace nss_cuda
