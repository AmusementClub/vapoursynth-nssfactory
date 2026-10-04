// SPDX-License-Identifier: GPL-2.0-only
// Device discovery and the D13 support classification for libnss_cuda.
// Host C++ only: no kernels are declared here except the load probe.
#pragma once

#include <cstddef>
#include <string>

namespace nss_cuda {

// D13: native = a compiled cubin runs on the device (same major, cubin minor
// <= device minor); jit = only the embedded PTX can be JIT-compiled (untested,
// best effort); unsupported = no image can run (below the oldest PTX target).
enum class Support { native, jit, unsupported };

const char* support_name(Support support);
// ptx_jit_available: the driver can JIT this build's PTX, i.e. the driver's
// CUDA version is at least the toolkit version the plugin was built with.
Support classify(int cc_major, int cc_minor, bool ptx_jit_available = true);
const char* compiled_architectures();

struct RuntimeInfo {
    int driver_version = 0;   // cudaDriverGetVersion, 0 when no driver
    int runtime_version = 0;  // cudaRuntimeGetVersion (static runtime)
    int device_count = 0;
    std::string error;        // non-empty when the driver/device query failed
    // PTX produced by a newer toolkit than the driver cannot be JIT-compiled
    // (cudaErrorUnsupportedPtxVersion); cubins are unaffected.
    bool ptx_jit_available() const { return driver_version >= runtime_version; }
};
RuntimeInfo runtime_info();

struct DeviceInfo {
    int index = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    std::size_t total_memory = 0;
    int multiprocessors = 0;
    Support support = Support::unsupported;
};
// Throws std::runtime_error with a user-facing message on failure.
DeviceInfo device_info(int device, const RuntimeInfo& runtime);

struct ProbeResult {
    bool ok = false;
    double milliseconds = 0.0;  // first launch includes module load / PTX JIT
    std::string error;
};
// Launches a trivial kernel on `device` and checks its result; proves that a
// usable image (cubin or JIT-compiled PTX) exists for the device.
ProbeResult probe(int device);

}  // namespace nss_cuda
