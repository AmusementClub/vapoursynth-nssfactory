// SPDX-License-Identifier: GPL-2.0-only
#include "cuda/runtime/context.hpp"
#include "cuda/runtime/error.hpp"

#include <cstdint>
#include <mutex>
#include <set>
#include <stdexcept>

namespace nss_cuda {
namespace {

std::string prefix(const char* filter) {
    return std::string("nss_cuda.") + filter + ": ";
}

}  // namespace

std::string signature(const char* shared) {
    return std::string(shared) + kBackendSignature;
}

BackendArgs parse_backend_args(const VSAPI* vsapi, const VSMap* in, const char* filter) {
    BackendArgs args;
    int err = 0;
    const int64_t device = vsapi->mapGetInt(in, "device_id", 0, &err);
    if (!err) {
        if (device < 0 || device > INT32_MAX) throw std::invalid_argument(prefix(filter) + "device_id must be >= 0");
        args.device_id = static_cast<int>(device);
    }
    const int64_t streams = vsapi->mapGetInt(in, "num_streams", 0, &err);
    if (!err) {
        if (streams < 1 || streams > kMaxStreams) {
            throw std::invalid_argument(prefix(filter) + "num_streams must be in [1, " + std::to_string(kMaxStreams) + "]");
        }
        args.num_streams = static_cast<int>(streams);
    }
    return args;
}

DeviceInfo acquire_device(int device_id, const char* filter, VSCore* core, const VSAPI* vsapi) {
    const RuntimeInfo runtime = runtime_info();
    if (runtime.device_count == 0) {
        throw std::invalid_argument(prefix(filter) + "no CUDA device available (" + runtime.error + ")");
    }
    DeviceInfo device;
    try {
        device = device_info(device_id, runtime);
    } catch (const std::exception& error) {
        throw std::invalid_argument(prefix(filter) + error.what());
    }
    const std::string label = device.name + " (sm_" + std::to_string(device.cc_major) + std::to_string(device.cc_minor) + ")";
    if (device.support == Support::unsupported) {
        throw std::invalid_argument(prefix(filter) + label + " is not supported by this build (architectures " +
                                    compiled_architectures() +
                                    (runtime.ptx_jit_available() ? "" : "; the driver is too old to JIT its PTX") + ")");
    }
    if (device.support == Support::jit) {
        static std::mutex mutex;
        static std::set<int> warned;
        std::lock_guard lock(mutex);
        if (warned.insert(device.index).second) {
            const std::string message = "nss_cuda: " + label +
                " has no native code in this build; running the compute PTX through the driver JIT "
                "(untested, first use compiles and may take minutes)";
            vsapi->logMessage(mtWarning, message.c_str(), core);
        }
    }
    return device;
}

DeviceGuard::DeviceGuard(int device) {
    NSS_CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) NSS_CUDA_CHECK(cudaSetDevice(device));
}

DeviceGuard::~DeviceGuard() {
    if (previous_ >= 0) cudaSetDevice(previous_);
}

}  // namespace nss_cuda
