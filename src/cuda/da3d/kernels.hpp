// SPDX-License-Identifier: GPL-2.0-only
// nss_cuda.DA3D device pipeline: the CPU model (include/nss/params/da3d.hpp)
// ported operation for operation, so the output is the CPU's bit for bit.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

struct Da3dTables {
    const float* twiddle;  // 64 re, then 64 im: e^{-2 pi i t / 64}
    const float* dist;     // 4096 regression distances, then 4096 weight distances
};

struct Da3dUnit {
    int channels = 1;  // 1, or 3 (joint RGB)
    int width = 0, height = 0;
    bool rgb = false;
    float sigma = 0.f;  // sigma / 255
    const float* noisy[3]{};
    const float* guide[3]{};
    float* out[3]{};
    // RGB scratch: the transformed planes (3 noisy, then 3 guide), plane floats each
    float* transformed = nullptr;
};

// Device work arrays for a plane of width x height and channels.
std::size_t da3d_work_bytes(int width, int height, int channels);
// Upload of the constant tables; table holds da3d_table_floats() floats.
std::size_t da3d_table_floats();
void da3d_fill_tables(float* host);
// Runs DA3D on the unit; work holds da3d_work_bytes. stats receives
// (patches, guide-only patches) of the unit, summed, on the host after the
// stream has synchronized: stats_device is 2 ints of device memory, cleared here.
void da3d_run(const Da3dUnit& unit, const Da3dTables& tables, void* work, std::size_t work_bytes,
              unsigned long long* stats_device, cudaStream_t stream);

}  // namespace nss_cuda
