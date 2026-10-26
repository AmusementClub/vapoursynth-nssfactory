// SPDX-License-Identifier: GPL-2.0-only
// LSSC on the device (the CPU lssc_denoise_plane model, version 4) for one plane:
//   - every grid patch is packed,
//   - the dictionary starts from DCT atoms plus LCG-sampled patches and gets
//     one K-SVD round on up to 256 sample patches (OMP with sparsity 8, then a
//     rank-one update per atom),
//   - its transpose, Gram matrix and inverse atom norms are formed,
//   - each patch is coded on its own by error-constrained Batch-OMP
//     (nss/params/lssc.hpp) with the CPU's operations in the CPU's order, one
//     warp a patch,
//   - the patches are aggregated in patch order and the noisy plane is added
//     at the image weight.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"
#include "nss/params/lssc.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

inline constexpr int kLsscSamples = 256;   // K-SVD sample patches
inline constexpr int kLsscSupport = 32;    // samples per atom update (nss::kSvdMaxN)
// Patches coded per alpha0 batch: their correlations (32 MiB at 256 atoms)
// stay in L2 between the product and the coding.
inline constexpr int kLsscBatch = 32768;

struct LsscWork {
    const float* plane;      // source plane (pitch = width)
    int width;
    RasterGrid grid;
    int block;
    int m;                   // block^2
    int np;                  // grid patches
    int atoms;               // min(256, np)
    int samples;             // min(256, np)
    int cap;                 // nss::lssc_omp_cap(m, atoms)
    float* patches;          // np * m: packed patches
    float* values;           // np * m: the centered patches, then the reconstructed ones
    float* mean;             // np
    float* yy;               // np: ||y||^2
    int* flags;              // np: 1 where a value is not finite
    float* alpha0;           // batch * atoms: D^T y of the batch being coded
    int batch;               // min(np, kLsscBatch)
    float* ubuf;             // lssc_code_scratch_bytes: the coding warps' u rows
    int code_slots;          // coding warps ubuf holds (lssc_code_slots)
    int* counters;           // one patch counter per batch
    float* dict;             // atoms * m, atom-major
    float* transpose;        // m * atoms, pixel-major
    float* gram;             // atoms * atoms
    float* inv_norm;         // atoms
    float* sample_y;         // samples * m
    float* sample_a;         // atoms * samples (column per sample)
    float* sample_r;         // samples * m
    int* ksvd_sync;          // atoms * (kLsscSamples / 32 + 1): support masks, then done flags
    AggregatePatch* meta;    // np
};

// Device bytes of the per-plane work arrays for np patches of m pixels.
std::size_t lssc_work_bytes(int m, int np);

void lssc_pack(const LsscWork& w, cudaStream_t stream);
// Atoms [first, atoms) from the patches sources[a - first] (device ints),
// then every atom normalized.
// sources[a - first] for atoms a in [first, atoms): the patch an LCG from
// `seed` picks (state = state * 1664525 + 1013904223, then state % np), the
// sequence the CPU model draws.
void lssc_lcg_sources(int* sources, int first, int atoms, int np, unsigned seed, cudaStream_t stream);
void lssc_patch_atoms(const LsscWork& w, const int* sources, int first, cudaStream_t stream);
void lssc_ksvd(const LsscWork& w, cudaStream_t stream);
// transpose, gram and inv_norm from dict.
void lssc_prepare(const LsscWork& w, cudaStream_t stream);
// Codes every patch into values and fills meta (sigma normalized to [0, 1]).
void lssc_code(const LsscWork& w, float sigma, cudaStream_t stream);
// Resident coding warps on a device; the warps a plane of np patches codes
// with (one per 32 patches at most), and the bytes of `slots` warps' u rows.
// Any number of warps gives the same output.
int lssc_code_warps(int device);
int lssc_code_slots(int device, int np);
std::size_t lssc_code_scratch_bytes(int slots, int m, int atoms);
// out = (num + weight src) / (den + weight), src where that denominator is 0.
void lssc_finish(const float* num, const float* den, const float* src, int width, int height, int pitch, float weight,
                 float* out, cudaStream_t stream);

}  // namespace nss_cuda
