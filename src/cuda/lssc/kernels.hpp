// SPDX-License-Identifier: GPL-2.0-only
// LSSC on the device (the CPU lssc_denoise_plane model) for one plane:
//   - every grid patch is packed; k-means (8 rounds, evenly seeded, empty
//     clusters steal the farthest member) groups them,
//   - the dictionary starts from DCT atoms plus LCG-sampled patches and gets
//     one K-SVD round on up to 256 sample patches (OMP with sparsity 8, then a
//     rank-one update per atom),
//   - per cluster, 16 proximal-gradient rounds of simultaneous sparse coding:
//     a gradient step on the centered patches followed by the group soft
//     threshold over the cluster's rows of coefficients,
//   - reconstruction D A + mean with the CPU's sanity fallbacks.
// Patch-order sums (centroids, row norms) are taken in ascending patch order
// by one thread each, so results are run-to-run identical.
#pragma once

#include "cuda/common/aggregate.hpp"
#include "cuda/common/match.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nss_cuda {

inline constexpr int kLsscSamples = 256;   // K-SVD sample patches
inline constexpr int kLsscSupport = 32;    // samples per atom update (nss::kSvdMaxN)
inline constexpr int kLsscRounds = 16;     // proximal-gradient rounds

struct LsscWork {
    const float* plane;      // source plane (pitch = width)
    int width;
    RasterGrid grid;
    int block;
    int m;                   // block^2
    int np;                  // grid patches
    int atoms;               // min(256, np)
    int clusters;            // min(64, np)
    int samples;             // min(256, np)
    float* patches;          // np * m: packed patches, centered in place for the coding
    float* values;           // np * m: residuals, then the reconstruction
    float* coef;             // np * atoms: coefficients, patch-major
    float* mean;             // np
    float* dict;             // atoms * m, atom-major
    float* centroids;        // clusters * m
    int* assign;             // np
    int* counts;             // clusters
    int* members;            // np: patch indices ordered by (cluster, patch)
    int* offsets;            // clusters + 1
    float* scale;            // clusters * atoms
    int* flags;              // clusters * 2: exploded, stopped; then 4 ints of scratch
    float* distance;         // np
    float* sample_y;         // samples * m
    float* sample_a;         // atoms * samples (column per sample)
    float* sample_r;         // samples * m
    float* update;           // (kLsscSupport + 2) * m: E, the old atom, the new one
    AggregatePatch* meta;    // np
};

// Device floats of the per-plane work arrays for np patches of m pixels.
std::size_t lssc_work_bytes(int m, int np);

void lssc_pack(const LsscWork& w, cudaStream_t stream);
// k-means steps. *changed (device int) is OR-ed with "an assignment changed".
void lssc_seed_centroids(const LsscWork& w, cudaStream_t stream);
void lssc_assign(const LsscWork& w, bool first, int* changed, cudaStream_t stream);
void lssc_accumulate(const LsscWork& w, cudaStream_t stream);
// Farthest member of a cluster with more than one member into *result
// (device int, -1 when none); then lssc_steal moves patch j into cluster c.
void lssc_farthest(const LsscWork& w, int* result, cudaStream_t stream);
void lssc_steal(const LsscWork& w, int cluster, int patch, cudaStream_t stream);
void lssc_count(const LsscWork& w, cudaStream_t stream);

// Atoms [first, atoms) from the patches sources[a - first] (device ints),
// then every atom normalized.
void lssc_patch_atoms(const LsscWork& w, const int* sources, int first, cudaStream_t stream);
void lssc_ksvd(const LsscWork& w, cudaStream_t stream);
// Power-iteration estimate of ||D||^2 into *lipschitz (device float).
void lssc_lipschitz(const LsscWork& w, float* work, float* lipschitz, cudaStream_t stream);

// Coding: center the patches, run the rounds, reconstruct into values and
// fill the aggregation records.
void lssc_center(const LsscWork& w, cudaStream_t stream);
void lssc_round(const LsscWork& w, int round, float mu, float sigma, cudaStream_t stream);
void lssc_reconstruct(const LsscWork& w, cudaStream_t stream);

}  // namespace nss_cuda
