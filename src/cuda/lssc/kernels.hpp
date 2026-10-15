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
    float* sample_y;         // samples * m
    float* sample_a;         // atoms * samples (column per sample)
    float* sample_r;         // samples * m
    int* ksvd_sync;          // atoms * (kLsscSamples / 32 + 1): support masks, then done flags
    int* tiles;              // coding tiles (m <= 64), see lssc_tile_capacity
    int* chunks;             // np + 64 (m <= 64): k-means member counts per 64-patch chunk
    float* partial;          // tile capacity * atoms: partial row norms
    AggregatePatch* meta;    // np
};

// Device floats of the per-plane work arrays for np patches of m pixels.
std::size_t lssc_work_bytes(int m, int np);

// Coding tiles (64 members of one cluster at most) for np patches in
// `clusters` clusters: at most np / 64 rounded up plus one per cluster.
// w.tiles holds 1 + 2 capacity + clusters + 1 + clusters * atoms ints.
__host__ __device__ inline int lssc_tile_capacity(int np, int clusters) { return (np + 63) / 64 + clusters; }

void lssc_pack(const LsscWork& w, cudaStream_t stream);
// k-means (nss::lssc_cluster_workspace) entirely on the device: seeding, up
// to `rounds` rounds (assignment, member lists, centroids, empty-cluster
// steals; a round after one that changed nothing is skipped), then the member
// counts, offsets and lists in (cluster, patch) order. state: 2 device ints.
void lssc_cluster(const LsscWork& w, int rounds, int* state, cudaStream_t stream);

// Atoms [first, atoms) from the patches sources[a - first] (device ints),
// then every atom normalized.
void lssc_patch_atoms(const LsscWork& w, const int* sources, int first, cudaStream_t stream);
void lssc_ksvd(const LsscWork& w, cudaStream_t stream);
// Power-iteration estimate L of ||D||^2 into *lipschitz (device float); the
// coding rounds take their step 0.9 / L from it.
void lssc_lipschitz(const LsscWork& w, float* work, float* lipschitz, cudaStream_t stream);

// Coding: center the patches, run the rounds, reconstruct into values and
// fill the aggregation records.
void lssc_center(const LsscWork& w, cudaStream_t stream);
void lssc_round(const LsscWork& w, int round, const float* lipschitz, float sigma, cudaStream_t stream);
void lssc_reconstruct(const LsscWork& w, cudaStream_t stream);

}  // namespace nss_cuda
