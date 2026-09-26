#pragma once

// Batched k-means step kernels for lssc_cluster_workspace.
//
// Bit-exactness contract (Track A): every per-pair SSD computed here uses the
// same ascending chunk order, the same per-element Sub + MulAdd sequence, the
// same ReduceSum tree, and the same scalar tail as SsdVec in
// src/cpu/common/vec.cpp, so each distance is bit-identical to a direct
// ssd_vec call on the same Highway target. Batching only interleaves
// independent accumulators across four centroids and resolves the dispatch
// target once per k-means step instead of once per (patch, centroid) pair.
// Winner selection keeps the reference strict-less-than, first-win,
// ascending-centroid order, and the accumulation step keeps ascending-patch
// order with the same FMA(1, x, y) per element as axpy_n.

namespace nss {

// One SSD, bit-identical to ssd_vec(a, b, n) on the same target.
float lssc_ssd_pair(const float* a, const float* b, int n);

// Four SSDs of `a` against centroids [0..3] stored contiguously with stride
// `n` starting at `cent`. out[r] is bit-identical to ssd_vec(a, cent + r*n, n).
void lssc_ssd4(const float* a, const float* cent, int n, float* out);

// Assignment pass of one k-means iteration. Writes assign[j] in [0, k) and
// returns nonzero exactly when the reference loop would set `changed`
// (first != 0 forces the flag without reading assign, preserving the
// reference short-circuit over uninitialized assign storage at iteration 0).
int lssc_cluster_assign_step(const float* patches, int m, int n, int lda,
                             const float* cent, int k, int first, int* assign);

// Centroid accumulation pass: for ascending j, cent[assign[j]] += patches[j]
// elementwise via FMA(1, x, y) (bit-identical to axpy_n(..., 1.f, m)) and
// ++acc[assign[j]]. Callers zero cent/acc first, exactly like the reference.
void lssc_cluster_accum_step(const float* patches, int m, int n, int lda,
                             const int* assign, float* cent, int* acc);

// Farthest-member scan for the empty-cluster steal: ascending j, skipping
// acc[assign[j]] <= 1, strict-greater first-win over per-pair SSD against the
// member's own centroid. Returns the steal index, or -1 when none qualifies.
int lssc_cluster_farthest(const float* patches, int m, int n, int lda,
                          const int* assign, const int* acc, const float* cent);

}  // namespace nss
