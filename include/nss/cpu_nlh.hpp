#pragma once

#include "nss/params.hpp"

namespace nss {

struct Match;
struct MatchBatchItem;

// NLH's common spatial shape keeps the 15 non-reference matches sorted while
// traversing candidates. Other group sizes and temporal matching stay on the
// shared matcher path.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
int nlh_spatial_match16(const float* ref, int stride, int width, int height, int bx, int by, int block,
                        int bm_range, Match* out);
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
int nlh_spatial_match_batch(const float* ref, int stride, int width, int height, const MatchBatchItem* items,
                            int count, Match* matches, int match_stride, int* counts);

// Orthonormal Haar (normalized LHWT), powers of two in [1,64]. in == out is allowed.
void haar1d(const float* in, float* out, int n);
void ihaar1d(const float* in, float* out, int n);

// For each of m rows of a column-major m×n group, write q nearest row indices
// (including self) into idx[r * q + j]. j=0 is always the query row.
void pixel_match(const float* group, int m, int n, int lda, int q, int* idx);

}  // namespace nss
