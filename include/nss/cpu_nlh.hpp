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

// Orthonormal Haar (normalized LHWT) on length n in {1,2,4,8,16}. Other n is a no-op. in == out is allowed.
void haar1d(const float* in, float* out, int n);
void ihaar1d(const float* in, float* out, int n);

// For each of m rows of a column-major m×n group, write q nearest row indices
// (including self) into idx[r * q + j]. j=0 is always the query row.
void pixel_match(const float* group, int m, int n, int lda, int q, int* idx);

inline int nlh_filter_work_floats(int m, int n, int q, int lda = 0) {
    const int qq = q < 1 ? 1 : q;
    const int nn = n < 1 ? 1 : n;
    const int mm = m < 1 ? 1 : m;
    const int ld = lda > 0 ? lda : mm;
    const int n_use = nn >= 16 ? 16 : (nn >= 8 ? 8 : (nn >= 4 ? 4 : (nn >= 2 ? 2 : 1)));
    const int idx_f = (mm * qq * static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(float)) - 1) /
                      static_cast<int>(sizeof(float));
    return ld * nn + 2 * qq * n_use + idx_f + 16;
}

// 2D orthonormal Haar on the q×n_use similar-pixel matrix.
// n_use is the largest Haar length in {2,4,8,16} that is <= n (no zero-pad).
// Snapshot input; write back only the query row (j=0) so later queries cannot
// overwrite earlier pixels. Coeff hard uses λσ, then structural hard (last two
// rows except column 0). Aggregation counts kept after structural hard.
void nlh_filter_group(float* patches, int m, int n, int lda, int q, float sigma, bool wiener,
                      const float* ref_patches, float* weight_out, float* work, int work_floats = 0);

}  // namespace nss
