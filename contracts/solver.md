# Shared solver contract (version 2 models; TWSC v3 / NLH v5)

Matrices are column-major with explicit leading dimensions. SVD permits
1≤m≤256 and 1≤n≤32 in the existing shared decomposition, subject to each filter's additional restrictions. Singular
values are descending, nonnegative, and have min(m,n) entries. U has m rows and
Vt has n columns; padding is untouched. For m<n only the first m components are
active. Nonfinite input or an unrepresentable rescaled spectrum returns failure.

Jacobi uses relative pair correlation tolerance 1e-6, at most 32 sweeps, and
stops when no pair rotates. Input maxima outside [2^-8,2^8] trigger bounded
power-of-two scaling before QR; singular values are scaled back afterwards.
The angle normalizes its two coordinates before squaring, avoiding the
unbounded squared-zeta intermediate. Zero QR pivots preserve the unchanged row
in later R columns.

A squared relative effective-rank floor of 1e-14 times the largest initial
Jacobi column norm removes FP32 numerical null columns before normalization.
Their singular value and U column are zero. This prevents a numerical null
column from becoming a spurious unit direction in PCA. The policy is shared by
tiny, 8-column and cross-group implementations. It intentionally bounds resolvable
rank; it does not promise relative accuracy for components below that floor.

`src/cpu/finishers.hpp` is the common source for WNNM reconstruction/shrink,
NCSR centralization thresholds, and PCA reconstruction
plus mean restoration. Single and batch paths retain their physical layouts and
SIMD row access. U/S-only, Q-replay and MCWNNM Gram/fallback routes are retained.
TWSC no longer uses the old PCA/row-soft finisher. Its isolated dynamic
Householder QR/Jacobi decomposition supports up to 768 rows and 256 columns,
with 128 bounded sweeps, FP64 fallback and explicit quality checks. The complete
three-weight ADMM, numerical bounds, NLH transforms and formula-to-test map are
specified in [the TWSC v3 / NLH v5 contract](twsc-nlh.md). Other filters retain the
shared limits and model semantic version 2.

WNNM uses C=8*sqrt(2*n)*sigma². With residual centering it shrinks all components;
otherwise the leading component is retained under the existing rule. Adaptive
weight is 1/kept if requested and kept>0, else 1. GemmNN overwrites every active
output with zero when its inner dimension is zero, without reading A/B or writing
padding. Centered rank-zero reconstruction then restores the saved mean, identically
for packed and padded groups.

Tests independently synthesize orthogonal DCT factors with known spectra and
check scale covariance, effective U/V orthogonality, reconstruction, full PCA
projection, padding and leading-zero columns. Repeated spectra compare the
subspace/result rather than requiring one vector basis. Geometric spectra cover
near-rank deficiency. `test_wnnm_zero_rank` and the real VS zero-rank oracle verify
final estimates and weights, not just the singular values.

Compiler behavior is a separate validation axis. Clang fast-math can fold an
integer bit_cast finite check using nofpclass assumptions; integer classification
uses an opaque register barrier under Clang. Its machine instruction stream has
no added barrier instruction. GCC 13 and GCC 15.2, sanitizer and ISA results must
be reported separately. A passing GCC 13 run does not stand in for GCC 15.2.
