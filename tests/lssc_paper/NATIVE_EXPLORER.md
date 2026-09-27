# Native LSSC exploration contract

This is an **opt-in research library**, not a new VapourSynth filter or a change
to `nss.LSSC`. Full author/paper reproduction remains unproven. P1/P2 Python math
is frozen. Mixed precision and algorithmic approximations are permitted only as
explicit, separately recorded exploration policies.

## Build and test

```sh
cmake -S . -B artifacts/lssc-native-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNSS_LSSC_EXPLORER=ON
cmake --build artifacts/lssc-native-build -j 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-paper/bin/python -m pytest -q tests/lssc_paper
```

`native.py` checks the C ABI and supports an alternate library through
`NSS_LSSC_NATIVE_LIBRARY`. The opt-in C++ implementation lives in `native/`.
There is no runtime Python dependency added to the production plugin.

## Retained definitions

- Finite grayscale input, nominal [0,1] units, no clipping. Sigma is converted
  from 8-bit units once by the CLI, never inferred from the clean image.
- Supplied initial dictionary: FP64, unit-norm columns, MATLAB patch-pixel order.
  External author assets remain research-only and are not bundled in source.
- Dense valid patches, raster locations, per-patch mean subtraction/restoration.
- P2 default: OLS energy-gain shared-support selection and overlapping members
  with covered seeds skipped. Literal raw-SSD threshold and even-window anchor
  remain explicit reference policies, not established MEX definitions.
- FP64 twice-reorthogonalized incremental QR, independent amplitudes per patch,
  chi-square residual budget, counting aggregation. Dynamic support is bounded
  by rank, not the old 8/32 storage limits. Numerical rank exhaustion is an error.
- The native OLS rank/eligibility threshold is squared projected norm greater
  than `128*eps64`, matching P2's eligibility policy. The optional Tropp score
  is diagnostic; this native QR rank policy is not claimed to reproduce every
  rank decision made by P1's SVD least-squares implementation.
- Streaming aggregation has a different FP64 addition order from Python. Exact
  repeat hashes and tolerance-based cross-implementation checks are separate.

## Precision and computation controls

| Control | Meaning |
|---|---|
| `precision=0` | Existing Highway FP64 correlations, FP64 dictionary and refits |
| `precision=1` | Existing FP32 Highway correlations; dictionary, QR, residuals and aggregation remain FP64 |
| `precision=2` | FP32 candidate scores plus FP64 near-tie refinement; heuristic, not certified bit-exact selection |
| `correlation=1` | Per-dictionary FP64 Gram, bounded initial correlations, rank-one recurrence and periodic refresh |
| `correlation=3` | Same Gram recurrence, with cached transposed dictionary and existing FP32 NN products |
| `correlation=7` | Same, using the existing SME leaf for eligible long products; unsupported/short shapes use logical-size Highway NN |
| `match_precision=1` | Existing `ssd_block` FP32 kernel on a cast pilot; grouping strategy remains separate |
| `batch` | Bounded patch packing/correlation panel; does not split a shared-support group |

Dictionary snapshots own their data. Every learning pass refreshes casts, norms
and Gram together. There is no global dictionary cache or request-order state.
All large native vectors use the existing `ResourceVector` and request budget.
FP64 full-image numerator and integer coverage are retained; all-patch coefficient
or reconstructed-occurrence matrices are not materialized. Python input/output
arrays and allocator/runtime overhead are outside the native tracked budget, so
campaigns also enforce and report process RSS.

## Explicit approximations

Group/support caps, residual-budget scaling, SSD threshold multipliers and atom
subsets are individually named policies. A support cap records an unmet budget
instead of relabeling it successful convergence. Group caps reuse `StableTopK`,
keep each seed and restore raster member order; they may increase group count
and cost. A 256-atom subset is not an equivalent prelearned 256-atom model.

Learning is **sampled, penalized alternating minimization**, not the author's
ODL schedule or an exact solution of constrained Eq.(8):

1. Deterministic stratified samples from original noisy patches only.
2. Singleton penalty `0.5*||Y-DA||_F^2 + lambda*||A||_1`, or group penalty with
   `||A||_(1,2)`. `lambda=learning_lambda*sigma`, times `sqrt(group_size)` for
   grouped codes. The lambda is not asserted equivalent to the chi-square budget.
3. FP32 GEMM/prox primitives, FP64 objective and step checks, monotone-restarted
   FISTA with backtracking. The final proximal-gradient mapping decides the
   reported convergence flag; iteration exhaustion remains visible.
4. FP64 accumulation of FP32 sufficient-statistic products. Image observations
   have equal weight; grouped training uses weight `1/group_size`. Every atom
   update is constrained to the unit ball and the fixed-code surrogate is checked
   for descent. Unused atoms are retained rather than randomly replaced.
5. Group plans are fixed after the learned pilot. Group-learning sample/cap
   budgets affect training only; dense final reconstruction is unchanged unless
   its separate approximation controls are enabled.

Updated columns may lie inside the unit ball; the pursuit uses actual norms.
FP64 output of a mixed solver does not imply that all products were FP64.
The FP32 transposed-correlation adapter is shared by pursuit scores and learning
gradients. Backend counters include both consumers, but not Gram preparation.

For SME, the 512-atom output is split into independent output-row panels of at
most 256 atoms. Patch rows are zero-padded only inside eligible multiplies (e.g.
81 to 96); centering, residual budgets, QR and aggregation retain the logical
81 rows and entire group. The dictionary is not reduced to 256 atoms. Short
groups and non-SME hosts do not pay the padding/splitting cost.

## Reused infrastructure

The native track reuses the production FP64 primitives unchanged
(`include/nss/cpu_twsc_full.hpp`, `src/cpu/twsc/math.cpp`, `src/cpu/lssc/gemm.hpp`,
`src/cpu/lssc/gemm.cpp`, `src/cpu/lssc/gemm_sme.cpp`, `tests/test_lssc_gemm.cpp`).
These provide the already-written FP64 GEMM, not a new third matrix backend.
FP32 GEMM, soft threshold, group prox, Lipschitz preparation, SSD, Top-K, checked
arithmetic and resource accounting come from the existing committed base.
The existing SME leaf, packing and guard-page/concurrent test were reused.
Parallel scheduling and GPU paths are not introduced speculatively.

## Evidence

`native_campaign.py` keeps each variant in a separate persistent process,
alternates measured order, verifies output/pilot/dictionary FP64 repeat hashes,
and records end-to-end adapter timing separately from native stage timing. It
archives the source and copies the exact loaded library before each campaign.
Warm passes, hashes, serialization and clean-image metrics are outside timing.
Inputs are preloaded (`timed_source_fills=0`). Failures remain in the results.

`native_audit.py` compares a saved fixed-D FP64 control to frozen P2/SVD at whole
image, every-group membership and sampled-support/refit levels. This is not
author-MEX equivalence or validation of a changed learning algorithm.

`native_sweep.py` verifies existing fixture hashes and runs cases serially.
`experiments/learning-v1.json` is an explicit exploration grid, not defaults.
For C4, `--require-cpu0` requires the verified two-thread SMT lane, records CPU
activity and rejects the whole paired set if any variant exceeds 1% sibling
activity or has steal. macOS local timings carry no C4 acceptance claim.
