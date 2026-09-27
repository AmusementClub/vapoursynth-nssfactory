# NCSR HQ optimization laboratory

The new `nss::ncsr_hq::denoise` CPU entry point implements one clustered-PCA
high-quality model. It has no `legacy`, `weighted`, or quality-mode selector.
The old high-quality prototype is compiled into `test_ncsr_hq` only and is
always called with its frozen high-quality configuration (`31`). It is an
engineering reference, not an official MATLAB reproduction oracle.

This integration adds the optimized CPU core and shared primitives. It does
not route the public VapourSynth `NCSR` filter to the new core: temporal
neighborhoods and `rclip` still need a deliberate high-quality-model migration.
No public defaults or existing filter behavior are changed by this laboratory.

## Three cumulative batches

1. **P0 — exact mechanical cleanup.** Remove unused SVD buffers; shrink only
   the target while retaining all neighbors for beta/variance; reuse pooled
   scratch; share one DCT fallback; store actual compact neighborhoods. Large
   owned vectors use `ResourceVector` and inherit the caller's `ResourceScope`.
2. **P1-R2 — real candidate batching.** `spatial_match_joint` groups up to four
   consecutive jobs and evaluates adjacent candidates in Highway lanes for
   blocks 6/7/9. It reuses the existing `SpatialSortedTopK` total ordering. This
   also removes the old x86 per-candidate worst-element scan; the speedup must
   not be attributed to SIMD alone. Other shapes and EMU128/SCALAR use the
   existing same-model spatial matcher. Existing batch APIs are unchanged.
3. **P2-R2 — dictionary-grouped projection/reconstruction.** Within 128-target
   tiles, deduplicate neighbor positions separately for each dictionary class.
   Project and reconstruct using the common `gemm_f32_f64` kernel. Discard
   coefficients every estimate/dictionary generation and commit results in
   original target order. No full-frame coefficient cache is allocated.

The search set, patch shape, stable match ordering, dictionary/training budget,
iteration schedule and aggregation order are unchanged. GEMM rounds each
product to float, then adds in the original order in double. It does not use
the existing float/FMA GEMM as a numerically interchangeable substitute.

## Numerical and memory contracts

- The candidate-lane SSD mirrors the pinned Highway 1.4 reduction tree.
  A Highway upgrade requires rerunning same-ISA exact tests; this is not an
  architecture-independent floating-point reduction claim.
- New SIMD translation units explicitly disable fast-math and implicit
  contraction; native SSD uses explicit Highway `MulAdd`.
- Tests cover non-dyadic input as well as random dyadic values, flat ties,
  borders, tails, invalid views, padded strides and guard pages.
- HQ tests compare entire buffers with the high-quality reference across
  patch/group sizes, sigma levels and dictionary refreshes; they also exercise
  concurrent calls, nonfinite-input rejection and budget failure/unwinding.
- The test-only reference retains the CPU library's original ISA/FMA compiler
  options. Moving it to a generic x86 test target without those options changes
  its arithmetic and is not a valid exact oracle for the frozen prototype.
- The public core accepts one finite spatial plane, normalized sigma, blocks
  1–16, groups 1–32, step 1–block and 1–12 iterations. Its caller owns the
  input/output extent and resource budget. It is not a temporal/color/rclip API.

## Reproduce tests

```sh
cmake -S . -B build-hq -DCMAKE_BUILD_TYPE=Release
cmake --build build-hq -j 2
ctest --test-dir build-hq --output-on-failure
```

On x86, the additional AVX2 and AVX3 tests verify the actually selected backend
and skip with code 77 when unavailable. Native NEON and forced EMU128 were
tested separately. ASan/UBSan checks include the guarded primitives; macOS
LeakSanitizer is unsupported and must not be reported as a passing leak check.

## Reproduce complete-frame A/B

Immutable source bundles, inputs, build/test logs, plugin hashes and output
buffers live in `artifacts/ncsr-hq-optimization/`. The source bundles
include isolated test-only VapourSynth adapters. Build those bundles separately
to reproduce the exact measured binaries; do not apply an experimental adapter
to the working public filter. The benchmark refuses a plugin without the HQ
trace, preventing an accidental comparison against the older public model.

```sh
taskset -c 0 python3 tests/ncsr_hq_bench.py campaign \
  --fixtures /path/to/fixtures128 \
  --plugins ref=/path/to/reference/libnss.so p2=/path/to/p2-r2/libnss.so \
  --cases house-128-s25 barbara-128-s25 peppers256-128-s25 \
  --repeats 3 --warmup 1 --samples 1 --out /path/to/results
python3 tests/ncsr_hq_report.py \
  --dataset /path/to/results=/path/to/fixtures128 --out /path/to/report
```

Use the verified C4 Spot environment, CPU 0 with CPU 1 as its idle SMT sibling,
identical source data and parameters, alternating order, preloaded source
frames and no concurrent builds. The audit recomputes hashes and quality from
raw output, excludes noisy CPU pairs, and requires at least two eligible pairs
per configuration for timing support. `--warmup 0` measures the first complete
full-shape frame; report it separately from warm measurements. An HD comparison
against P0 measures batches 2+3, not the cumulative gain against the original
unoptimized reference. Failed/regressed candidates remain archived.
