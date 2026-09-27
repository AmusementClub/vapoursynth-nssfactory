# vapoursynth-nssfactory

A work-in-progress VapourSynth factory for classical NSS (non-local self-similarity) denoisers — NLM, BM3D, WNNM, MCWNNM, TWSC, NLH, NCSR, LSSC — honoring the pre-AI era of these algorithms.

CPU plugin `libnss.so` / `libnss.dylib`, namespace `nss`. Linux x86-64 (AVX2 minimum), plus native AArch64/arm64 NEON builds in the recorded release matrix. Constant Gray / YUV / RGB 32-bit float clips only.

This project is WIP. Qualified NEON preview packages pass the Plan02 release gates on Ubuntu 24.04 / GCC 15 / AArch64 and macOS 27 / M4 Max / arm64 with VapourSynth R75 — see the [release and evidence summary](#release-and-evidence-summary) for compatibility, measured speedups and conditional floating-point differences. CUDA and Vulkan are future routes.

The current source additionally uses mixed-precision LSSC OMP by default: FP32 correlation screening, ordered FP64 refinement, and the existing FP64 solve/residual. The measured configuration-specific gains and regressions were accepted for integration (1080p essentially neutral on three platforms; small-image regressions up to −4.45% accepted). The existing preview packages and release-performance tables describe the frozen r3 source, before this change.

On supported Apple ARM builds, LSSC reconstruction can select an opt-in SME matrix leaf for eligible long products. Packing, transpose and ordinary SIMD fallback use Highway; only the 27-line FP32 ZA outer-product leaf uses Arm ACLE intrinsics. Measured on M4 Max at 1080p: b8 1.454×, b16 1.704×, with 762-case byte-identical outputs and zero measured quality loss. `Backend()` keeps reporting the Highway target and additionally reports `lssc_sme_compiled` / `lssc_sme_available`; availability does not mean every shape uses SME. Enable with `-DNSS_ENABLE_LSSC_SME=ON` for the isolated SME candidate; the default is the Highway-only path. The existing r3 preview packages predate this source change.

## Release and evidence summary

Condensed from the dated evidence reports of the September 2026 campaigns; the full reports are maintained outside the source tree. Ratios are same-host time ratios against each round's own frozen baseline and are **not multiplicative** across rounds.

**Release gates (Plan02, r3 preview).** N-FUNCTIONAL / N-ACCELERATED / N-RELEASE all passed within the recorded scope: Linux AArch64 (Ubuntu 24.04, C4A Neoverse V2, GCC 15.2, VapourSynth R75, glibc ≥ 2.38) and macOS arm64 (M4 Max, macOS 27, Apple Clang 21, ad-hoc signed), with native x86 C4 as regression control. Seven native lanes each ran 29 C++ tests, 9 real scripts and 762 public configurations; rolling passed 384 requests across 1/2/4/8 workers; 24 workspace-limit failure injections recovered. Windows (x64/ARM64), older macOS, universal/notarized builds and SVE/SME/Metal are outside the release scope.

**Measured performance.** NEON r3 (corrected functional baseline → final): BM3D Basic 1.54–1.55×, WNNM 1.34–1.38×, TWSC 1.32–1.35×, NCSR 1.31–1.36×, MCWNNM 1.11–1.35×, NLH/LSSC/NLM ≈1.0×. x86 default experiment mask 2305 (SortedTopK + bounded buffer reuse + ring/direct scratch): g16 chains ≈1.66×, g32 chains ≈1.71×, rolling 1.09–1.13×; lazy raster and homogeneous dispatch stayed off. The AVX2 default-dispatch campaign passed 24 primary configurations on C4 and Ryzen 5950X (known 5950X regressions: NLH q2 −5.4%, g8 −3.7%). TWSC G24/G32 FP64 batch (x86 default ON): ≈17–19% at 1080p. The most recent exact-track round (all outputs hash-identical): NLH image_match kernel 1.184×, LSSC k-means batch 1.093×, TWSC GEMM interleave 1.059×, NCSR row-major codes 1.086×.

**Quality findings.** In the default-parameter comparison matrix (312 runs), NLH is strongest on average (31.19 dB PSNR at 128²) and NLM's default h=1.2 is nearly inert. Recorded low-noise regressions: NCSR/LSSC can fall below the noisy input at σ≤5, and NLH blind estimation loses detail at low noise. Against author references in bounded comparisons, the current LSSC/NCSR defaults trail the full author pipelines by ≈5.4/5.1 dB; the experimental NCSR HQ kernel comes within ≈0.13–0.16 dB of the author NCSR but is not wired to the public `nss.NCSR` filter.

**Numerical policy and retained failures.** Cross-backend admission is conditional per-case with preserved failure arrays; two strict whole-group oracles (NCSR EMU c21, NLM c160 frame4) remain recorded failures. Plan01's no-regression gate did not pass and is retained as such (confirmed: NCSR 1080p up to +4.8%, BM3D temporal r1 up to +2.9%, NLM +2.0%). Recorded correctness costs: LSSC FP64 OMP up to +74% against the older FP32 path in one configuration; temporal migration costs at the 2026-09-06 integration (BM3D/WNNM motion r1 ≈ +20%).

## Usage

```python
core.nss.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2, string channels = "AUTO", int wmode = 0, float wref = 1.0, clip rclip = None])

core.nss.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8, int[] group_size = 8, int[] block_step, int[] bm_range = 7, int radius = 0, int[] ps_num, int[] ps_range = 4, string temporal_mode = "legacy", int rolling_chunk = 4, int rolling_cache_chunks, int rolling_cache_limit = 1])

core.nss.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 0, int adaptive_aggregation = 1, clip rclip = None])

core.nss.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 1, int adaptive_aggregation = 0, clip rclip = None, int admm_iter = 10, float rho = 3.0, float mu = 1.001, int iters = 2, float delta = 0.1])

core.nss.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0, int block_size = 8, int block_step = 1, int group_size = 90, int search_window = 60, int bm_range, int radius = 0, int ps_num = 2, int ps_range = 4, float lambda2 = 1.0, clip rclip = None, int iters = 12, float delta = 0.0, int admm_iter = 10, float rho = 0.5, float mu = 1.1, float tol = 1e-6, int memory_limit_mb])

core.nss.NLH(clip clip[, float[] sigma, string noise_model = "auto", int[] block_size, int[] block_step, int[] group_size, int[] search_window, int bm_range, int radius = 0, int ps_num = 2, int ps_range = 4, int[] q, clip rclip = None, int basic_iters, float lambda_basic, float hard_strength, int wiener_iters, float wiener_sigma_scale, int memory_limit_mb])

core.nss.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, clip rclip = None, int iters = 2, float delta = 0.1])

core.nss.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int radius = 0])

core.nss.VAggregate(clip clip, clip src[, int radius = 0, int[] planes])

core.nss.Version()  # returns version:data
```

With `radius = 0`, BM3D returns a normal-height spatial result and `temporal_mode="rolling"` has no
effect. With `radius > 0`, the default/`legacy` route returns the weighted intermediate for an explicit
`VAggregate`; `temporal_mode="rolling"` is an experimental route that returns a normalized normal-height
result directly. Earlier C4 workloads measured the pre-correction rolling route slower than legacy.
The corrected route remains experimental pending its new paired performance gate. Other temporal filters also expose
their weighted intermediate directly when `radius > 0`.

BM3D accepts `block_size` values 1, 2, 4, 8, 12, 16, and 32. The 12-point path is intended for
high-noise DCT profiles; 8 remains the general-purpose default.

NLM currently supports `wmode=0` (Welsch) only.

TWSC uses model semantic version 3 and NLH version 5. They replace their earlier
simplified algorithms. TWSC implements the complete three-weight ADMM objective
with the selected author preprocessing. NLH uses iterative Basic, repeated Wiener
gains and full pixel aggregation, with a calibrated **linear** Basic threshold
`k_h * hard_strength * sigma_channel`. Sigma remains an 8-bit noise standard
deviation. Omitted NLH fields select the measured Gray-low, Gray-high or real-noise
preset; explicit fields override only that field. The former `hard_tau` parameter
and its quadratic threshold rule are removed. These choices change outputs and
cost. There is no legacy runtime mode; TWSC also rejects `lambda1`, including zero.
The full TWSC/NLH formulas, defaults, limits and migration notes are part of the
model contract maintained with the project's design documents outside the source
tree. NLH v5 defaults come from a joint parameter search with recorded quality
tradeoffs (notably blind-estimate detail loss at low noise); v4 introduced the
calibrated linear threshold coefficient 2.025 against this project's BM3D Basic
response, and v5 retains sigma units while changing preset coefficients.
TWSC uses one stable balanced default (`block_size=8`, `block_step=1`,
`group_size=90`, `iters=12`) across noise levels; sigma controls denoising
strength rather than changing the compute geometry. Explicit TWSC fields override
the corresponding default. NLH estimates noise when `sigma` is omitted.
Explicit `sigma=0` preserves that input plane.
An output-preserving AVX2 large-group SVD validation optimization is integrated
(measured 1.14–1.24× on bounded pairs); measured `block_step` speed/quality
tradeoffs require explicit user parameters and are not defaults.
`bm_range=r` explicitly means `search_window=2*r+1`; supplying both is an error.
Earlier package/performance reports describe their frozen source, not these models.

## Compilation

CMake ≥ 3.24, C++20, VapourSynth API4 headers. [Highway](https://github.com/google/highway) 1.4.0 is fetched at configure time.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The first TWSC SVD-library milestone is opt-in and experimental. Set
`-DNSS_ENABLE_SVD_LIBRARY=ON` only with an LP64 LAPACKE/DGESDD provider; CMake
probes the header, symbols, `lapack_int` width and column-major ABI, then builds
the isolated `test_twsc_svd_library` target. Automatic TWSC dispatch remains
disabled while provider threading and resource bounds are being qualified.

Install `libnss.so` into the VapourSynth plugin directory.

Fresh x86 builds also enable `NSS_AVX2_DEFAULTS=ON`, with configuration-scoped
AVX2 ports validated on C4 and Ryzen 5950X. `NSS_AVX2_EXPERIMENT=0` adds no
experimental candidates; use `-DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT=0`
to retain the AVX2 reference routes. See the [AVX2 campaign report](docs/avx2-port-campaign.md)
for dispatch limits, numerical replays, per-configuration timings and measured
NLH fallback overhead.

AArch64 builds use `-DNSS_HWY_TARGET_MODE=neon` (also the ARM
`dynamic` target policy), with `NSS_AVX2_DEFAULTS=OFF` and experiment 0.
SVE/SVE2 are excluded until their separate port is validated. The
`portable-test` mode selects Highway EMU128 for diagnostic comparisons;
it requires a supported compiler (use Clang) and is not a release mode.
Build macOS arm64 separately and load `libnss.dylib`.
`core.nss.Backend()` reports compiled/runtime targets and an executed dispatch
probe; this does not certify acceleration of every kernel. See the
[release and evidence summary](#release-and-evidence-summary).
For native ARM build/host verification, run `tests/arm_native_gate.py --help`.
It defaults to GCC 15 for NEON, requires fresh build/result paths and a native
R75 runtime, and retains source/toolchain/backend identities and failure logs.
The extended TWSC/NLH mathematical gates require NumPy and SciPy in the test
Python environment; these are test dependencies, not plugin dependencies.
The gate includes guarded tail/stride fixtures and a 762-case public shape and
1/2/4-thread matrix. On Linux GCC, add `--sanitize` for ASan/UBSan; standalone
C++ tests also check leaks, while the external Python/VS runtime has leak
detection disabled. Pixel captures support separate cross-build comparisons;
this build/host gate is one component of the separate, completed release matrix. The r3 release evidence pack (readiness records, NEON matrices and dated reports) is maintained outside the source tree; when it is placed alongside this checkout, `python3 tests/arm_isa_lab/validate_release.py --source-root <frozen-release-source> --out <report.json>` re-validates it, keeping the archived package decision separate from current-source validation.

For a current source/build identity, generate a reproducibility manifest with:

```sh
python3 tests/source_manifest.py --out /tmp/nss-manifest.json --build <build-dir> \
  --baseline-label <baseline> --candidate-label <candidate>
```

It records the
commit, dirty status, non-ignored working-tree files, CMake identity and any
attached plugin/input/quality/performance artifacts. A release manifest must be
generated from the exact release source root, not from the current dirty tree.

Fresh builds select `NSS_BM_EXPERIMENT=2305`: AVX3 SortedTopK for b8 groups of
at least 16, BM3D patch/work reuse, and rolling target-ring/direct-scratch
aggregation. The ordinary spatial b8/g8 route is largely unchanged. Existing
CMake caches retain their previous setting; use `-DNSS_BM_EXPERIMENT=2305`
to select the BM combination. For the pure-correctness reference use
`-DNSS_BM_EXPERIMENT=0 -DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT=0`.
Cached-worst remains an alternative (`=2`); it cannot be combined with SortedTopK.
The bounded C4 selection campaign behind this mask measured g16 chains ≈1.66×,
g32 chains ≈1.71× and rolling 1.09–1.13× against its frozen baseline. Rolling remains explicitly requested through
`temporal_mode="rolling"`; its improvements here are relative to unoptimized
rolling, not a claim that it is faster than legacy mode.

The FP64 TWSC G24/G32 batch path is enabled by default on fresh x86 builds after
same-host complete-frame validation; use `-DNSS_TWSC_MIDGROUP_BATCH=OFF` for the
scalar control. ARM keeps it off until a native full-frame timing gate is
available. `-DNSS_GEMM_MULTI_ACCUM=ON` remains an opt-in exact
float-product/FP64-accumulation GEMM candidate. A source manifest records both
flags for each build.

## License

GPLv2. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The length-12/16/32/64 DCT kernels in `src/cpu/bm/dct_codelet_*.hpp` are generated
by [FFTW](https://www.fftw.org/) genfft (`gen_r2r`) and are distributed under
GPLv2 or later. Copyright (c) 1997-1999, 2003, 2007-14 Massachusetts Institute
of Technology and Matteo Frigo. They are mapped onto Highway and are not linked
against `libfftw3`. Regenerate with `tools/gen_dct_codelets.sh`.

### BM3D effective sigma and temporal contract

BM3D kernels use `sigma_eff = 0.75 * sigma_user / 255` for both Basic and Wiener.
The public default remains sigma=3 and bm_range=7. Only the scaled 8x8x8 kernel
converts effective noise to its coefficient scale (64); its inverse scale is 4096.
Changing block/group/search/temporal settings can change output, even at the same
sigma. This is a parameter calibration contract, not pixelwise BM3DCPU equality.

Temporal matching compares every candidate against the original center reference,
with independent forward/backward prediction and unique candidates. Only actual
frames participate at clip boundaries. The fixed-height fat layout is unchanged:
center c, slice t-c+radius contains the contribution to target t. VAggregate now
collects the target slice from all valid centers n-radius through n+radius. Zero
sigma writes identity only in the center slice; zero total weight copies src[n].
The shared single- and multichannel matcher fixes also affect other temporal NSS
filters. Existing intermediates should be regenerated with the same plugin build.
Rolling still returns normal-height output directly and remains experimental.


### Versioned CPU contracts and memory limits

Plan 01's common contracts (matching, temporal, noise, failure, solver and numerical tolerances) are maintained with the design documents outside the source tree. Fat intermediates now carry
version, radius, center and model/profile properties. `VAggregate` rejects
missing or conflicting identity. Its optional `allow_legacy=1` accepts wholly
untagged contributions only when the caller guarantees the current destination
semantics; regenerate intermediates from older incorrect temporal implementations.

All filters accept `memory_limit_mb` as the last optional argument (default
1024 MiB). A node exceeding its tracked buffer budget reports a VS error. It does
not silently reduce group size, radius or iteration count. Frame properties
`_NSSResourceBytes`, `_NSSResourcePeak` and `_NSSResourceLimit` expose byte
accounting; the six categories and framework/RSS boundaries are specified in
the failure contract. Rolling keeps one serialized workspace per
node, and evicted chunks remain charged while requests still hold them.

The user-facing BM3D noise profile remains unchanged across backends. For old NSS
generic configurations that previously used `sigma/255`, the migration is
`new_sigma = old_sigma * 4/3`; the old spatial fused 8×8×8 route keeps its sigma.
The noise contract records the exact compatibility scope and the pinned bm3dcpu
comparison. Current low-rank solvers use a documented relative
convergence and effective-rank policy; corrected old outputs are not bitwise
compatibility goldens. These correctness and safety changes have measured runtime
regressions in some configurations; the accepted optimization defaults remain
unchanged.

### NEON numerical compatibility

Cross-compiler/EMU outputs use the project's conditional numerical-tolerance policy; they are not guaranteed byte-identical. DCT/SSD near ties, iterative references and LSSC dictionary training can amplify ordinary rounding. The release evidence retains the actual outliers, independent replay evidence and quality guards (see the summary above). FP64 LSSC OMP fixes a near-null residual artifact but has a separately reported migration cost against the older FP32 implementation. The default mixed-precision search retains FP64 refinement, solve and residual updates, with the full FP64 search as its numerical fallback.
