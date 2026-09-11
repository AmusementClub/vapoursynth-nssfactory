# vapoursynth-nssfactory

A work-in-progress VapourSynth factory for classical NSS (non-local self-similarity) denoisers — NLM, BM3D, WNNM, MCWNNM, TWSC, NLH, NCSR, LSSC — honoring the pre-AI era of these algorithms.

CPU plugin `libnss.so` / `libnss.dylib`, namespace `nss`. Linux x86-64 (AVX2 minimum), plus native AArch64/arm64 NEON builds in the recorded release matrix. Constant Gray / YUV / RGB 32-bit float clips only.

This project is WIP. Qualified NEON preview packages pass the Plan02 release gates on Ubuntu 24.04 / GCC 15 / AArch64 and macOS 27 / M4 Max / arm64 with VapourSynth R75. See the [release report](PLAN02_RELEASE_20260907.md) for compatibility, measured speedups and conditional floating-point differences. CUDA and Vulkan are future routes.

The current source additionally uses mixed-precision LSSC OMP by default: FP32 correlation screening, ordered FP64 refinement, and the existing FP64 solve/residual. The measured configuration-specific gains and regressions were accepted for integration; see the [integration record](PLAN02_MIXED_INTEGRATION_20260908.md). The existing preview packages and release-performance tables describe the frozen r3 source, before this change.

On supported Apple ARM builds, LSSC reconstruction also selects an SME matrix leaf for eligible long products. Packing, transpose and ordinary SIMD fallback use Highway; only the 27-line FP32 ZA outer-product leaf uses Arm ACLE intrinsics. The measured M4 Max gains, numerical/resource checks and fallback coverage are recorded in [LSSC SME integration](LSSC_SME_INTEGRATION_20260908.md). `Backend()` keeps reporting the Highway target and additionally reports `lssc_sme_compiled` / `lssc_sme_available`; availability does not mean every shape uses SME. Configure with `-DNSS_ENABLE_LSSC_SME=OFF` for the Highway-only control. The existing r3 preview packages predate this source change.

## Usage

```python
core.nss.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2, string channels = "AUTO", int wmode = 0, float wref = 1.0, clip rclip = None])

core.nss.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8, int[] group_size = 8, int[] block_step, int[] bm_range = 7, int radius = 0, int[] ps_num, int[] ps_range = 4, string temporal_mode = "legacy", int rolling_chunk = 4, int rolling_cache_chunks, int rolling_cache_limit = 1])

core.nss.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 0, int adaptive_aggregation = 1, clip rclip = None])

core.nss.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 1, int adaptive_aggregation = 0, clip rclip = None, int admm_iter = 10, float rho = 3.0, float mu = 1.001, int iters = 2, float delta = 0.1])

core.nss.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0, int block_size, int block_step = 1, int group_size, int search_window = 60, int bm_range, int radius = 0, int ps_num = 2, int ps_range = 4, float lambda2 = 1.0, clip rclip = None, int iters, float delta = 0.0, int admm_iter = 10, float rho = 0.5, float mu = 1.1, float tol = 1e-6, int memory_limit_mb])

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
See [TWSC/NLH formulas, defaults, limits and migration](contracts/twsc-nlh.md).
The [NLH defaults and optimization report](NLH_DEFAULTS_OPTIMIZATION_20260909.md)
records the joint parameter search, quality tradeoffs, admitted SIMD shapes and
separate parameter/kernel measurements. The earlier
[linear-NLH calibration report](docs/nlh-linear-calibration-20260908.md) records the
v4 BM3D response reference; v5 retains sigma units while changing preset coefficients.
The earlier [alignment evidence report](docs/twsc-nlh-alignment-20260908.md) separates
independent numerical checks, quality changes and same-model timings.
TWSC chooses block/group/iterations from active-channel noise RMS; NLH estimates
noise when `sigma` is omitted. Explicit `sigma=0` preserves that input plane.
TWSC's [performance report](docs/twsc-speed-20260909.md) records the AVX2 SVD
validation optimization and measured `block_step` speed/quality tradeoffs.
`bm_range=r` explicitly means `search_window=2*r+1`; supplying both is an error.
Earlier package/performance reports describe their frozen source, not these models.

## Compilation

CMake ≥ 3.24, C++20, VapourSynth API4 headers. [Highway](https://github.com/google/highway) 1.4.0 is fetched at configure time.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

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
[Plan02 release report and validation scope](PLAN02_RELEASE_20260907.md).
For native ARM build/host verification, run `tests/arm_native_gate.py --help`.
It defaults to GCC 15 for NEON, requires fresh build/result paths and a native
R75 runtime, and retains source/toolchain/backend identities and failure logs.
The extended TWSC/NLH mathematical gates require NumPy and SciPy in the test
Python environment; these are test dependencies, not plugin dependencies.
The gate includes guarded tail/stride fixtures and a 762-case public shape and
1/2/4-thread matrix. On Linux GCC, add `--sanitize` for ASan/UBSan; standalone
C++ tests also check leaks, while the external Python/VS runtime has leak
detection disabled. Pixel captures support separate cross-build comparisons;
this build/host gate is one component of the separate, completed release matrix. To verify the recorded r3 release evidence after this checkout advances, use `python3 tests/arm_isa_lab/validate_release.py --source-root build-plan02-release-source-r3 --out artifacts/c4a/plan02-mixed-integration-20260908/r3-release-validation.json`. The explicit source root keeps the archived package decision separate from current-source validation.

Fresh builds select `NSS_BM_EXPERIMENT=2305`: AVX3 SortedTopK for b8 groups of
at least 16, BM3D patch/work reuse, and rolling target-ring/direct-scratch
aggregation. The ordinary spatial b8/g8 route is largely unchanged. Existing
CMake caches retain their previous setting; use `-DNSS_BM_EXPERIMENT=2305`
to select the BM combination. For the pure-correctness reference use
`-DNSS_BM_EXPERIMENT=0 -DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT=0`.
Cached-worst remains an alternative (`=2`); it cannot be combined with SortedTopK.
See [bounded C4 selection](docs/c4-selection-20260906.md) for measured algorithm,
group-size and temporal cases. Rolling remains explicitly requested through
`temporal_mode="rolling"`; its improvements here are relative to unoptimized
rolling, not a claim that it is faster than legacy mode.

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

Plan 01's common contracts are in [contracts/](contracts/), with numerical limits
in [tolerances.json](contracts/tolerances.json). Fat intermediates now carry
version, radius, center and model/profile properties. `VAggregate` rejects
missing or conflicting identity. Its optional `allow_legacy=1` accepts wholly
untagged contributions only when the caller guarantees the current destination
semantics; regenerate intermediates from older incorrect temporal implementations.

All filters accept `memory_limit_mb` as the last optional argument (default
1024 MiB). A node exceeding its tracked buffer budget reports a VS error. It does
not silently reduce group size, radius or iteration count. Frame properties
`_NSSResourceBytes`, `_NSSResourcePeak` and `_NSSResourceLimit` expose byte
accounting; the six categories and framework/RSS boundaries are specified in
[failure.md](contracts/failure.md). Rolling keeps one serialized workspace per
node, and evicted chunks remain charged while requests still hold them.

The user-facing BM3D noise profile remains unchanged across backends. For old NSS
generic configurations that previously used `sigma/255`, the migration is
`new_sigma = old_sigma * 4/3`; the old spatial fused 8×8×8 route keeps its sigma.
See [noise.md](contracts/noise.md) for the exact compatibility scope and the
pinned bm3dcpu comparison. Current low-rank solvers use a documented relative
convergence and effective-rank policy; corrected old outputs are not bitwise
compatibility goldens. These correctness and safety changes have measured runtime
regressions in some configurations; the accepted optimization defaults remain
unchanged.

### NEON numerical compatibility

Cross-compiler/EMU outputs use the conditional policy in `contracts/tolerances.json`; they are not guaranteed byte-identical. DCT/SSD near ties, iterative references and LSSC dictionary training can amplify ordinary rounding. The release report retains the actual outliers, independent replay evidence and quality guards. FP64 LSSC OMP fixes a near-null residual artifact but has a separately reported migration cost against the older FP32 implementation. The default mixed-precision search retains FP64 refinement, solve and residual updates, with the full FP64 search as its numerical fallback.
