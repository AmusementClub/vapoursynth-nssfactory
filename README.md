# vapoursynth-nssfactory

A work-in-progress VapourSynth factory for classical NSS (non-local self-similarity) denoisers — NLM, BM3D, WNNM, MCWNNM, TWSC, NLH, NCSR, LSSC — honoring the pre-AI era of these algorithms.

CPU plugin `libnss.so` / `libnss.dylib`, namespace `nss`. Linux x86-64 (AVX2 minimum), plus native AArch64/arm64 NEON builds in the recorded release matrix. Constant Gray / YUV / RGB 32-bit float clips only.

This project is WIP. Qualified NEON preview builds pass the release gates on Ubuntu 24.04 / GCC 15 / AArch64 and macOS 27 / M4 Max / arm64 with VapourSynth R75. CUDA and Vulkan are future routes.

The current source additionally uses mixed-precision LSSC OMP by default: FP32 correlation screening, ordered FP64 refinement, and the existing FP64 solve/residual. Published preview packages predate this change.

On supported Apple ARM builds, LSSC reconstruction can select an opt-in SME matrix leaf for eligible long products. Packing, transpose and ordinary SIMD fallback use Highway; only the 27-line FP32 ZA outer-product leaf uses Arm ACLE intrinsics. `Backend()` keeps reporting the Highway target and additionally reports `lssc_sme_compiled` / `lssc_sme_available`; availability does not mean every shape uses SME. Enable with `-DNSS_ENABLE_LSSC_SME=ON` for the isolated SME candidate; the default is the Highway-only path. Published preview packages predate this change.

## Usage

See the [API guide](docs/api/README.md) for per-filter parameters, defaults,
measured cost tiers and pitfalls.

```python
core.nss.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2, string channels = "AUTO", int wmode = 0, float wref = 1.0, clip rclip = None])

core.nss.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8, int[] group_size = 8, int[] block_step, int[] bm_range = 7, int radius = 0, int[] ps_num, int[] ps_range = 4])

core.nss.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 0, int adaptive_aggregation = 1, clip rclip = None])

core.nss.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 1, int adaptive_aggregation = 0, clip rclip = None, int admm_iter = 10, float rho = 3.0, float mu = 1.001, int iters = 2, float delta = 0.1])

core.nss.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0, int block_size = 8, int block_step = 1, int group_size = 90, int search_window = 60, int bm_range, int radius = 0, int ps_num = 2, int ps_range = 4, float lambda2 = 1.0, clip rclip = None, int iters = 12, float delta = 0.0, int admm_iter = 10, float rho = 0.5, float mu = 1.1, float tol = 1e-6, int memory_limit_mb])

core.nss.NLH(clip clip[, float[] sigma, string noise_model = "auto", int[] block_size, int[] block_step, int[] group_size, int[] search_window, int bm_range, int radius = 0, int ps_num = 2, int ps_range = 4, int[] q, clip rclip = None, int basic_iters, float lambda_basic, float hard_strength, int wiener_iters, float wiener_sigma_scale, int memory_limit_mb])

core.nss.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, clip rclip = None, int iters = 2, float delta = 0.1])

core.nss.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int radius = 0])

core.nss.VAggregate(clip clip, clip src[, int radius = 0, int[] planes])

core.nss.Version()  # returns version:data
```

With `radius = 0`, a temporal filter returns a normal-height spatial result. With `radius > 0` it returns
the weighted intermediate for an explicit `VAggregate` call. LSSC has no temporal mode.

BM3D accepts `block_size` values 1, 2, 4, 8, 12, 16, and 32. The 12-point path is intended for
high-noise DCT profiles; 8 remains the general-purpose default.

NLM currently supports `wmode=0` (Welsch) only.

TWSC implements the complete three-weight ADMM objective; NLH uses iterative
Basic/Wiener filtering with a calibrated linear threshold. Both take sigma as
an 8-bit noise standard deviation. TWSC uses one balanced default
(`block_size=8`, `block_step=1`, `group_size=90`, `iters=12`) across noise
levels; sigma controls denoising strength rather than changing the compute
geometry. NLH estimates noise when `sigma` is omitted, and any other omitted
field falls back to a measured Gray-low, Gray-high or real-noise preset;
explicit fields override only that field. Explicit `sigma=0` preserves that
input plane. `bm_range=r` explicitly means `search_window=2*r+1`; supplying
both is an error. See the API guide pages for [TWSC](docs/api/twsc.md) and
[NLH](docs/api/nlh.md) for the full parameter and model details.

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

### CUDA plugin

`-DNSS_ENABLE_CUDA=ON` also builds `libnss_cuda` (namespace `nss_cuda`). It
needs nvcc from CUDA 12.4 or newer; Windows builds use nvcc with MSVC `cl.exe`.
All nine filters are available:

- `core.nss_cuda.BM3D`: spatial, `ref`/Wiener, and temporal. See
  `docs/api/bm3d.md`.
- `core.nss_cuda.NLM`: all channel modes, temporal `d`, and `rclip`. See
  `docs/api/nlm.md`.
- `core.nss_cuda.WNNM`, `core.nss_cuda.MCWNNM` and `core.nss_cuda.NCSR`: spatial
  and temporal. See `docs/api/wnnm.md`,
  `docs/api/mcwnnm.md` and `docs/api/ncsr.md`.
- `core.nss_cuda.NLH` and `core.nss_cuda.TWSC`: given or blind sigma,
  Gray/YUV/RGB, and temporal. See `docs/api/nlh.md`
  and `docs/api/twsc.md`.
- `core.nss_cuda.LSSC`. See `docs/api/lssc.md`.
- `core.nss_cuda.VAggregate`, which also accepts CPU fat intermediates
  (a host-side reduction).
- `core.nss_cuda.Version()`.
- `core.nss_cuda.Backend()`, which reports the device, driver/runtime
  versions and the support level.

Each filter takes the same arguments as its `nss` counterpart, plus
`device_id` and `num_streams` at the end, and gives the same errors under the
`nss_cuda` name. The one difference in behaviour is the temporal output
(below). Outputs agree with the CPU plugin to 60 dB PSNR or better
(not bit-for-bit) and are identical from run to run.

- **`device_id`** (default 0) selects the GPU; one filter instance uses one
  device.
- **`num_streams`** (default up to 3) is the number of frames an instance
  processes at once. Each stream owns its device buffers, so the default is
  lowered until the streams fit `memory_limit_mb`.
- **`memory_limit_mb`** caps the instance's device and pinned host memory. A
  limit that cannot hold one stream is a creation error; nothing degrades
  silently.
- **Temporal filtering.** With `radius > 0` the device filters return
  finished, normal-height frames: no `VAggregate` call is needed.
  - BM3D, WNNM and single-round NCSR keep the temporal accumulation on the
    device; the other filters reduce the intermediate inside the filter.
  - `temporal_mode="legacy"` returns the fat intermediate for `VAggregate`
    instead, exactly as the CPU plugin does; use it to mix backends or to
    keep an existing script unchanged.
  - The finished-frame mode needs more memory than legacy; at 4K raise
    `memory_limit_mb` (the creation error names the amount).
- **Driver.** A prebuilt plugin needs a driver for the CUDA release it was
  built with (12.9 for the release packages) or newer.

- **Default architectures:** native code for sm_75, sm_86, sm_89 and sm_120,
  plus compute_75 PTX. Toolkits older than 12.8 cannot target sm_120, so RTX 50
  GPUs run that PTX through the driver JIT.
- **Other sm_75+ GPUs:** they also run the PTX through the driver JIT, which is
  untested and best effort. JIT only works when the driver supports at least
  the CUDA version the plugin was built with.
- **Faster single-GPU builds:** set `-DNSS_CUDA_ARCHITECTURES=120-real`, for
  example.

`-DNSS_BUILD_CPU=OFF` configures only the CUDA plugin, without Highway or the
CPU kernels and tests.

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNSS_ENABLE_CUDA=ON
cmake --build build-cuda -j && ctest --test-dir build-cuda -L cuda
```

Fresh x86 builds also enable `NSS_AVX2_DEFAULTS=ON`, with configuration-scoped
AVX2 ports validated on C4 and Ryzen 5950X. `NSS_AVX2_EXPERIMENT=0` adds no
experimental candidates; use `-DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT=0`
to retain the AVX2 reference routes.

AArch64 builds use `-DNSS_HWY_TARGET_MODE=neon` (also the ARM
`dynamic` target policy), with `NSS_AVX2_DEFAULTS=OFF` and experiment 0.
SVE/SVE2 are excluded until their separate port is validated. The
`portable-test` mode selects Highway EMU128 for diagnostic comparisons;
it requires a supported compiler (use Clang) and is not a release mode.
Build macOS arm64 separately and load `libnss.dylib`.
`core.nss.Backend()` reports compiled/runtime targets and an executed dispatch
probe; this does not certify acceleration of every kernel.
For native ARM build/host verification, run `tests/arm_native_gate.py --help`.
It defaults to GCC 15 for NEON, requires fresh build/result paths and a native
R75 runtime, and retains source/toolchain/backend identities and failure logs.
The extended TWSC/NLH mathematical gates require NumPy and SciPy in the test
Python environment; these are test dependencies, not plugin dependencies.
The gate includes guarded tail/stride fixtures and a 762-case public shape and
1/2/4-thread matrix. On Linux GCC, add `--sanitize` for ASan/UBSan; standalone
C++ tests also check leaks, while the external Python/VS runtime has leak
detection disabled. Pixel captures support separate cross-build comparisons.

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
Cached-worst remains an alternative (`=2`); it cannot be combined with SortedTopK. Rolling remains explicitly requested through
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

The length-12/16/32/64 DCT kernels in `src/cpu/bm/dct_codelet_*.hpp` (and their copies
in `src/cuda/bm3d/`, compiled as scalar device code) are generated
by [FFTW](https://www.fftw.org/) genfft (`gen_r2r`) and are distributed under
GPLv2 or later. Copyright (c) 1997-1999, 2003, 2007-14 Massachusetts Institute
of Technology and Matteo Frigo. They are mapped onto Highway and are not linked
against `libfftw3`. Regenerate with `tools/gen_dct_codelets.sh`.

### NEON numerical compatibility

Cross-compiler/EMU outputs use the project's conditional numerical-tolerance policy; they are not guaranteed byte-identical. DCT/SSD near ties, iterative references and LSSC dictionary training can amplify ordinary rounding. FP64 LSSC OMP fixes a near-null residual artifact but has a separately reported migration cost against the older FP32 implementation. The default mixed-precision search retains FP64 refinement, solve and residual updates, with the full FP64 search as its numerical fallback.
