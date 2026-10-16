# Building

## Release packages

The release packages need no build. Each holds one plugin; copy it into the
VapourSynth plugin directory.

| Package | Plugin | Runs on |
|---|---|---|
| `linux-x86_64` | `libnss.so` | x86-64 with AVX2; glibc 2.34 and the libstdc++ of GCC 13 or newer: Ubuntu 24.04, Debian 13, Fedora 39 or later |
| `linux-arm64` | `libnss.so` | glibc 2.38 or newer (libstdc++ is built in): Ubuntu 24.04, Debian 13, Fedora 39 or later |
| `macos-arm64` | `libnss.dylib` | Apple silicon, macOS 13.3 or newer |
| `windows-x86_64` | `libnss.dll` | Windows x86-64 with AVX2 and the Microsoft Visual C++ 2015–2022 Redistributable (x64) |
| `linux-x86_64-cuda` | `libnss_cuda.so` | as `linux-x86_64`, with an NVIDIA driver for CUDA 12.9 or newer |
| `windows-x86_64-cuda` | `libnss_cuda.dll` | an NVIDIA driver for CUDA 12.9 or newer and the Visual C++ Redistributable as above |

On an older Linux (Ubuntu 22.04, Debian 12) the plugin fails to load with a
`GLIBCXX` version error; build it from source there.

## Requirements

- CMake 3.24 or newer and a C++20 compiler (GCC, Clang; on Windows, Clang
  targeting MSVC).
- VapourSynth API4 headers. With a pip-installed VapourSynth, pass
  `-DVapourSynth_INCLUDE_DIR=<site-packages>/vapoursynth/include`.
- [Highway](https://github.com/google/highway) 1.4.0 is fetched at configure
  time.
- For the CUDA plugin: nvcc from CUDA 12.4 or newer (Windows: nvcc with MSVC
  `cl.exe`).

## CPU plugin

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Copy `libnss.so` (`libnss.dylib`, `libnss.dll`) into the VapourSynth plugin
directory. `core.nss.Backend()` reports the compiled and running SIMD targets.

- **x86-64** needs AVX2; AVX-512 is used where available.
- **AArch64 / Apple silicon** builds use NEON (`-DNSS_HWY_TARGET_MODE=neon`,
  also what `dynamic` selects there) with `-DNSS_AVX2_DEFAULTS=OFF`. SVE is not
  used. Build macOS arm64 on its own and load `libnss.dylib`.

## CUDA plugin

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNSS_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda -L cuda
```

This also builds `libnss_cuda` (namespace `nss_cuda`). `-DNSS_BUILD_CPU=OFF`
builds only the CUDA plugin, without Highway or the CPU kernels.

- **Architectures:** native code for sm_75, sm_86, sm_89 and sm_120, plus
  compute_75 PTX. Other GPUs of compute capability 7.5 or newer run that PTX
  through the driver JIT (untested, best effort; the driver must support the
  CUDA version the plugin was built with). Toolkits older than 12.8 cannot
  target sm_120, so RTX 50 GPUs then use the JIT too.
- **One GPU only:** `-DNSS_CUDA_ARCHITECTURES=120-real` (for example) builds
  much faster.

## Options

| Option | Default | Meaning |
|---|---|---|
| `NSS_ENABLE_CUDA` | OFF | Build the CUDA plugin. |
| `NSS_BUILD_CPU` | ON | Build the CPU plugin and its tests. |
| `NSS_CUDA_ARCHITECTURES` | see above | CUDA architectures to compile for. |
| `NSS_HWY_TARGET_MODE` | `dynamic` | `avx2`, `avx3`, `neon` pin one target; `portable-test` (Clang only) is a diagnostic build, not a release mode. |
| `NSS_AVX2_DEFAULTS` | ON on x86 | The validated AVX2 kernel ports. OFF keeps the reference routes. |
| `NSS_AVX2_EXPERIMENT` | 0 | Bitmask of experimental AVX2 candidates (none by default). |
| `NSS_BM_EXPERIMENT` | 2305 | BM3D kernel selection bitmask (AVX-512 sorted top-k for groups of 16 and more, patch reuse, rolling aggregation). 0 is the pure-correctness reference. An existing CMake cache keeps its old value. |
| `NSS_BM_FUSED_TEMPLATE` | ON | BM3D's fused group filter on 256-bit and wider targets. OFF restores the previous paths (outputs agree to rounding). |
| `NSS_TWSC_MIDGROUP_BATCH` | ON on x86 | Batched FP64 TWSC solver for groups of 24 and 32. |
| `NSS_ENABLE_LSSC_SME` | OFF | Apple SME matrix leaf for LSSC on supported Apple silicon. |
| `NSS_GEMM_MULTI_ACCUM` | OFF | Experimental float-product / FP64-accumulation GEMM. |
| `NSS_ENABLE_SVD_LIBRARY` | OFF | Experimental TWSC SVD through an LP64 LAPACKE `DGESDD` provider; builds only its test target, not used by the filter yet. |

The reference build for numerical comparisons is
`-DNSS_BM_EXPERIMENT=0 -DNSS_AVX2_DEFAULTS=OFF -DNSS_AVX2_EXPERIMENT=0`.

## Numerical reproducibility

- A given build reproduces its output bit for bit.
- Builds for different instruction sets or compilers are not byte-identical:
  near ties in block matching, iterative solvers and LSSC's dictionary training
  can amplify ordinary rounding. They agree within the tolerances in
  `contracts/tolerances.json`.
- The CUDA plugin agrees with the CPU plugin to 60 dB PSNR or better and is
  identical from run to run on a given GPU, driver and build.

## Verification tools

- `tests/arm_native_gate.py --help`: native ARM build and host gate (GCC 15,
  NEON; `--sanitize` adds ASan / UBSan on Linux GCC).
- `tests/source_manifest.py`: records the commit, dirty state, CMake identity
  and attached artifacts of a build:

  ```sh
  python3 tests/source_manifest.py --out /tmp/nss-manifest.json --build <build-dir> \
    --baseline-label <baseline> --candidate-label <candidate>
  ```

- Python checks need NumPy and SciPy (test dependencies only).
