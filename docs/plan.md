# vapoursynth-nssfactory Development Plan

Status: the CPU comprehensive optimization campaign has seven algorithms that have passed the formal C4 gate; only BM3D has not yet reached `1.25x`, so the campaign as a whole remains unpassed.
This document is the top-down construction blueprint: goals → constraints → architecture → modules → interfaces → phases. All implementation follows this document; to change a decision, change this document first.

## Current Status (2026-09-01)

- The original directory `/Users/owen/Documents/vapoursynth-nssfactory` remains at baseline commit `223003e`, unmodified.
- The active checkout `/Users/owen/Documents/vapoursynth-nssfactory-work` has historical baseline `310adb0`; CPU optimization changes remain in the uncommitted working tree, and as requested no commits were created this round.
- The active checkout builds Release, VapourSynth host, and ASan/UBSan configurations successfully; all three CTest suites pass at **14/14 passed**. After removing the temporary NLM benchmark probes and doing a clean Release rebuild, still **14/14 passed**. ASan uses `halt_on_error=1`/UBSan stacktrace checks; the macOS-unsupported leak detector is not enabled. The ARM host can load `libnss.dylib` and list all entry points, but does not run filter paths requiring AVX2.
- The remote development regression host `owen@192.168.50.3` (5950X) passes Release CTest at **14/14 passed**. NLM, BM3D Basic→Final, WNNM `radius=1`, TWSC, MCWNNM, NCSR, NLH, and LSSC each completed 1/2-thread frame hash smoke tests as independent rebuild nodes; these results are development regressions, not formal performance gates.
- A `c4-highcpu-2` Spot instance `nss-c4-topdown-20260831125510` with `STANDARD` PMU was manually created on `dsmvc-avx512-pmu/us-central1-a`, and the eight CPU algorithms' 1080p Topdown L1/L2/L3 and `perf record` were completed over SSH. Hardware counters were non-zero, all eight `perf.data` files parse, and Lost Samples are 0; this manual flow did not depend on the Policy Troubleshooter API. The instance was kept temporarily for the immediate paired A/B; after all experiments it must be manually deleted and confirmed nonexistent.
- Formal paired C4 has confirmed WNNM batched SVD median ≈ **1.2067x** (threshold 1.15x), NCSR ≈ **1.3075x** (threshold 1.15x), NLM 15-pair median **1.334901x** (range 1.288994x–1.342539x, threshold 1.30x); NLM 15-pair output hashes all identical.
- TWSC's Highway U/S-only SVD passed formal C4: 15-pair median **1.400585688480x**, lowest single pair across 6 valid configurations **1.021764648674x**, PSNR vs baseline `146.541 dB`. MCWNNM's Gram spectral-shrink passed formal C4: 15-pair median **1.577800007075x**, lowest single pair across 12 valid configurations **0.992922490995x**, PSNR `150.049 dB`, repeat bit-exact.
- NLH's Highway q4/n16 Wiener, vector-only low 4-lane Haar, q4 fixed top-3 PixelMatch and masked tail passed formal C4: 7-pair median **1.402357948x** (range 1.401808564x–1.403821488x, threshold 1.30x). Max absolute error vs baseline `7.15e-7`, PSNR `142.58 dB`; candidate repeats are element-wise identical; worst ratio across the 24-config legal block/group/q matrix was `0.994732259x`.
- LSSC's AVX-512 Highway 2-vector×8-column `GemmNN` and fused ISTA update/group-soft/check passed formal C4: 7-pair median **1.331358620x** (range 1.326825634x–1.340783378x, threshold 1.15x). All 2,073,600 output floats are bit-identical to baseline; the worst ratio across all 7 legal block/step combinations was `1.166268191x`, and outputs are float-for-float identical.
- Withdrawn performance experiments: candidate-lane matcher regressed across the board; TWSC using the same batched SVD reached only ~**1.094x**; MCWNNM SVD prefiltering ~**0.824x**; NLM horizontal prefix and full-vector exp underflow branch both ~**0.99x**. These failed implementations are not in the current candidate set.
- BM3D's current formal paired speedup on real images is **1.109129869x**, below the `1.25x` target. Based on 9 real 1080p images, 18,360 reference blocks, and 4,024,251 candidates, the FMA/reduction-tree matching model confirms: the strongest pure first4 implementable schedule saves only **15.0301%** of SSD row work; even counting the tail-DC (which must read the tail and is not yet proven float-exact) as zero-cost, the center-priority schedule yields only **18.1883%**, below the **19.1205%** needed to pass the gate. The first4 exact-pruning route is closed; evidence in `artifacts/c4/bm3d-pruning-20260901/bm-first4-v7-summary.txt`.

### CPU Comprehensive Optimization Status Matrix

| Checkpoint | Current Status | Evidence / Remaining Boundary |
|---|---|---|
| W0 Correctness baseline & observation protocol | Partially complete | Corrected SVD, fixed seed/JSON v2 added; no independent checkpoint commit yet |
| W1 Matcher & batch core | Complete (implementation level) | Scalar differential, stable sort, finite-value fallback, and bounded ordered commit passed; current wrapper is still bucketed-per-item kernel invocation, not cross-lane SIMD |
| W2 BM3D main pipeline | Partially complete | Basic/Final, radius 0/1, batch host integrated, Basic→Final plugin smoke/hash passed; formal non-default matrix and C4 `>=1.25x` not yet achieved |
| W3 WNNM/SVD | C4 gate passed | Batched host, SVD quality tests, `radius=1` smoke/hash passed; formal paired C4 median ≈ `1.2067x`, above `1.15x` |
| W4 NLM lane | C4 gate passed | Fused Highway Welsch, fixed-s horizontal reduction, zero-weight skip, compact temporal/base-row accumulation retained; 15-pair median `1.334901x`, hashes all identical |
| W5 TWSC/MCWNNM/NCSR/NLH | C4 gate passed | TWSC `1.400585688480x`, MCWNNM `1.577800007075x`, NCSR ≈ `1.3075x`, NLH `1.402357948x`; each legal-config-matrix lowest pair no lower than `0.98x` |
| W6 LSSC prepared pipeline | C4 gate passed | Prepared context, prefix members, workspace-only temporaries, 2×8 GemmNN and fused ISTA done; formal median `1.331358620x`, worst of 7 valid combos `1.166268191x`, output float-identical |
| W7 Formal C4 campaign & closeout | In progress | Eight-algorithm Topdown complete; WNNM, TWSC, MCWNNM, NCSR, NLM, NLH, LSSC passed; only BM3D at `1.109129869x` has not reached `1.25x` |

This round's batch integration order is frozen as **BM3D → WNNM → TWSC → MCWNNM → NCSR → NLH → LSSC**. All seven non-BM3D algorithms have passed their C4 gates; the CPU comprehensive optimization cannot be declared complete until BM3D passes. CUDA/Vulkan and the GPU phases do not count toward this round's definition of done.

---

## 0. In One Sentence

A VapourSynth **API4** plugin factory: NSS-type denoisers sharing host / search / aggregation protocols, **C++20 + Highway** on CPU, and **mutually independent CUDA and Vulkan** pipelines on GPU. At runtime **pure CPU xor pure CUDA xor pure Vulkan**, with no cross-device algorithm round-trips and no CUDA↔Vulkan interop.

First platform target: **Linux x86-64, minimum AVX2**.

---

## 1. Locked Decisions

| # | Decision | Rejected Alternative | Rationale |
|---|---|---|---|
| D1 | VS **API4** (`VapourSynthPluginInit2` / `VSHelper4.h`) | API3 | New plugins carry no legacy ABI burden |
| D2 | CPU SIMD = **Google Highway**, dynamic dispatch | ISPC, VCL, hand-written AVX2-only, Halide | Same compiler as C++; Zen4 gets `AVX3_ZEN4`; no second compiler |
| D3 | Device-closed at runtime: search→filter→aggregate all on the same device within a frame | CPU block search / GPU SVD hybrids | PCIe + small buffers + thread sync would eat small-matrix gains |
| D4 | Three artifacts: `libnss.so`, optional `libnss_cuda.so`, optional `libnss_vk.so` | Single `.so` + `backend=`; CUDA↔Vulkan interop | Machines without the driver must not be dragged down; users pick device by namespace |
| D5 | WNNM SVD = **in-house 64×8 tall-skinny** (QR + 8×8 Jacobi) | Hard dependency on MKL/AOCL | At default `block_size=8, group_size=8`, vendor LAPACK call overhead exceeds computation cost |
| D6 | Internal working format **f32** | Mixed u8/u16 in kernel | Consistent with BM3D/WNNM references; integer input boundaries handled at edges |
| D7 | VS threads = CPU parallelism; MKL/OpenBLAS/Highway inner loops don't spawn threads | Filter-internal OpenMP | Avoid oversubscription with `fmParallel` |
| D8 | Temporal aggregation is a **separate stage** `VAggregate`. On the GPU side this stage must be an **independent device kernel**, with intermediates (num/den stacks) kept on device, only the final frame DTOH'd back. `BM3Dv2` is just syntactic sugar for `BM3D`+`VAggregate` | Merging aggregation into the collaborative filter kernel; GPU BM3D emitting fat intermediates then **CPU** `VAggregate` | Merging into the BM kernel breaks occupancy/memory access; CPU aggregation requires moving `(2R+1)×2` planes — exactly the path that BM3DCUDA's GPU `VAggregate` speedup eliminates. The two-stage API also allows passing aggregated basic to Final's `ref` |
| D9 | CPU/GPU **not guaranteed bitwise identical** | Cross-device gold standard | Matches BM3DCUDA status quo; compare by PSNR/residual |
| D10 | LSSC / NCSR / NLH **not in v1**. MCWNNM / TWSC are wave 2 (§14), not started in parallel with NLM/BM3D/WNNM | Six algorithms concurrently | No stable VS reference; first establish the factory with NLM/BM3D/WNNM |
| D11 | GPU: **CUDA first**, then **Vulkan**; each GPU tree is internally closed | Vulkan first, both GPUs in parallel, SYCL/Halide unification | CUDA has nlm-cuda / BM3DCUDA as reference; Vulkan is a port, not an exploration |
| D12 | First GPU round still consumes **CPU VSFrame** (HTOD→kernel→DTOH) | VS R80 GPU VideoNode immediately | Core GPU path not stable yet; get the compute right first |

Reference implementations (for behavioral comparison only, no toolchain copying):

- [vs-nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc) — NLM algorithm and parameters
- [VapourSynth-WNNM](https://github.com/WolframRhodium/VapourSynth-WNNM) — BM + WNNM + V-BM3D search semantics
- [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) — GPU closed pipeline, CPU AVX2 BM3D, parameter names

---

## 2. Goals and Non-Goals

### 2.1 Goals

1. Users call `core.nss.NLM` / `nss.BM3D` / `nss.BM3Dv2` / `nss.WNNM` for CPU denoising.
2. Optional `core.nss_cuda.*` / `core.nss_vk.*` for GPU denoising, with parameters matching CPU, and data does not return to host for collaborative filtering.
3. Block matching, patch packing, and weighted writeback written once within the CPU tree, with three algorithms plugging in different filters.
4. Highway runtime selection of AVX2 / AVX3 / AVX3_ZEN4 from one source tree.
5. Quality comparable with reference plugins at agreed parameters (see §8).

### 2.2 Non-Goals (explicitly not in v1)

- Windows / macOS / ARM distribution
- ISPC, Halide, SYCL, HIP, Metal, Mojo
- CPU+GPU within a frame, or CUDA+Vulkan
- Runtime backend switching
- GPU WNNM (later: device-side batched small SVD, one implementation per backend)
- Abstracting CUDA and Vulkan into a single kernel
- NLH / LSSC / NCSR / MCWNNM / TWSC (not in v1; wave 2, see §14)
- Bitwise-identical output with reference plugins
- Variable resolution / variable format clips
- Color space conversion inside kernels (RGB↔OPP left to scripts or a later `nss.RGB2OPP`)

---

## 3. Layered Architecture (Top to Bottom)

```
Scripts / vs-jetpack
        │
        ▼
┌───────────────────────────────────────┐
│  VS API4 host                         │  Device bound at creation; getFrame only dispatches
│  Format checks, requestFrame, workspace  │
└───────────────┬───────────────────────┘
                │
     ┌──────────┼──────────────────┐
     ▼          ▼                  ▼
┌──────────┐ ┌────────────┐ ┌────────────┐
│ CpuPipe  │ │ CudaPipe   │ │ VulkanPipe │  Three compile-time trees; no cross-calls
│ libnss   │ │ nss_cuda   │ │ nss_vk     │
└────┬─────┘ └─────┬──────┘ └─────┬──────┘
     ▼             ▼              ▼
 Highway        CUDA kernels   Vulkan compute shaders
 + tiny SVD   (all-device)      (all-device)
```

Rules:

- **Language boundaries may align with CPU address space** (Highway ↔ tiny SVD ↔ host); **may not cross PCIe, and may not cross CUDA context ↔ VkDevice**.
- CUDA and Vulkan **do not share kernel source**. Only shared artifacts: `params` defaults, distance definitions, test vectors, documentation.
- NLM and BM-type algorithms **do not share inner kernels** (whole-frame distance map vs. per-reference 8×8×K groups).
- GPU implementation order for the same algorithm: CUDA first (references + mature tooling), then port the **validated algorithm** to Vulkan — never explore both sides simultaneously.

---

## 4. Repository Layout

The first version lays out the full directory tree as a compilable "empty factory"; algorithm kernels are filled in by phase.

```text
vapoursynth-nssfactory/
  CMakeLists.txt
  cmake/
    FindVapourSynth.cmake
    Highway.cmake                 # FetchContent, pinned version
  docs/
    plan.md                       # this document
  include/nss/
    version.hpp
    params.hpp                    # default parameters, distance enum; includable from CPU and GPU
    plane.hpp                     # non-owning view: ptr, width, height, stride
  src/
    host/
      plugin.cpp                  # VapourSynthPluginInit2
      validate.cpp                # clip/format/parameter validation
      workspace.cpp               # thread_id → aligned buffer
      filter_nlm.cpp
      filter_bm3d.cpp
      filter_wnnm.cpp
    cpu/
      dispatch.hpp                # HWY_EXPORT / HWY_DYNAMIC_DISPATCH wrapper
      nlm/
        nlm.cpp                   # Highway target functions
      bm/
        match.cpp                 # window + predictive search + partial_sort (scalar C++)
        ssd.cpp                   # Highway SSD
        pack.cpp                  # im2col / centering
        dct8.cpp                  # 8×8 DCT (Highway)
        haar.cpp                  # reserved
        agg.cpp                   # weighted col2im (Highway)
      wnnm/
        svd_tiny.cpp              # QR + Jacobi, no LAPACK
        shrink.cpp                # weighted singular values + reconstruction
    cuda/                         # NSS_ENABLE_CUDA; filled in Phase 5
      plugin.cpp
      nlm.cu
      bm3d.cu
    vulkan/                       # NSS_ENABLE_VULKAN; filled in Phase 6
      plugin.cpp
      shaders/                  # GLSL, compiled to SPIR-V at build time
        nlm.comp
        bm3d.comp
      nlm.cpp
      bm3d.cpp
  tests/
    CMakeLists.txt
    ref/                          # no large videos committed; generator scripts and hashes only
    test_svd.cpp
    test_nlm_cpu.cpp
    test_bm3d_cpu.cpp
    vs/                           # vspipe scripts, optional
  .github/workflows/linux.yml
```

Plugin identifiers:

| | CPU | CUDA | Vulkan |
|---|---|---|---|
| id | `com.nssfactory.nss` | `com.nssfactory.nss_cuda` | `com.nssfactory.nss_vk` |
| namespace | `nss` | `nss_cuda` | `nss_vk` |
| library | `libnss.so` | `libnss_cuda.so` | `libnss_vk.so` |

---

## 5. External Filters (Script Layer)

Parameter names align with reference plugins to ease migration from vs-jetpack and existing scripts. V1 requires **constant format, 32-bit float**; planes processed independently (BM3D `chroma` not in v1).

### 5.1 `nss.NLM`

Aligned with `nlm_ispc.NLMeans`:

```text
clip:vnode; d:int:opt; a:int:opt; s:int:opt; h:float:opt;
channels:data:opt; wmode:int:opt; wref:float:opt; rclip:vnode:opt;
```

Defaults: `d=1, a=2, s=4, h=1.2, channels="AUTO", wmode=0, wref=1.0`.

### 5.2 `nss.BM3D` / `nss.BM3Dv2`

Aligned with the common BM3DCUDA subset:

```text
clip:vnode; ref:vnode:opt; sigma:float[]:opt;
block_step:int[]:opt; bm_range:int[]:opt; radius:int:opt;
ps_num:int[]:opt; ps_range:int[]:opt;
```

- Without `ref` → Basic (hard threshold)
- With `ref` → Final (Wiener)
- When `radius>0`, `BM3D`/`WNNM` output **unaggregated intermediates** (same fat-frame layout as current bm3dcuda: num/den per temporal offset), then chain into this plugin's `nss.VAggregate` / `nss_cuda.VAggregate` / `nss_vk.VAggregate`
- `BM3Dv2` is just creation-time sugar chaining the two steps above; **internally it is still two kernels** (collaborative filtering → aggregation), not aggregation written into the BM kernel
- Does **not** call HOVE `bm3d.VAggregate`; the GPU path does **not** pull intermediates back to CPU for aggregation
- `group_size` in v1 is **fixed at 8** (same as bm3dcpu, DCT-friendly)
- `block_size` in v1 is **fixed at 8**

### 5.3 `nss.VAggregate`

```text
clip:vnode; src:vnode; radius:int:opt; planes:int[]:opt;
```

`clip` is the unaggregated intermediate, `src` is the original (copies attributes, unprocessed planes). Semantics match bm3dcuda's built-in `VAggregate` (padding copies; unlike HOVE's zero padding — documented).

CPU: plain plane reduction.
GPU: independent compute kernel, input intermediates on device, output final frame.

### 5.4 `nss.WNNM`

Aligned with WNNM:

```text
clip:vnode; sigma:float[]:opt; block_size:int:opt; block_step:int:opt;
group_size:int:opt; bm_range:int:opt; radius:int:opt;
ps_num:int:opt; ps_range:int:opt; residual:int:opt;
adaptive_aggregation:int:opt; rclip:vnode:opt;
```

Defaults identical to the WolframRhodium plugin (including the accelerated `block_size/step/group_size`).
When `block_size≠8`, the tiny SVD uses general Jacobi (still no LAPACK); the main tested path remains 8.

Temporal search semantics follow **WNNM / old V-BM3D** (not BM3DCUDA's accelerated search that may hit duplicates). CPU BM3D and WNNM **share this search**.

---

## 6. Internal C++ Boundaries

### 6.1 Views and Parameters (device-free)

```cpp
// include/nss/plane.hpp
struct PlaneView {
    const float* ptr;
    float*       mut;      // only non-null for output planes
    int width, height, stride;
};

enum class Distance { SSD /* v1 does only this */ };

struct SearchConfig {
    int block = 8;
    int step  = 8;
    int group = 8;
    int bm_range = 7;
    int radius = 0;
    int ps_num = 2;
    int ps_range = 4;
};
```

`params.hpp` holds only defaults and validation ranges, no VS types.

### 6.2 CPU Pipeline (same process, same address space)

```text
host getFrame
  → fill PlaneView (including temporal pointer array, length 2R+1)
  → CpuNlm::run            or
  → CpuBm::match  → Filter → CpuBm::aggregate
```

- `CpuBm::match`: C++ control flow (window, predictive search, `partial_sort`) + Highway SSD.
- `Bm3dHard` / `Bm3dWiener` / `WnnmShrink`: take packed group matrices only.
- NLM does **not** go through `CpuBm`.

Highway conventions:

- Kernels written inside `HWY_NAMESPACE`, exported with `HWY_ONCE` at end of file.
- External C++ wrappers expose only non-template names like `void nlm_distance_luma_f32(...)`.
- Dynamic dispatch targets: **AVX2 required**; AVX3 and AVX3_ZEN4 enabled per CPUID. No SSE4 fallback (v1 minimum is AVX2; `mapSetError` at load if CPUID insufficient).
- `HWY_DISABLED_TARGETS` disables anything below AVX2.

Tiny SVD conventions:

- Input column-major, lda aligned to 8.
- Outputs thin U (64×k), S (k), Vt (k×8), k ≤ min(m,n).
- On failure (NaN / non-convergence) returns an error code; host skips collaborative filtering for that group and increments a counter (debug frame properties optional).

### 6.3 GPU Pipelines (one each for CUDA and Vulkan)

`src/cuda/**` and `src/vulkan/**` must not `#include` `src/cpu/**`, and they may not include each other.

Shared conventions (implemented per backend; don't abstract a common GPU base class prematurely):

- Device buffers allocated at creation; `getFrame`: HTOD → kernels → DTOH **final frame**.
- Parallelism: `num_streams` default 2, pool attached to the filter instance, **not** `thread_id → unlimited streams/command buffers`.
- Temporal: collaborative filter writes num/den to **device intermediate buffers** (no DTOH for aggregation); an **independent** `VAggregate` kernel reduces on device, only the final `H×W` result returns to host.
- `BM3Dv2` launches both kernels in one `getFrame`, intermediates not exposed as VS frames. The fat-frame DTOH is unavoidable for the two-stage script API (`BM3D` + `VAggregate`) when intermediates must surface as VS frames — on GPU the **recommended interface is the sugar `BM3Dv2`**; the two-stage API is left for CPU and debugging.
- First round does not use VS GPU VideoNode.

CUDA specifics: `.cu`, streams, pinned host buffers; reference nlm-cuda / BM3DCUDA.
Vulkan specifics: `glslangValidator`/`glslc` → SPIR-V at build time; compute queue; descriptor sets set up at creation; validation layers only in Debug.

CMake currently builds the CPU plugin by default; CUDA/Vulkan remain future phases not yet exposed as config switches. `params.hpp` is shared by all three sides from Phase 0.

---

## 7. Host Behavior (shared by all filters)

1. `fmParallel`.
2. `VSFilterDependency`: `radius==0` uses `rpStrictSpatial` (if the API allows) otherwise `rpGeneral`; `radius>0` uses `rpGeneral` over `[n-R, n+R]`.
3. Workspace: `unordered_map<thread::id, Buffer>` + `shared_mutex`, 64-byte aligned. Freed at filter destruction.
4. Only process planes where `sigma[plane]!=0` (or the NLM channels mask); others `copyFrame`.
5. Output frame properties copied from input clip.
6. `rclip`/`ref`: same resolution, format, frame count; matching is done on ref, filtering on clip (BM3D Final's Wiener reads both).

---

## 8. Quality and Testing

Layers:

| Layer | Content | When |
|---|---|---|
| Unit | tiny SVD vs Eigen/LAPACK residual; 8×8 DCT round-trip; SSD vs naive loop | Phase 0–1 |
| Kernel | Synthetic AWGN small images (64×64 / 128×128) PSNR | On each algorithm landing |
| Plugin | `vspipe` vs reference plugin, fixed-seed noise | Phase 2 onward |
| Regression | No GPU-vs-CPU bitwise; PSNR floor recorded | CI |

Comparison targets:

| This filter | Reference | Tolerance |
|---|---|---|
| `nss.NLM` | `nlm_ispc.NLMeans` same params | Closest; Highway vs ISPC ULP differences allowed, PSNR target &lt; 0.05 dB (synthetic) |
| `nss.BM3D` | `bm3dcpu.BM3D` same params, `radius=0` | Implementation differences allowed; baseline PSNR recorded |
| `nss.WNNM` | `wnnm.WNNM` same params, `radius=0`, `residual=0` | SVD path differs, PSNR target &lt; 0.1 dB |
| `nss_cuda.NLM` | This repo's `nss.NLM` | PSNR only, no bitwise |
| `nss_vk.NLM` | This repo's `nss.NLM` (fallback: `nss_cuda.NLM`) | Same |
| `nss_cuda` / `nss_vk` BM3D | This repo's `nss.BM3D` | Same |

CI (Linux x86): build + unit tests. `vspipe` comparisons gated behind `NSS_REF_TEST=ON` (manual/nightly); machines without reference plugins not forced.

---

## 9. Build

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Dependencies:

- CMake ≥ 3.24
- C++20 compiler (gcc 12+ / clang 16+)
- VapourSynth headers (pkg-config `vapoursynth`)
- Highway: FetchContent pinned to a **git tag** (in `cmake/Highway.cmake`; upgrades in separate PRs)
- CUDA tree: CUDA toolkit (Phase 5)
- Vulkan tree: Vulkan headers + loader + `glslc`/`glslangValidator` (Phase 6)
- No ISPC, no MKL, no Eigen runtime dependency (Eigen optional for tests only)

Compiler flags: `-O3 -ffast-math` only on Highway kernel TUs; host validation code does not use fast-math.

Install: `libnss.so` → vapoursynth plugin directory (pkg-config `libdir/vapoursynth`).

---

## 10. Phase Plan (Construction Order)

Principle: at the end of each phase, the repo compiles and the corresponding filter or test runs; no empty stubs count as done.
**Overall order: CPU gold standard → CUDA locks in the GPU algorithm → Vulkan is a portable translation.**

### Phase 0 — Empty factory loads

**Goal:** `vspipe` can call `core.nss.Version()` or an identity filter, proving API4 + CMake + Highway linking works.

- CMake, FindVapourSynth, Fetch Highway
- `plugin.cpp`: `configPlugin` + `Version`
- `workspace` skeleton
- CPUID gate: no AVX2 → error on filter load
- CI: configure + build

**Done when:** `libnss.so` is autoloaded, `core.nss.Version()` returns git describe.

**Not doing:** any denoising.

### Phase 1 — `nss.NLM` CPU

**Goal:** single/multi-plane f32 NLM, algorithm per vs-nlm-ispc (distance map → horizontal/vertical box filter → weight accumulation → finish).

- Highway implementation of distance / box / accum / finish
- `wmode`: only 0 (Welsch) initially; others later
- `rclip`, `d/a/s/h`, `channels`
- Unit tests + small-image comparison vs nlm-ispc (if available)

**Done when:** 1080p `d=1,a=2,s=4` runs; synthetic noise PSNR comparable to reference.

### Phase 2 — Shared BM + `nss.BM3D` (radius=0)

**Goal:** search and aggregation become reusable CPU modules; BM3D Basic/Final spatial mode works.

- `match.cpp`: spatial window SSD, no-threshold group inclusion (consistent with WNNM/BM3DCUDA spatial strategy)
- `dct8.cpp` + hard / Wiener
- `agg.cpp` weighted writeback
- `ref` behavior per BM3DCUDA docs
- `group_size=8, block_size=8` hardcoded

**Done when:** recorded PSNR baseline vs `bm3dcpu` at same params, `radius=0`; no crashes, no NaN.

### Phase 3 — Temporal BM3D + WNNM

**Goal:** the same `CpuBm::match` feeds WNNM's tiny SVD; `radius>0` goes through the separate `nss.VAggregate`.

- predictive search (WNNM semantics)
- `svd_tiny.cpp` + weighted nuclear norm + adaptive aggregation
- `residual` switch
- `nss.VAggregate` (CPU) + `nss.BM3Dv2` sugar (internally invokes two steps)
- SVD unit tests vs naive implementation

**Done when:** `nss.WNNM` PSNR within §8 threshold vs reference plugin on synthetic images; `radius>0` uses this plugin's `VAggregate`, no HOVE dependency.

### Phase 4 — CPU Hardening (short)

Cleanup only; no new algorithms here.

- NLM `wmode` remaining modes (if drop-in nlm-ispc compatibility desired)
- README usage examples
- Record CPU performance orders of magnitude (not a gate)

**Done when:** three CPU filters usable in scripts; Phase 5 has a comparison baseline.

### Phase 5 — CUDA NLM

**Goal:** first closed GPU pipeline. Algorithm locked in Phase 1; only the executor changes.

- `libnss_cuda.so` + `nss_cuda.NLM`
- pinned + `num_streams=2`, reference nlm-cuda
- PSNR comparison against `nss.NLM`

**Not doing:** BM3D, Vulkan, GPU WNNM.

### Phase 6 — CUDA BM3D + CUDA VAggregate

**Goal:** main GPU NSS path. Collaborative filtering and aggregation are **two device kernels**; performance comes from "intermediates never return to host," not from fusing steps into one kernel.

- `nss_cuda.BM3D`: `radius=0` outputs final frame directly; `radius>0` outputs device intermediates (sugar path doesn't expose fat VS frames)
- `nss_cuda.VAggregate`: independent aggregation kernel (the main speedup vs CPU `VAggregate`)
- `nss_cuda.BM3Dv2`: launch BM3D kernel → VAggregate kernel → DTOH final frame in the same `getFrame`
- Compare with `nss.BM3D` + `nss.VAggregate`

**Not doing:** CUDA WNNM; no GPU intermediates → CPU `VAggregate`.

### Phase 7 — Vulkan NLM

**Goal:** translate the Phase 5-verified NLM GPU algorithm to compute shaders, proving the Vulkan host (queues, descriptors, SPIR-V build) can run NSS.

- `libnss_vk.so` + `nss_vk.NLM`
- Build-time shader compilation
- Compare with `nss.NLM` (optionally cross-check against `nss_cuda.NLM`)

**Not doing:** redesigning NLM on Vulkan; no VS GPU VideoNode.

### Phase 8 — Vulkan BM3D + VAggregate

**Goal:** port Phase 6: independent BM kernel + independent aggregation kernel, not a second exploration.

- `nss_vk.BM3D` / `nss_vk.VAggregate` / `nss_vk.BM3Dv2` sugar
- Compare with `nss.BM3D` + `nss.VAggregate`

### Later (not in current plan)

- CUDA / Vulkan WNNM (batched small SVD, each backend its own)
- VS GPU VideoNode (R80+) bound to `nss_vk` (still never mixed with CUDA)
- Second-wave NSS: MCWNNM, TWSC (priority, sharing WNNM's block search/SVD), then NLH, LSSC, NCSR (§14)
- `chroma` CBM3D, integer formats, RGB2OPP, Windows wheel
- NLM `wmode≠0` (if not done in Phase 4)

---

## 11. PR Breakdown Within Phases

Each PR must compile. Dependencies by number.

| PR | Title | Phase | Depends On |
|---|---|---|---|
| P0.1 | Repo skeleton: CMake, directories, `docs/plan.md` | 0 | — |
| P0.2 | API4 plugin entry + `nss.Version` + AVX2 gate | 0 | P0.1 |
| P0.3 | Highway FetchContent + empty `HWY_DYNAMIC_DISPATCH` smoke test | 0 | P0.2 |
| P1.1 | NLM host parameter validation & workspace | 1 | P0.3 |
| P1.2 | NLM Highway kernel (luma f32) | 1 | P1.1 |
| P1.3 | NLM multi-plane / channels / rclip | 1 | P1.2 |
| P1.4 | NLM tests & reference comparison script | 1 | P1.3 |
| P2.1 | `SearchConfig` + spatial block match + SSD | 2 | P0.3 |
| P2.2 | 8×8 DCT + hard threshold + aggregate | 2 | P2.1 |
| P2.3 | `nss.BM3D` host (Basic, radius=0) | 2 | P2.2 |
| P2.4 | Wiener Final (`ref`) | 2 | P2.3 |
| P3.1 | Tiny SVD library + unit tests | 3 | P0.3 |
| P3.2 | WNNM shrink wired into CpuBm | 3 | P2.1, P3.1 |
| P3.3 | `nss.WNNM` host | 3 | P3.2 |
| P3.4 | `nss.VAggregate` + predictive search; `BM3Dv2` sugar | 3 | P2.4, P3.3 |
| P4.* | CPU README and wmode | 4 | Phase 3 |
| P5.1 | CUDA plugin skeleton + stream pool + empty Version | 5 | Phase 3 stable |
| P5.2 | `nss_cuda.NLM` | 5 | P5.1, Phase 1 |
| P6.1 | `nss_cuda.BM3D` spatial | 6 | P5.2, Phase 2 |
| P6.2 | `nss_cuda.VAggregate` independent kernel | 6 | P6.1 |
| P6.3 | `nss_cuda.BM3Dv2` sugar: two kernels + final-frame DTOH | 6 | P6.2, Phase 3 |
| P7.1 | Vulkan plugin skeleton + SPIR-V build | 7 | P5.2 proving GPU NLM works |
| P7.2 | `nss_vk.NLM` | 7 | P7.1 |
| P8.1 | `nss_vk.BM3D` / `BM3Dv2` | 8 | P7.2, P6.2 |

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Highway DCT scaling inconsistent with bm3dcpu's FFTW | Round-trip unit test; sigma mapping may need calibration, documented |
| In-house SVD truncation differs from MKL `sgesdd`, degrading perceived WNNM quality | Residual unit tests; write Jacobi iteration cap and ε as constants, compare against reference |
| NLM box filter boundary differs from ispc clamp | Function-by-function comparison against vs-nlm-ispc Horizontal/Vertical |
| Temporal search differs from BM3DCUDA and users call it a bug | Document: follows WNNM/V-BM3D, not the CUDA accelerated search |
| Temptation to fuse VAggregate into the BM3D kernel "to save one launch" | Keep kernels separate; BM3Dv2 only chains at the host level. bm3dcuda's GPU aggregation speedup comes from less DTOH, not fewer launches |
| FetchContent fails on offline CI | Cache a Highway tarball or use a submodule (decide by P0.3) |
| Premature common CUDA/Vulkan GPU abstraction layer | Wait until NLM lands on both backends; never delay the first GPU for "unification" |
| No reference implementation for Vulkan BM3D | Only port CUDA-validated algorithms; no Vulkan BM3D before Phase 7 |
| Temptation of VS R80 GPU frames | Phases 5–8 use CPU frames in/out; native GPU node is a separate later item |

---

## 13. Current Next Step (Historical Execution Pointer)

The following P0.1/P0.2 were the original execution pointers in the blueprint; the corresponding CMake skeleton, API4 entry point, and `nss.Version` already exist. For current CPU closeout work, see the remaining execution plan at the end of this section.
The Highway smoke test is P0.3, immediately following, to avoid getting mired in target macros on day one.

Open items that do **not block P0** (update this document when needed):

- Which upstream Highway tag to pin
- Whether `nss.Version` is a standalone filter or just part of the plugin description string
- Whether the plugin id should switch to a personal/organization domain (currently placeholder `com.nssfactory.nss`)

---

## 14. Unimplemented Algorithms and Function Reuse (Wave 2)

V1 CPU gold standard is done: `nss.NLM` (`wmode=0` only), `nss.BM3D`/`BM3Dv2`/`VAggregate`, `nss.WNNM`. This section is only a queue — it does not change D3/D5/D7/D8, nor unfreeze `block_size=8`.

References (behavioral comparison only, no Matlab vendoring):

- [MCWNNM, ICCV 2017](https://arxiv.org/abs/1705.09912) / [csjunxu/MCWNNM-ICCV2017](https://github.com/csjunxu/MCWNNM-ICCV2017)
- [TWSC, ECCV 2018](https://arxiv.org/abs/1807.04364) / [csjunxu/TWSC-ECCV2018](https://github.com/csjunxu/TWSC-ECCV2018)
- NLH (Hou/Xu 2019, lifting Haar), LSSC (Mairal 2009), NCSR (Dong 2013) still lack stable VS references; scheduled after MCWNNM/TWSC

### 14.1 One-Line Summaries

| Algorithm | Input | Collaborative Filter | Position Relative to This Factory |
|---|---|---|---|
| **MCWNNM** | Joint RGB (YUV444/RGBS) | Concatenate patches into `3p²×n`, row weights `W=min(σ)/σ_c`, ADMM; Z step = ClosedWNNM | Multi-channel WNNM + weighted data term; ADMM needed since no closed form |
| **TWSC** | Grayscale or RGB concat | Three weights: channel/row noise `W1`, column noise `W2`, sparse weight `Wsc`. Group SVD as dictionary, soft threshold (`λ1=0`) or ADMM | Same block search; filtering swaps nuclear norm for sparse coding |
| **NLH** | Grayscale (RGB later) | Block matching → pixel-level NSS → lifting Haar + double hard, then Wiener | BM skeleton + new Haar/pixel matching; `haar.cpp` still reserved, unbuilt |
| **LSSC** | Grayscale | Clustering (not window top-k) + joint sparsity + overcomplete dictionary | Different search semantics; new dictionary-learning kernel |
| **NCSR** | Grayscale | Window BM + sparse coding noise (centered on nonlocal means) + iterative regularization | Search reusable; sparse coding close to TWSC |
| NLM `wmode≠0` | Whole-frame distance map | Existing distance/box/accum; swap vertical weight function | Does not go through CpuBm |
| `chroma` CBM3D | YUV444 | Match on Y, UV reuse coordinates | Matching exists; pack needs multi-plane |
| CUDA/Vulkan | Validated algorithms | Entirely on device; no CPU search + GPU SVD | No CPU kernel reuse (D3) |

MCWNNM model: `min_X ‖W(Y−X)‖_F² + ‖X‖_{w,*}`. ADMM splits into diagonal weighted X step + `svd_economy` + ClosedWNNM. ClosedWNNM is exactly the singular-value update in the existing `wnnm_shrink`: `σ̂ = (s + sqrt(s² − C))/2` (`ClosedWNNM.m`).

TWSC hot path (`λ1=0`, AWGN demo): subtract mean → `svd_economy` gives `D` → `S = sqrt(max(s² − n σ_col², 0))` → `C = soft(DᵀY, σ_col² / S)` → `X = DC` → aggregate with `W2`. ADMM only when `λ1≠0`.

### 14.2 Existing Functions (Direct Reuse)

| Existing | File | MCWNNM | TWSC | NLH | LSSC | NCSR | Notes |
|---|---|---|---|---|---|---|---|
| `spatial_match` / `predictive_match` / `ssd_block` | `match.cpp` `ssd.cpp` | ✓ | ✓ | ✓ block-level | ✗ clustering | ✓ | Window top-k SSD; LSSC needs separate cluster |
| `pack_patch` / `unpack_patch` | `pack.cpp` | single-channel rows | single-channel rows | ✓ | ✓ | ✓ | RGB concat: see 14.3 `pack_patch_nch` |
| `aggregate_add` / `aggregate_finish` | `agg.cpp` | ✓ per channel | ✓ | ✓ | ✓ | ✓ | With column weight `W2`, `w` is per patch |
| `svd_economy` | `svd_tiny.cpp` | ✓ per ADMM step | ✓ once per group | ✗ | dictionary step | dictionary step | `kSvdMaxM=256` suffices for `3×8²=192`; `block=16` RGB needs 768 or rejection |
| `wnnm_shrink` / ClosedWNNM | `shrink.cpp` | extract SV kernel for Z step | ✗ | ✗ | ✗ | ✗ | Extract `wnnm_sv_shrink(S, C)` for ADMM reuse; do not fork the formula |
| `residual` mean removal | `shrink.cpp` | common | required | optional | optional | optional | Promote to `group_center` |
| `vaggregate_reduce` | `agg.cpp` | `radius>0` | same | same | same | same | Temporal protocol unchanged |
| `dct_1d` / `dct_lines` | `dct8.cpp` | ✗ | ✗ | alternative | ✗ | ✗ | NLH's main path is Haar, not DCT |
| NLM distance/box/accum | `nlm.cpp` | ✗ | ✗ | ✗ | ✗ | ✗ | Whole-frame distance map, not in BM groups |
| host: `fmParallel`, workspace, validate, fat layout | `src/host/*` | ✓ | ✓ | ✓ | ✓ | ✓ | New filters only swap the Filter |

`kSvdMaxN=32` is insufficient for the papers' default `nlsp≈70`. Wave 2 either keeps the 32 upper bound on `group_size` (factory contract) or raises N separately; do not change the V1 WNNM limit for a Matlab default.

### 14.3 Shared New Functions for Unimplemented Parts (Write Once)

MCWNNM and TWSC share **the same author, same block search, same concatenation, same ADMM shell**. Abstract these first, then write each filter:

| New Kernel | Consumers | Behavior |
|---|---|---|
| `pack_patch_nch` / `unpack_patch_nch` | MCWNNM, TWSC, CBM3D | Pack multi-plane patches into `nch·p²` columns; matching on Y-only or RGB concat distance |
| `channel_weight_diag` | MCWNNM, TWSC | `W_c = min(σ)/σ_c` (MCWNNM `NSig`; TWSC row weights share the source) |
| `group_center` | MCWNNM, TWSC, WNNM residual | Row-mean add/subtract |
| `wnnm_sv_shrink` | WNNM, MCWNNM ClosedWNNM | Extract the SV update from current `wnnm_shrink` |
| `admm_weighted_x` | MCWNNM, TWSC `λ1≠0` | `X = (W²Y + ρ/2 (Z − A/ρ)) / (W² + ρ/2)` diagonal step |
| `soft_threshold` | TWSC, NCSR, LSSC | `sign(B)·max(\|B\|−τ, 0)` |
| `iter_regularize` | TWSC, NCSR, MCWNNM outer loop | `x += δ(y − x)` |
| `noise_estimate_pca` | MCWNNM/TWSC real noise | Liu PCA (`NoiseEstimation.m`); v1 may accept user `sigma[]` only |
| `haar_2d` / `haar_group` | NLH (BM3D haar reserved) | Lifting Haar; parallel to DCT, does not replace `dct8` |
| `pixel_match` | NLH only | In-group row/pixel matching; existing `spatial_match` insufficient |

Do NOT abstract:

- K-SVD / overcomplete dictionaries: only LSSC/NCSR, no VS reference
- Pixel-level NSS: only NLH
- NLM `wmode` weight functions: stays in `nlm.cpp`
- GPU kernels: D3 — CPU block search does not feed device SVD

### 14.4 Suggested Landing Order (Still "Later")

1. Extract `wnnm_sv_shrink` + `group_center` (WNNM behavior unchanged)
2. `pack_patch_nch` + optional concat-distance match
3. `nss.MCWNNM`: host copied from WNNM, Filter = ADMM(`admm_weighted_x` + SVD + `wnnm_sv_shrink`); RGB/YUV444; three-channel `sigma`
4. `nss.TWSC`: same host/pack; `λ1=0` uses `soft_threshold` first; ADMM later
5. NLH: fill `haar.cpp` + `pixel_match` (barely overlaps 1–4's new kernels; can be parallel files with 3)
6. NCSR: wait for TWSC's `soft_threshold` / in-group PCA dictionary to stabilize
7. LSSC: last. Clustering + overcomplete dictionary + joint sparsity (ℓ_{1,2}); does not reuse window top-k

GPU: wave-2 algorithms are CPU-only by default (same as current WNNM). CUDA/Vulkan still only follow the validated NLM/BM3D.

### 14.5 NLH / LSSC / NCSR: Do Not Treat as One Big Concurrent Round

All three are NSS, but their **Filter kernels have almost zero overlap**. What can be shared is already in the V1 BM host; the unwritten new kernels are per-algorithm. Spinning up three agents to write filters simultaneously will collide on `match.cpp` / `pack.cpp` / `cpu_api.hpp` / `plugin.cpp`, and produce three copies of "my own sparse/transform."

Pipelines (only steps relevant to this factory):

```text
NLH:  spatial_match → pack → pixel_match(rows) → lifting Haar → double hard
       → agg → (stage2) Haar(noisy, basic) → Wiener
NCSR: spatial_match → pack → group_center → PCA/dictionary → per-column sparse
       → code-domain nonlocal means → shrink coding noise → iter_regularize → agg
LSSC: extract all patches → cluster (Si=Sj) → overcomplete D (K-SVD/ODL)
       → group ℓ_{1,2} joint sparsity (OMP/LARS, not soft threshold) → agg
```

| Kernel | NLH | NCSR | LSSC | Exists? |
|---|---|---|---|---|
| `spatial_match` / pack / agg / host | ✓ | ✓ | pack/agg/host | Yes. LSSC does **not** use window top-k |
| `pixel_match` + `haar_*` | ✓ | ✗ | ✗ | No. NLH only |
| `soft_threshold` + in-group SVD as D | ✗ | ✓ (close to TWSC) | ✗ | No. Wait for TWSC |
| `iter_regularize` | outer loop K≈2, different semantics | ✓ | optional | No. One line; used by NCSR/TWSC |
| cluster + overcomplete D + OMP/ℓ_{1,2} | ✗ | PCA dictionary, not K-SVD | ✓ | No. Only LSSC is worth writing new |
| BM3D `dct_lines` / `wnnm_shrink` | Haar not DCT | not nuclear norm | not nuclear norm | Not reused in these three Filters |

Therefore:

- **Cannot** "do NLH+LSSC+NCSR together." Shared design has already converged (CpuBm host); no large shared implementation blocks remain.
- **NCSR depends on TWSC's sparse primitives** — do not write two sets of `soft_threshold` / PCA-D in parallel with TWSC.
- **NLH is file-independent** (new `haar.cpp`, `pixel_match`, `filter_nlh.cpp`); can be parallel files with MCWNNM/TWSC, but do **not** change the top-k contract of `spatial_match`.
- **LSSC goes last, alone.** No VS reference; SPAMS/OMP/dictionary learning is bigger than the other two, and clustering semantics conflict with the existing matcher.

### 14.6 New Algorithm TODO: DA3D / BM4D (2026-09-11)

The following is a new implementation plan, all pending; interface names are drafts. First establish CPU reference paths, then wire into the plugin and optimize. Suggested order: DA3D → BM4D, each accepted independently.

#### DA3D: Final-Stage Denoising Driven by a Guide Image

Reference: [Pierazzo / Facciolo, IPOL 2017, paper and reference source](https://www.ipol.im/pub/art/2017/203/). DA3D takes the original noisy image and an existing denoising result, performing frequency-domain shrinkage on shape- and data-adaptive patches, as a post-processor on top of existing denoisers.

- [ ] **Reference and contract:** Pin the IPOL paper, source version and hash; verify the source license; itemize patch selection, adaptive weights, frequency shrinkage, aggregation and boundary rules; create a re-runnable reference output.
- [ ] **Draft interface:** Plan `nss.DA3D(clip, ref, sigma, planes)`; `clip` is the original noisy image, `ref` is a required denoised guide; define size/format/frame-count matching, sigma units, zero sigma, and unprocessed plane behavior. V1 processes frame-by-frame; define grayscale and multi-plane support boundaries.
- [ ] **CPU kernel:** First implement a comparable scalar path — guide-driven adaptive patches, dynamic sampling, frequency shrinkage, inverse transform, weighted normalization; verify the paper's large-patch (64×64) path. Transform type and normalization follow the reference; do not directly substitute the existing BM3D DCT.
- [ ] **Plugin integration:** Add an independent DA3D kernel and host entry; reuse format conversion, plane selection, bounded workspace; connect to `ref` produced by BM3D/NCSR etc.; verify dual-input frame dependencies, concurrency determinism, boundary coverage, and zero-aggregation-weight handling.
- [ ] **Quality acceptance:** With fixed noisy/ref/sigma, compare against IPOL, recording per-pixel error and PSNR/SSIM; compare guide vs DA3D output across flat gradients, textures, edges, tiny images, and various noise levels; keep regression samples; do not assume all guides benefit.
- [ ] **Performance & delivery:** Optimize Highway/transform and scratch only after correctness; paired on the same machine, record DA3D incremental time, full "guider → DA3D" time, and peak memory; complete parameter docs, script examples, and regression tests before marking done.

#### BM4D: 3D Block Grouping and 4D Collaborative Filtering

Reference: [Maggioni et al., TIP 2013, Nonlocal Transform-Domain Filter for Volumetric Data Denoising and Reconstruction](https://doi.org/10.1109/TIP.2012.2210725); [authors' software and paper directory](https://webpages.tuni.fi/foi/GCF-BM3D/index.html). BM4D stacks similar 3D volume blocks into 4D groups; mapping video frames to volumes is this project's adaptation and cannot be called V-BM4D with motion-trajectory construction.

- [ ] **Version and scope:** Freeze the BM4D paper/reference version and parameters being reproduced, distinguishing the classic algorithm from later correlated-noise extensions; start with real-valued AWGN — Rician, complex reconstruction, and correlated noise are scheduled separately. Check the authors' software use/redistribution license; define boundaries for reference comparison and independent implementation.
- [ ] **Volume data and interface contract:** Plan `nss.BM4D`, defining VS frame-sequence → `(x, y, z)` volume mapping, volume-block depth, search range, group size, and Basic/Final guide inputs; freeze rules for head/tail frames, short clips, scene changes, and plane handling. Public signatures registered only after these contracts are set.
- [ ] **CPU baseline:** Implement 3D volume-block SSD search, stable grouping, volume packing, 4D separable transform, Basic hard-threshold, Final Wiener, and overlap-weighted reconstruction; verify transform scale, noise variance, and aggregation weights stage by stage. Reuse only confirmed-applicable primitives from the 2D matcher/pack; do not treat BM3D's temporal search as BM4D.
- [ ] **VS scheduling and memory:** Derive the complete frame dependency for Basic→Final and all contributing blocks; design bounded volume workspace, caching, and deterministic reduction; evaluate whether the existing `VAggregate` protocol can express volume-block contributions — do not directly apply the 2D patch layout. Verify random frame access, multithreading, window seams, and memory ceilings.
- [ ] **Correctness and quality acceptance:** First compare two-stage outputs on fixed synthetic 3D volumes and the authors' reference, then test static, moving, occluded, and scene-change videos; record numerical error, PSNR/SSIM, temporal consistency, and ghosting. List whole-volume reference vs. bounded video window differences as an engineering adaptation, not a paper-reproduction pass.
- [ ] **Optimization and delivery:** After the baseline passes, evaluate Highway 3D distance, transform batching, and volume-block reuse; report paired Basic→Final runtime, peak memory, and quality trade-offs per block/search configuration. Complete format/parameter regressions, docs, and examples; CUDA/Vulkan scheduled independently later.

Multi-agent only after **shared kernels are merged and files don't overlap**:

| Stage | Who | Writes | Forbidden |
|---|---|---|---|
| S0 one person | Abstract interfaces / implement existing reuse points | `group_center` if not yet extracted; `cpu_api.hpp` declarations | Three filter hosts |
| S1 at most two lanes | A: NLH (haar + pixel_match + filter_nlh) B: MCWNNM or TWSC | Own new files + own `filter_*.cpp` | The other's kernels; `match.cpp` control flow |
| S2 | NCSR | `filter_ncsr.cpp`, calling TWSC's landed soft/PCA | New dictionary solvers |
| S3 | LSSC | `cluster.cpp` `omp.cpp` `filter_lssc.cpp` | Turning `spatial_match` into clustering |

Forbidden: three filter agents simultaneously editing `plugin.cpp` / `params.hpp` / `match.cpp`; or abstracting a shared sparse framework first then filling three houses. Land one (NLH or TWSC sparse) first, the second calls it, and LSSC decides later whether to share OMP.

## 15. CPU Comprehensive Optimization — Remaining Execution Plan

This section is the execution baseline after 2026-08-31, covering only CPU comprehensive optimization. The original CUDA/Vulkan phases remain historical planning and do not affect this round's definition of done. External parameters, defaults, output layouts, iteration counts, and per-group CPU APIs must remain unchanged; batch is only an internal capability.

### 15.1 Workstreams and Dependencies

| Workstream | Depends On | Current Status | Completion Gate |
|---|---|---|---|
| W0 Correctness & observation | corrected SVD | Partially complete | Independent corrected-baseline checkpoint; JSON pins revision/compiler/CPU/params/size/seed/thread/warmup/wall-time fields |
| W1 matcher/batch | W0 | Implementation complete | Scalar-vs-batch differential; stable tie/NaN; short tail and bounded ordered commit; cannot drop groups |
| W2 BM3D | W1 | Partially complete | Basic/Final, ref, radius 0/1, fat intermediate, non-default parameters; C4 at least `1.25x` |
| W3 WNNM/SVD | W1 | Partially complete | Reconstruction/orthogonal/singular-value thresholds, failed-group behavior, reference PSNR; C4 at least `1.15x` |
| W4 NLM | W0 | Partially complete | Gray/UV/YUV/RGB, `d/a/s` boundaries, stride/subsampling; C4 1080p Gray `d=1` at least `1.30x` |
| W5 TWSC → MCWNNM → NCSR → NLH | W1, each prior | Partially complete | Each filter independent checkpoint, scalar differential, plugin smoke, 1/2-thread hash; TWSC/MCWNNM/NCSR `>=1.15x`, NLH `>=1.30x` |
| W6 LSSC prepared | W0, W1 | Partially complete | Frame-local D/Dᵀ/Lipschitz, prefix members, workspace-only cluster/dict/OMP, finite/PSNR; C4 `>=1.15x` |
| W7 Formal campaign | W2–W6 | Incomplete | C4 runner generates complete artifacts, all corresponding gates pass, docs matrix matches artifacts |

W5's batch integration order is fixed as **TWSC → MCWNNM → NCSR → NLH**; the global group order is **BM3D → WNNM → TWSC → MCWNNM → NCSR → NLH → LSSC**. Any lane that regresses total workload by more than 2% or exceeds quality thresholds stops integration and returns to the nearest correctness checkpoint.

### 15.2 Implementation Constraints and Stop Rules

- The matcher's sort key is distance, self-first, time, coordinates, original candidate ordinal; finite distances take priority over NaN/Inf. The `block=8/group=8` fast path must fall back to the stable generic path for non-finite input.
- The batch wrapper may bucket by `(m,k,channels,algorithm,basic,residual)`, but the current "bucket-then-call-kernel-per-item" approach does not count as cross-group lane SIMD; without workload-level evidence, do not claim lane speedups.
- NLM keeps stripe-local/halo with per-thread ≤1 MiB scratch; mixed plane strides must use the stride-aware path — do not pass plane 1's stride to all planes. If a new lane fails the C4 gate, keep the correct version and withdraw the experimental lane; do not declare the CPU comprehensive plan complete until NLM passes.
- The LSSC prepared context lives only within the current frame and does not cache input-dependent dictionaries; hot-path temporaries for cluster, dictionary, OMP, and reconstruct must come from the caller's workspace with no implicit heap allocations.
- PMU is diagnostic only. Only actual non-zero hardware counts and successful `perf record` on the guest count as evidence; `perf list`, CPU docs, or single top-down percentages are not merge conditions.

### 15.3 Formal C4 Runner

Fixed entry point:

```text
tests/run_c4_cpu_gate.sh \
  --project <project> \
  --zone <zone> \
  --vs-venv-tar <path>
```

The runner requires the caller to explicitly supply the unique corrected baseline via `NSS_C4_BASELINE_REF` or `NSS_C4_BASELINE_DIR`; no longer defaults to a `HEAD` that may contain erroneous SVD. The automated runner may still do fail-closed checks for zone, quota, machine type, and unique instance name, but the Policy Troubleshooter API is not a precondition for manual SSH campaigns: when the user has confirmed create/delete permissions, directly create a uniquely-named instance, run over SSH, pull artifacts, then manually delete and confirm absence. Use `COPYFILE_DISABLE=1` when packaging on macOS to prevent AppleDouble `._*.cpp` files being globbed as source by CMake on the Linux guest.

The guest uses the pre-built VapourSynth venv tar passed in; it does not install or replace the runtime online. Baseline/candidate source and build configs are uploaded separately, run interleaved in independent processes; the test process is pinned to CPU 0, with metadata recorded for initialization, independent warmup, varying frame numbers, single/multi-threading, and three content shapes. WNNM, TWSC, MCWNNM, NCSR, NLM, NLH, and LSSC each have their own formal C4 artifacts and cannot substitute for one another; BM3D must independently reach `1.25x` and complete the legal configuration matrix to produce a passing campaign.

The exit `trap` deletes only the instance created this time and polls to confirm deletion; any failure in Spot admission, SSH, guest build, artifacts, or deletion reports the exact status without creating fallback resources. `tests/c4_guest_gate.sh` only handles guest build/JSON standard-library parsing and paired statistics; historical `310adb0` old text benchmark output is only parsed for compatibility — the final artifact uses the current schema.

### 15.4 2026-08-31 All-Algorithm Topdown Baseline

Uniform conditions: 1920x1080, fixed seed, single VapourSynth process, single thread pinned to CPU 0, sibling CPU 1 idle; guest is Xeon Platinum 8581C with 1 physical core / 2 SMT, 2 MiB L2. Each algorithm collected Topdown L1/L2/L3 and an independent `cycles:u perf record`. L3 sub-events ran only ~2.94% of the time, so they are directional diagnostics only, not performance gates.

| Algorithm | ms/frame | Backend | Core | Memory | `perf record` Main Hotspots |
|---|---:|---:|---:|---:|---|
| NLM | 228.0 | 63.6% | 45.0% | 19.5% | VerticalWelsch 40.5%, Horizontal 26.3% |
| BM3D | 82.5 | 35.5% | 26.0% | 10.3% | SpatialMatch8 58.2%, Bm3dFilter8 19.4% |
| WNNM | 247.4 | 38.1% | 25.3% | 12.9% | JacobiSvd8 27.7%, ApplyHouseholder 20.3%, matcher 19.7% |
| TWSC | 529.0 | 37.4% | 25.0% | 12.9% | Jacobi 25.5%, matcher 19.0%, Householder 18.8% |
| NCSR | 550.0 | 40.8% | 26.0% | 14.4% | Jacobi 24.7%, Householder 18.2%, matcher 17.9% |
| LSSC | 533.0 | 35.0% | 27.6% | 8.7% | GemmNN 44.9%, reconstruct 20.8%, SsdVec 12.0% |
| NLH | 9048.0 | 23.9% | 17.7% | 5.9% | SpatialMatch 31.5%, Haar 20.9%, SsdBlock 17.6%, PixelMatch 14.5% |
| MCWNNM | 5855.0 | 39.3% | 24.9% | 14.4% | Householder 29.8%, Jacobi 26.9%, DotN 6.9% |

Full artifacts are in `artifacts/c4/topdown-20260831-current/`: 24 Topdown files, 8 non-empty `perf.data`, 8 symbol reports; directory archive SHA-256 `3f48665408b40df41755b0a8c8a4b36c69cf04851ea4d8ea9c527f40b2a9d979`; guest source package SHA-256 `569980db0d746406b7ab09e1061dec5028f9d8712b016adad3c599fe0a952654`.

2026-09-01 rerun using the current optimized working tree with the same C4/1920x1080/seed/affinity/frame-count protocol: all 24 Topdown files, 8 non-empty `perf.data`, and 8 symbol reports complete, `perf record` all with zero lost samples. Relative to the corrected profile above, NLM/BM3D/WNNM/TWSC/NCSR/LSSC/NLH/MCWNNM were `1.2140x/1.1007x/1.4280x/1.3955x/1.3283x/1.3068x/1.3980x/1.5724x` respectively; relative to the original historical session's rounded runtimes, 7/8 were faster, with NLH still at `0.8243x`. The original VM's raw PMU files were lost with its deletion, so that column is explicitly marked historical-approximate; full boundaries, Topdown changes, and current hotspots are in `artifacts/c4/topdown-20260901-optimized/comparison.md`. Current artifact archive SHA-256: `23f232bc99024afd7c8510f82420e14830cba12ad5e4d36e4a01b39c09433b50`.

The first Highway experiment mapped matcher candidates to SIMD lanes; the plugin-level three-group A/B median ratios were BM3D `0.3856x`, WNNM `0.6446x`, TWSC `0.6601x`, NCSR `0.6703x`, NLH `0.9618x`. The experiment regressed across the board and was precisely withdrawn; `SpatialMatch8` is no longer a near-term rewrite target. The current highest shared-benefit direction is instead a true cross-group `n=8` SVD/Householder Highway batch, followed by NLM pass fusion, NLH group=16 matcher/Haar, and LSSC blocked GEMM.

The subsequent BM3D first4 v7 re-check used the same four FMA accumulators and reduction tree as `SpatialMatch8`, verifying the final ordinal-stable top-8 of all implementable schedules. The strongest zero-overhead pure-first4 baseline ceiling is `1.216935x`; even assuming the tail-DC's tail load/reduction, downward error protection, and branches are all zero-cost, the strongest implementable ceiling is still only `1.242308x`. Therefore no further investment in first4, seed warmup, center-order, or tail-projection prototypes; BM3D only evaluates structural approaches that significantly change full-matcher or `Bm3dFilter8` throughput.

The cross-reference AVX-512 prototype placed two adjacent raster references and their same-offset candidates into the two 8-lane halves of one ZMM, keeping the original FMA/reduction tree and independent stable top-8. It reduced total instructions from ~`13.05B` to `11.73B`, but IPC dropped from `2.51` to `2.24` and cycles rose from ~`5.20B` to `5.24B`; the 7-pair plugin median was only `0.993488750904x`, with output hashes all identical. The prototype was fully withdrawn, with no extension to similar four-reference/ZMM paths; evidence in `artifacts/c4/bm3d-refpair-20260901/summary.txt`.

`Bm3dFilter8`'s final group-IDCT→num/den direct aggregation was also tested in isolation: although it removed the 64-vector `G` writeback/re-read, row-major writes into the overlapping aggregation buffer dropped the 7-pair median to `0.940219322526x`. Versus the original path, max abs `1.79e-7`, PSNR `155.24 dB` — quality-compatible but a clear performance failure, fully withdrawn; evidence in `artifacts/c4/bm3d-invacc-20260901/summary.txt`.

The matcher top-k control structure's scalar max-heap was also pre-checked in isolation: OFF/ON C4 Release CTest both 14/14, but the 7-pair, 4,000-default-matcher paired median was only `0.734050493338x` (median `6.380710 ms → 8.712793 ms`). Scalar branches and heap entry moves are significantly slower than the current Highway mask/popcount/permutation insertion, so it was not promoted to a full plugin campaign and the code was fully withdrawn; evidence in `artifacts/c4/bm3d-heaptopk-20260901/summary.txt`.

### 15.5 Definition of Done

Only when W0–W7 implementation, correctness, quality, performance, and documentation evidence are all complete, with NLM C4 `>=1.30x` and every filter gate passed, will the CPU comprehensive optimization be marked complete. The remote `192.168.50.3` 5950X is only for quick development regression; it cannot replace the formal C4 gate, nor can local CTest or CI artifacts count as hardware performance evidence.

### 15.6 2026-09-01 C4 PMU Snapshot Baseline

Reusable dependencies on the C4 golden image machine were migrated into `/opt/nss-c4`; after removing `/tmp` experiment sources, build trees, performance data, caches, and history, a global snapshot `nss-c4-pmu-env-20260901` was created (project `dsmvc-avx512-pmu`, storage location `us`, 50 GB, status `READY`). The snapshot pins Ubuntu 24.04, kernel `6.17.0-1022-gcp`, GCC 13.3.0, CMake 3.28.3, perf 6.17.13, Python 3.12.3, NumPy 2.5.2, VapourSynth R75, and Highway 1.4.0 commit `2607d3b5b0113992fe84d3848859eae13b3b52c1`, and persists `perf_event_paranoid=-1`, `nmi_watchdog=0`.

The independent restore verification used a `c4-highcpu-2` Spot with `STANDARD` PMU: all 9 1920x1080 Gray8 real-image sample hashes passed, `cycles:u` was non-zero, `perf record` non-empty with zero lost samples; a network-isolated fresh Release build passed 14/14 CTest and completed a BM3D plugin smoke on `MAPPA.gray8`. The restore verification machine and its auto-delete boot disk, the original golden machine, and its original boot disk were all deleted; only the `READY` snapshot remains.

Subsequent C4 instances must be restored from this snapshot via `/Users/owen/.codex/skills/nss-c4-pmu-spot/`; falling back to public images on snapshot restore or Spot admission failure is not permitted. After each instance restore, `guest_verify.sh` must still run, and the current source must be packaged and uploaded separately; real-image campaigns must record sample basename/hash and must not mix with fixed-synthetic historical baselines as A/B. This infrastructure closure does not change §15.5's definition of done: BM3D currently at `1.1007x < 1.25x`, W7 still incomplete.

### 15.7 2026-09-01 DCT Codelet / NCSR FastExp Verification Status

This round was completed and passed formal C4 verification: `DctLines` for n=16/32/64 integrated genfft forward/inverse Highway codelets (8/16 lanes), with the n=2/4, n=8, and non-x86 fallback contracts unchanged; NCSR's two weight paths were unified into Highway `FastExp` dynamic dispatch. Local Apple ARM Release CTest and C4 candidate Release CTest both **14/14 passed**; on C4 the 16-lane codelet was confirmed to actually compile and execute.

The formal paired gate used only `03144bc` as this round's current boundary, no longer mixing `ed97d8b` (which enables different Highway targets) as a broad baseline. The historical NCSR SVD gate of ~`1.3075x` remains independently preserved in `artifacts/c4/svd-batch-20260831/ncsr-svd-paired.tsv`; it does not count toward this round's pass/fail. The first attempt was voided due to wrong baseline selection and a unified `120 dB` quality threshold unsuited to the ~`1e-7` DCT recombination error; it is kept in `artifacts/c4/dct-ncsr-round-20260901/attempt1/` as a failed audit only.

The final `nssfactory.c4.cpu-gate.v2` artifact is `artifacts/c4/dct-ncsr-round-20260901/final/attempt2/c4.json`:

- BM3D's 12 block/group configurations had a minimum of `1.015698x` vs `03144bc`, with the affected-config median `1.364205x`; default BM3D output is bit-exact.
- NCSR at 1920x1080 had a 7-pair median of `0.993227x`, meeting this round's incremental threshold of no more than 2% regression; the historical `1.3075x` broad gate is not recomputed or merged.
- LSSC block=8/16 minimum `1.005299x`; default LSSC output bit-exact.
- Candidate BM3D/NCSR repeat runs had consistent hashes. All approximate configurations were finite, with minimum PSNR `114.2066 dB` and maximum absolute error `7.29263e-5`, passing the unified `PSNR >= 110 dB && max_abs <= 1e-4` threshold.
- Candidate BM3D and NCSR `cycles:u perf record` were 100176 and 79332 bytes respectively, both reports showing `Total Lost Samples: 0`; the restored environment's `guest_verify.sh` passed again.

The final archive `artifacts/c4/dct-ncsr-round-20260901/final/dct-ncsr-round-20260901-final.tar.gz` has SHA-256 `242a8604a2d26a430ad4bffaf6c48676f8ab849104ab1f61cf7ebd52818ae225`; after download, the outer hash and all 11 internal file hashes re-verified. Spot instance `nss-c4-dct-ncsr-20260901-a` and its auto-delete boot disk were deleted and confirmed nonexistent.

Therefore the **DCT codelet / NCSR FastExp optimization round is complete**, but §15.5's overall CPU campaign definition of done is unchanged: this round's codelets do not touch the default BM3D n=8 hot path, default BM3D still has not reached `1.25x`, and W2/W7 and the CPU comprehensive plan cannot be marked complete.


## 2026-09-05 Optimization Retrospective and Maintenance Supplement

This section supplements the current status while preserving the earlier historical design. b4/b12/b16, the latest b16 output sink,
VAggregate integration, failed candidates, and cross-algorithm applicability boundaries are in the [optimization review](../PROJECT_CONSOLIDATED_TIMELINE_20260927.md)(原报告已并入综合时间线,2026-09-27).

This round retained the BM3D/NLH prepared commit, NLH bounded scratch reuse and invalid-parameter cleanup,
and removal of the unused b8 transpose function; the BM3D scratch-reuse experiment was not retained. Final C4 had 15/15 CTest
and 89 plugin comparison configurations passing; six seven-pair performance re-measurements were all within the 2% regression threshold, but r1 was about 1.58% slower,
which is a clear maintenance trade-off and must not be advertised as a speedup for all configurations. Evidence and resource cleanup status are in
[this round's report](../artifacts/c4/maintenance-20260905-a1/summary.md).

The next round should prioritize building a top-k replay of real distance streams and checking the b8 Wiener active interval;
the b12 output layout can borrow from the b16 sink, but must continue to keep independent TUs, reduction order, and full-chain gates.


## 2026-09-05 DCT Layout and b8/g8 Experiment Supplement

This section appends this round's status while preserving the earlier historical design and comprehensive CPU goals. Detailed evidence and candidate tables are in the
[layout experiment report](../PROJECT_CONSOLIDATED_TIMELINE_20260927.md)(原报告已并入综合时间线,2026-09-27); experiment entry points are in the
[kernel lab](bm-kernel-lab.md).

- b12 cross-patch nine-image real Basic→Final was 1.015621×, but the g2 Basic control item
  could not confirm meeting the 1% regression bound after fifteen pairs, so it was kept in the lab. The 8+4 sink did not pass full-chain screening.
- b8 generic dual-patch and row-transpose inlining both failed ordinary plugin b4/g1 comparisons; the dual-patch's
  single isolated integration revision also failed. Both b4 matrix layouts failed AVX2/AVX-512 microkernel comparisons.
- b8/g8 TopK tables and paired partial-sums gained no full-chain benefit; FFTW transpose inlining
  was 1.054446× over the nine-image full chain, but g4 Basic exceeded the 1% regression threshold, and the independent-TU revision did not resolve it.
- PMU now distinguishes Basic/Wiener, but this was not expanded into G/R/group-axis rewrites unsupported by evidence.
- A general lab, directional timing, real patch corpus, TopK-classified replay, guard pages, and stride
  checks were implemented; repeat A/B, numerical comparison, and PMU collection are centralized in the `tests/c4_*` tools.
  New experiments reuse the existing inline transpose, default-off; public parameters, production dispatch, and floating-point compile options remain unchanged.

This round integrated no new production speedup candidates. If b8 layout work continues, a floating-point-compatible boundary that isolates compiler codegen changes should be established first, and the frozen cross-block plugin comparison retained. The shared matcher
currently keeps its original implementation; multi-channel matching, SVD, Haar, and NLM are not expanded into independent rewrites.
Final maintenance regression, performance acceptance, and temporary resource cleanup are in the final acceptance section of the report above; this section does not mark
historical W2/W7 or the overall CPU comprehensive optimization plan as complete.

The maintenance control matrix's NCSR item could not confirm the 1% regression bound after fifteen pairs, so runtime transpose
deduplication was also withdrawn; the pre-frozen matrix was not shrunk to claim a pass. In the end, only the experiment/test tooling
integration and test coverage were delivered, ordinary production code restored to its original state, and closure was achieved with same-ISA plugin binary identity and re-regression
verification; see the layout report's final delivery section for details.

The final tooling version completed C4 verification: the three dispatch variants' ordinary plugins were byte-identical to their corresponding frozen baselines,
each with 15/15 CTest and 197 plugin comparisons passing; the lab passed 17/17 CTest, both ISA checks,
rolling and VAggregate regressions passed. The old b16/SSD lab's AVX2-unsupported boundary continues to be
explicitly reported as skipped, without fabricating microkernel timings for that path. Final evidence and resource cleanup records are in
`artifacts/c4/layout-20260905-a1/summary.md`.


## 2026-09-05 BM3D correctness and dual-region C4 campaign

B0 d572c52 freezes the current dirty runtime and lab. P0 4154cdf corrects fixed
reference matching, real frame bounds, target aggregation, zero-sigma identity,
rolling dependencies, and effective BM3D sigma. Both regional C4 hosts passed
16/16 CTest plus independent public sigma, time and concurrency oracles.
P1/P2 evaluation and final default-off verification are recorded in
bm3d-correctness-campaign.md. No optimization candidate passed the full gate.

## 2026-09-06 BM3D campaign completion

The final P runtime includes the later overflow and Wiener-alias repairs, plus
exact unit-weight VAggregate identity on AVX2. Dynamic, AVX2 and AVX3 plugins
are byte-identical to independently assembled pure-correctness references;
each passed 16/16 CTest and 230/230 plugin comparisons on YUL. TLV independently
verified dynamic with the same binary hash. Both lab builds passed 18/18 CTest,
and retained rolling and VAggregate regressions passed.

The user's revised budget targets about 40 seconds per complete A/B bench group,
including calibration and warm-up. Frame counts are calibrated instead of using
thousands of fixed requests. Fewer than seven pairs, incomplete random groups,
and earlier cached-only ring timings cannot pass the admission gate. Completed
historical evidence is preserved. Every optimization remains default-off.

Migration covered 1,080 synthetic-motion/real-image cases; memory, allocation,
4K, matching replay and PMU diagnostics are archived. Natural-video acceptance
is unverified. The earlier B0-to-F temporal correctness-cost diagnostic shows
roughly 30%-35% more runtime, and is not an optimization claim. Full outcomes,
source/artifact hashes, residual proof boundaries, and exact two-VM cleanup
records are linked from the campaign report. This does not complete or change
the historical W2/W7 or overall CPU performance targets.

## 2026-09-06 Bounded per-configuration selection

Under the user's revised >1.02x acceptance rule, SortedTopK, bounded BM3D reuse,
and rolling ring/direct-scratch extraction form the selected default mask 2305.
Cached-worst remains an accepted alternative for its qualifying configurations;
lazy raster and homogeneous dispatch did not reach 1.02x in the bounded matrix.
Three C4 Spot hosts ran independent lanes with approximately 30-60 second group
budgets. All candidate numerical matrices and the combined dynamic/AVX2 checks
passed. Measured configuration results, single-pair limitations, source hashes,
raw audits and cleanup records are in the 2026-09-06 bounded-selection report (maintained outside the source tree).