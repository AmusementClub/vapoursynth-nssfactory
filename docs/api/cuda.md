# nss_cuda.* API Reference

`libnss_cuda` is the CUDA plugin (namespace `nss_cuda`), built with
`-DNSS_ENABLE_CUDA=ON`. It has all nine filters of the CPU plugin. This page
holds everything that is specific to the device: the two extra arguments, the
temporal output, memory, numerics and measured speed. Parameters, defaults,
the papers and pitfalls are on each filter's own page and hold for both
plugins.

| Function | Filter page | Section here |
|---|---|---|
| `core.nss_cuda.BM3D` | [bm3d.md](bm3d.md) | [BM3D and VAggregate](#bm3d-and-vaggregate) |
| `core.nss_cuda.VAggregate` | [bm3d.md](bm3d.md) | [BM3D and VAggregate](#bm3d-and-vaggregate) |
| `core.nss_cuda.NLM` | [nlm.md](nlm.md) | [NLM](#nlm) |
| `core.nss_cuda.WNNM` | [wnnm.md](wnnm.md) | [WNNM](#wnnm) |
| `core.nss_cuda.MCWNNM` | [mcwnnm.md](mcwnnm.md) | [MCWNNM](#mcwnnm) |
| `core.nss_cuda.NCSR` | [ncsr.md](ncsr.md) | [NCSR](#ncsr) |
| `core.nss_cuda.NLH` | [nlh.md](nlh.md) | [NLH](#nlh) |
| `core.nss_cuda.TWSC` | [twsc.md](twsc.md) | [TWSC](#twsc) |
| `core.nss_cuda.LSSC` | [lssc.md](lssc.md) | [LSSC](#lssc) |
| `core.nss_cuda.Version()` | | |
| `core.nss_cuda.Backend()` | | reports the device, driver/runtime versions and the support level |

## Shared by every filter

Each filter takes the same arguments as its `nss` counterpart, plus
`device_id` and `num_streams` at the end of the argument list, and gives the
same errors under the `nss_cuda` name. The one difference in behaviour is the
temporal output (below). Outputs agree with the CPU plugin to 60 dB PSNR or
better (not bit-for-bit; the gate is in `tests/data/cuda_tolerances_v1.json`)
and are identical from run to run on a given GPU, driver and build.

### Arguments

| Parameter | Default | Meaning |
|---|---|---|
| `device_id` | 0 | CUDA device index. One filter instance uses one device. |
| `num_streams` | 1 | How many frames (or temporal chunks) an instance has in flight on the device at once, 1 to 16. Each stream owns its device buffers. |
| `memory_limit_mb` | none | Caps the instance's device and pinned host memory (see below). |

- **`num_streams`.** The host copies of several frames (or temporal chunks)
  run around one stream, which keeps the device busy for the fast filters:
  1080p BM3D on an RTX 5080 runs at about 790 fps with one stream and with
  three, and at `radius = 1` at about 310 fps with one and 350 with three.
- **`memory_limit_mb`** has no default in `nss_cuda`: without it the filter
  takes what its plan needs, and a device allocation that fails is reported
  as the CUDA out-of-memory error. With it, the value caps the instance's
  device and pinned host memory and the internal batches are fitted to it; a
  limit that cannot hold the streams is a creation error. Nothing degrades
  silently in either case.

### Temporal output

With `radius > 0` the device filters return finished, normal-height frames:
no `VAggregate` call is needed. (NLM is temporal through `d` and always
returns normal-height frames.)

| Parameter | Default | Meaning |
|---|---|---|
| `temporal_mode` | `"rolling"` | `"rolling"` returns finished frames; `"legacy"` returns the fat intermediate for `VAggregate`, exactly as the CPU plugin does. |
| `rolling_chunk` | 4 | Frames accumulated per rolling chunk, 1 to 64. |
| `rolling_cache_chunks` | 1 | Finished chunks kept for later frame requests at first, 1 to 64. Given alone, the cache stays at this size. |
| `rolling_cache_limit` | 16 | What the cache may grow to, 1 to 64 and at least `rolling_cache_chunks`. It keeps one more chunk each time two requests miss on chunks it dropped recently, so alternating between positions settles after a few misses. Under `memory_limit_mb` it grows only into what the limit leaves. (On the CPU the two arguments name one fixed size.) |

- BM3D, WNNM and single-round NCSR keep the temporal accumulation on the
  device and copy back only final frames. The other filters (and NCSR with
  more rounds) reduce the fat intermediate with `nss_cuda.VAggregate` inside
  the filter; the rolling arguments only matter where the accumulation stays
  on the device.
- `temporal_mode="legacy"` is for mixing backends or keeping an existing
  script unchanged: the fat intermediate is a plain VS frame with versioned
  properties, so either backend's `VAggregate` can reduce it.
- The finished-frame mode needs more memory than legacy.
- Sequential rendering, a chained second stage and a temporal filter
  downstream compute each chunk once. Fully random access costs a chunk per
  miss; `temporal_mode="legacy"` does not depend on the order.

### Driver and GPU support

- A prebuilt plugin needs a driver for the CUDA release it was built with
  (12.9 for the release packages) or newer.
- The default build has native code for sm_75, sm_86, sm_89 and sm_120, plus
  compute_75 PTX. Other sm_75+ GPUs run the PTX through the driver JIT, which
  is untested and best effort; `Backend()` reports which of the two a device
  gets. Building is described in the repository `README.md`.

## BM3D and VAggregate

```python
basic = core.nss_cuda.BM3D(clip, sigma=25)
final = core.nss_cuda.BM3D(clip, ref=basic, sigma=25)
# Temporal: finished frames by default (temporal_mode="rolling").
temporal = core.nss_cuda.BM3D(clip, sigma=25, radius=1)
# The CPU plugin's two-step form, for mixing backends or debugging.
fat = core.nss_cuda.BM3D(clip, sigma=25, radius=1, temporal_mode="legacy")
temporal = core.nss_cuda.VAggregate(fat, clip, radius=1)
```

- **`num_streams`.** One stream already keeps the device busy; more add
  little (about 15% for the rolling mode). `VAggregate` accepts `device_id`
  and `num_streams` but does not use a device.
- **Device-resident.** Matching (spatial or predictive temporal), collaborative
  filtering and aggregation all run on the device.
  - Spatial and rolling output copy only final frames back.
  - Legacy `radius > 0` copies back the fat intermediate, because it must
    exist as a VS frame.
- **Rolling.**
  - Each chunk of frames keeps a ring of the temporal window on the device.
  - A chunk that follows the one its stream ran last carries on from it: the
    window frames stay on the device and the sums that the earlier centers
    left for its frames are kept, so it runs `rolling_chunk` centers and
    uploads `rolling_chunk` frames instead of `rolling_chunk + 2R` and
    `rolling_chunk + 4R`. Any other chunk starts afresh. The output is the
    same either way.
  - A frame is finished as soon as its last center has run, so a plane keeps
    2R + 1 slices of sums whatever `rolling_chunk` is; the chunk's finished
    frames wait on the device and are downloaded together.
  - Under a `memory_limit_mb` without room for it, the planes of a clip take
    turns on one state (each chunk then starts afresh unless the clip has one
    plane) and finished frames are downloaded one by one.
  - Each center frame's contributions are added into the chunk's target
    frames: as exact fixed-point sums for the shapes that aggregate inside
    the filter kernel (below), in ascending order as the CPU does for the
    rest.
  - It agrees with `nss_cuda.VAggregate(nss_cuda.BM3D(..., radius=R,
    temporal_mode="legacy"))` to rounding (8 ulp measured) for the former
    shapes and bit for bit for the latter.
- **`nss_cuda.VAggregate` runs on the host.** Its inputs are already host
  frames, and uploading 2(2R+1) planes per frame costs several times more than
  the sum itself. It sums slices in the CPU's order and divides with IEEE
  rounding.
- **Interop.** `nss.VAggregate` and `nss_cuda.VAggregate` accept each other's
  BM3D output. They agree within a few ulp: the CPU's fast-math division may
  be up to 2 ulp off IEEE rounding.
- **`final=1`.** The basic estimate stays on the device: one upload and one
  download per plane instead of three and two. The legacy intermediate with
  `final=1` is the two calls chained.
- **Numerics.** The transform math is the CPU's 3D DCT, as butterflies: the
  8-point ones of the CPU's fused path and the same generated codelets for
  12, 16, 32 and 64.
  - Shapes whose group fits registers are filtered several groups per warp:
    block 4, 8 or 16 with a group of at least 2 and up to 128 samples per
    lane (group x block), plus 16 / 16, and 4 / 64, 8 / 32 and 16 / 32 for the
    hard-threshold stage. The
    others run one block per group, with the cube in shared memory when it
    fits 24 KB and in device memory otherwise.
  - The output is not bit-identical to the CPU, but stays within the 60 dB
    gate.
  - Spatial and rolling filtering of most shapes aggregates from inside the
    filter kernel with integer atomics on fixed-point sums (exact, so
    independent of the order); legacy temporal output and the remaining
    shapes sort their patches and sum them in a fixed order.
- **Memory.** With `memory_limit_mb`, the value also caps the rolling chunk
  cache, and the filter runs with smaller internal batches to fit (the output
  does not change). The smallest limit one stream accepts at 4K, for GRAYS /
  YUV420 / YUV444 or RGB:
  - Spatial: about 350 / 370 / 420, and 420 / 430 / 480 with `ref`.
  - Rolling, `radius = 1`: 1140 / 1300 / 1780, and 1490 / 1650 / 2130 with `ref`.
  - Rolling, `radius = 2`: 1590 / 1750 / 2220, and 2130 / 2280 / 2760 with `ref`.
  - Legacy with `ref`: 990 / 1090 / 1370 at `radius = 1`, 1630 / 1790 / 2260
    at `radius = 2`.
  - Rolling with `final=1`: 1810 / 1970 / 2440 at `radius = 1`, 2760 / 2920 /
    3390 at `radius = 2`.
- **Performance.** At 1080p GRAYS on an RTX 5080, one stream, against
  bm3dcuda with the same search (measured 2026-10-06):
  - Spatial: about 1.25x with 32 VapourSynth threads (the frame transfers
    bound it); the kernels take the same time.
  - Rolling, frames asked for in order: 2.7x at `radius = 1` and 3.3x at
    `radius = 2` with 32 threads, 1.4x with one thread. Random access:
    1.15x with one thread, 2.2x with 32. It keeps the CPU's predictive
    search and reproducible sums.
  - `final=1` against the two calls,
    GRAYS: 303 to 730 fps spatial, 297 to 458 at `radius = 1`, 293 to 347 at
    `radius = 2`; YUV420: 204 to 433 spatial, 192 to 297 at `radius = 1`;
    YUV444 with `chroma=1`: 95 to 241 spatial, 95 to 199 at `radius = 1`.
    Rolling reads 4R frames on each side of a chunk that starts afresh
    (2R for one stage) and keeps a source ring of 4R + 1 frames.
  - `chroma=1` at YUV444: 243 fps spatial (separate planes 286: the host
    copies bound both, and three planes per frame go less evenly), 238 fps
    at `radius = 1` (separate 222) and 229 at `radius = 2` (separate 179).
    bm3dcuda with `chroma=True`: 200, 94 and 65.

## NLM

All channel modes, temporal `d`, and `rclip`.

- **Device-resident.** The whole frame runs on the device: clamped distance
  maps, the (2s+1)² box sums, Welsch weights, the symmetric accumulation and
  the `wref` normalization.
- **One launch per frame.** A frame is one kernel launch: a block of threads
  owns a 32 x 32 tile of the output and keeps its sums in registers through
  every offset of every frame pair, so the frames are read from device memory
  once and only the result is written.
- **Window frames stay on the device.** A frame of the clip is uploaded once,
  however many windows it is part of, in any order of requests and with any
  number of streams. Threads stage their uploads and copy their results out
  while the stream works on other frames.
- **Memory.** By default the device holds `2d + 4` frames of the clip (with
  one stream). Under `memory_limit_mb` it holds fewer, down to one window. For
  `d` above 8, a patch or search window too large for a tile's shared memory
  (48 KiB: the defaults use 16), or a limit below one window, the filter falls
  back to one launch per offset with only the centre frame and the current
  backward/forward pair resident; the smallest accepted limit is never above
  what that takes.
- **Numerics.** The GPU uses the same model as the CPU, exact box sums and
  the CPU's fast exponential operation for operation.
  - The output is not bit-identical to the CPU (the CPU takes its sums over
    rows as running sums), but is far inside the 60 dB gate (above 144 dB
    measured).
  - The output is run-to-run identical: every pixel is accumulated by one
    thread in a fixed offset order.

## WNNM

- **Device-resident.** Matching, the per-group SVD shrinkage and the
  aggregation all run on the device.
- **Temporal output.** The temporal accumulation stays on the device.
- **Numerics.** The SVD is taken through the FP32 Gram matrix with a cyclic
  Jacobi eigensolver.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (86–143 dB on the frozen references).
  - The output is run-to-run identical: every group is solved in a fixed
    order, and aggregation is ordered.
  - Groups of up to 8 patches run one thread per group. Larger groups run one
    thread block per group with a round-robin parallel Jacobi.

## MCWNNM

- **Device-resident.** The joint three-channel matching, the ADMM solve of
  every group, the aggregation and all outer rounds run on the device.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Numerics.** Each ADMM step shrinks through the FP32 Gram matrix with a
  cyclic Jacobi eigensolver.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (135–142 dB on the frozen references).
  - One thread handles each group in a fixed order, and aggregation is
    ordered.
- **Memory.** The ADMM state costs about 12 KiB of device memory per group at
  the defaults, so a 1080p frame runs in several batches.

## NCSR

- **Device-resident.** Matching, the per-group PCA and centralized shrinkage,
  the aggregation and all outer rounds run on the device.
- **Temporal output.** The temporal accumulation stays on the device when
  `iters = 1`; with more rounds the fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Numerics.** The PCA is taken through the FP32 Gram matrix with a cyclic
  Jacobi eigensolver, and the column weights use `expf` where the CPU uses
  its fast exponential.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (105–139 dB on the frozen references).
  - The output is run-to-run identical: every group is solved in a fixed
    order, and aggregation is ordered.
  - Groups of up to 8 patches run one thread per group. Larger groups run one
    thread block per group with a round-robin parallel Jacobi.

## NLH

Given or blind sigma, Gray / YUV / RGB.

- **Device-resident.** The whole frame pipeline runs on the device: the
  RGB/YUV conversion, the blind noise estimate, every Basic round over the
  request window, the Wiener round and the per-pixel aggregation.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Frame properties.** The `_NSS*` diagnostics match the CPU; `_NSSSigma` of
  a blind estimate agrees to float precision.
- **Numerics.** Matching, pixel selection, the Haar transform and the
  thresholds follow the CPU model; the per-pixel sums are accumulated in FP32
  where the CPU uses FP64.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (77–107 dB on the frozen references; the differences are coefficients
    that fall on the other side of the hard threshold).
  - The output is run-to-run identical.

## TWSC

Given or blind sigma, Gray / YUV / RGB.

- **Device-resident.** The blind noise estimate, the joint matching, the
  per-group dictionary and ADMM solve, the aggregation and every round run on
  the device.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Numerics.** The dictionary comes from an FP32 Jacobi eigendecomposition of
  the Gram matrix of the smaller group side, and the solver runs in FP32
  (FP64 was measured and gave the same agreement with the CPU).
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (92–143 dB on the frozen references) and run-to-run identical.
- **Frame properties.** `_NSSSigma`, `_NSSGroups` and the shape properties
  match the CPU. `_NSSADMMMaxIterGroups` counts this backend's own
  non-converged groups; `_NSSSvdDoubleGroups` and `_NSSSylvesterResidual` are
  CPU-solver diagnostics and are reported as 0.
- **Speed.** The default settings remain very heavy (one 64x64
  eigendecomposition per pixel position per round): about 8 s for a 128x128
  frame on an RTX 5080, against about 190 s on 16 CPU threads.

## LSSC

- **Device-resident.** Patch packing, k-means, the dictionary (DCT atoms,
  sampled patches, one K-SVD round), the per-cluster sparse coding, the
  reconstruction and the aggregation run on the device. The host only
  sequences the steps and keeps index bookkeeping (cluster member lists and
  the k-means round control).
- **Numerics.** The K-SVD sparse codes use FP64 correlations and solves as on
  the CPU; an atom's rank-one update takes the dominant singular triplet from
  an FP64 power iteration instead of a full SVD. Everything else is FP32.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (139–141 dB on the frozen references) and run-to-run identical.
- **Memory.** The coefficients cost `min(256, patches)` floats per grid patch,
  so a dense `block_step` on a large frame takes that much device memory.
- **Speed.** The dictionary update is sequential over the atoms, so the gain
  over the CPU is small: about 1–2x the CPU plugin at 16 threads for 1080p.
