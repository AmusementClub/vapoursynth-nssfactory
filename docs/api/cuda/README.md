# nss_cuda.* Reference

`libnss_cuda` (namespace `nss_cuda`) runs the eight denoisers of the CPU
plugin on NVIDIA GPUs. The arguments, defaults and models are the same; the
output agrees with the CPU plugin to 60 dB PSNR or better (not bit for bit)
and is identical from run to run on a given GPU, driver and build. Each page
below describes every argument of one function; the papers and the reasoning
behind the defaults are on the CPU pages.

## Speed

Frames per second at 1080p, default arguments, on an RTX 5080 (Ryzen 9 5950X
host), one instance, frames requested in order. Median over a set of nine test
images with Gaussian noise of sigma 3, each timed for at least 7 s after a
warm-up; the images differ by at most 3%.

| Filter | YUV420 | RGB | Temporal, `radius=1` (YUV420 / RGB) |
|---|---:|---:|---:|
| [BM3D](bm3d.md) | 495 | 280 | 455 / 230 |
| [NLM](nlm.md) | 565 | 235 | (temporal by default, `d=1`) |
| [WNNM](wnnm.md) | 500 | 285 | 455 / 230 |
| [MCWNNM](mcwnnm.md) | — | 200 | — / 135 |
| [NCSR](ncsr.md) | 390 | 215 | 265 / 135 |
| [LSSC](lssc.md) | 205 | 145 | (no temporal mode) |
| [NLH](nlh.md) | 37 | 31 | 13 / 11 |
| [TWSC](twsc.md) | 77 | 44 | |

- NLH runs its default blind noise estimation; giving `sigma` skips it.
- TWSC's default is a speed setting. Its quality geometry (`block_step=1,
  group_size=90, iters=12`) denoises 8 to 10 dB better and takes about 80 s
  per YUV420 1080p frame and 45 s per RGB frame; its page has the trade-offs.
- MCWNNM takes RGB only.
- The CPU plugin's speed is on the [CPU API guide](../README.md).

## Functions

| Function | GPU page (all arguments) | CPU page (algorithm, paper) |
|---|---|---|
| `core.nss_cuda.BM3D` | [bm3d.md](bm3d.md) | [bm3d.md](../bm3d.md) |
| `core.nss_cuda.NLM` | [nlm.md](nlm.md) | [nlm.md](../nlm.md) |
| `core.nss_cuda.WNNM` | [wnnm.md](wnnm.md) | [wnnm.md](../wnnm.md) |
| `core.nss_cuda.MCWNNM` | [mcwnnm.md](mcwnnm.md) | [mcwnnm.md](../mcwnnm.md) |
| `core.nss_cuda.NCSR` | [ncsr.md](ncsr.md) | [ncsr.md](../ncsr.md) |
| `core.nss_cuda.LSSC` | [lssc.md](lssc.md) | [lssc.md](../lssc.md) |
| `core.nss_cuda.NLH` | [nlh.md](nlh.md) | [nlh.md](../nlh.md) |
| `core.nss_cuda.TWSC` | [twsc.md](twsc.md) | [twsc.md](../twsc.md) |

```python
core.nss_cuda.Version()                 # version:data
core.nss_cuda.Backend([int device_id])  # device, driver and runtime versions, support level
```

`VAggregate` is the CPU plugin's alone: the GPU filters return finished
frames, and `core.nss.VAggregate` reduces their legacy temporal output.

## Shared by every filter

Each filter takes the arguments of its `nss` counterpart (same names,
defaults, ranges and error messages, under the `nss_cuda` name) plus these:

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `device_id` | 0 | device index | The GPU the instance runs on. `core.nss_cuda.Backend(device_id)` reports whether it is supported. |
| `num_streams` | 1 | [1, 16] | Frames (or temporal chunks) in flight on the GPU at once. One stream keeps the GPU busy; each extra stream adds its own buffers. |
| `memory_limit_mb` | none | > 0 | Caps the instance's GPU and pinned host memory; the work is batched to fit, the output does not change, and a limit too small for the streams is an error when the filter is created. Without it the filter takes what it needs. |

Input conventions (`sigma`, `rclip`, `bm_range`, frame properties) are those
of the [CPU plugin](../README.md#conventions).

### Temporal output

With `radius > 0` the GPU filters return finished frames: the temporal
accumulation, and every outer round of MCWNNM, NCSR and TWSC, stays on the
GPU. NLM is temporal through `d`; LSSC has no temporal mode.

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `temporal_mode` | `"rolling"` | rolling / legacy | `"rolling"` returns finished frames; `"legacy"` returns the CPU plugin's intermediate, for `core.nss.VAggregate`. |
| `rolling_chunk` | 4 | [1, 64] | Frames computed together. |
| `rolling_cache_chunks` | 1 | [1, 64] | Finished chunks kept for later frame requests at first. |
| `rolling_cache_limit` | 16 | [1, 64], >= `rolling_cache_chunks` | What the cache may grow to when requests come back to chunks it dropped. (On the CPU the two arguments name one fixed size.) |

- Rendering in order computes every frame once: each chunk carries on from
  the one before it. Random access recomputes the frames around each chunk.
- Rolling output agrees with the legacy intermediate reduced by `VAggregate`
  (bit for bit for NLH and TWSC, to rounding for the others) and needs more
  GPU memory than legacy output.

### Driver and GPU support

- The release packages are built with CUDA 12.9 and need a driver for it or
  newer.
- Native code covers compute capability 7.5, 8.6, 8.9 and 12.0 (RTX 20, 30,
  40 and 50 series). Other GPUs of 7.5 or newer run through the driver's PTX
  JIT (untested, best effort); `Backend()` reports which one a device gets.
- Building is described in [Building](../building.md).
