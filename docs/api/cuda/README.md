# nss_cuda.* Reference

`libnss_cuda` (namespace `nss_cuda`) runs the filters of the CPU plugin
(all but `VAggregate`) on NVIDIA GPUs. The arguments, defaults and models are the same; the
output agrees with the CPU plugin to 60 dB PSNR or better (not bit for bit)
and is identical from run to run on a given GPU, driver and build. Each page
below describes every argument of one function; the papers and the reasoning
behind the defaults are on the CPU pages.

## Performance notes

- The defaults are fast settings. Every filter page lists which arguments
  cost the most (denser references, larger groups and windows, more rounds).
- Several search shapes have dedicated, faster block-matching paths that give
  the same matches as the general search (BM3D, WNNM and NCSR; see
  [BM3D's optimized search shapes](bm3d.md#optimized-search-shapes)). When a
  shape is up to you, prefer one of them.
- NLH runs a blind noise estimation by default; giving `sigma` skips it.
- TWSC's default is a speed setting; its quality geometry is far slower (its
  page has the trade-offs).
- MCWNNM takes RGB only.
- The release packages are built with CUDA 13.3; older packages (up to r2)
  were built with CUDA 12.9, which gives the same output.
- Third-party plugins with the same algorithms: [Against other plugins](compare.md).

## Closer to the papers

The defaults are fast settings shared by most filters (`block_step=8`, small
groups). Each filter page has a "Closer to the papers" section with arguments
nearer its paper and their quality; they are much slower than the defaults.
In short, with σ the noise's standard deviation in 8-bit units (one 1080p RGB
test frame with Gaussian noise added; PSNR against the frame without it,
noisy input 24.97 dB):

| Filter | Arguments | PSNR at sigma 15: default → these |
|---|---|---:|
| [BM3D](bm3d.md#closer-to-the-papers) | `final=1, block_step=3, group_size=32, group_size_basic=16, bm_range=19, sigma=4/3 σ` | 31.5 → 34.4 |
| [WNNM](wnnm.md#closer-to-the-papers) | `block_step=4, group_size=32, bm_range=15, sigma=1.41 σ` | 26.3 → 33.7 |
| [MCWNNM](mcwnnm.md#closer-to-the-papers) | `block_step=4, group_size=32, iters=4` | 25.6 → 33.7 |
| [NCSR](ncsr.md#closer-to-the-papers) | `block_step=4, group_size=16` | 29.4 → 33.4 |
| [LSSC](lssc.md#closer-to-the-papers) | `block_step=3` | 31.9 → 32.2 |
| [TWSC](twsc.md#closer-to-the-papers) | `block_step=4, group_size=90, iters=4` | 25.3 → 36.8 |
| [NLH](nlh.md#closer-to-the-papers) | defaults (or `sigma=σ`) | 35.2 (35.6) |
| [NLM](nlm.md#closer-to-the-papers) | `a=10, s=3, d=0, h=0.6 σ` | 25.0 → 33.3 |

- One image: take these as starting points, not tuned presets.

## Sigma at the default settings

`sigma` sets each filter's strength; it equals the noise's σ only for some
filters and settings. At the default geometry the best `sigma` for Gaussian
noise of σ is (9 anime frames at half size and 3 at 1080p, σ 1 to 20: one
factor per filter stays within 0.1 dB of the best at every σ; PSNR on the
test frame above at σ 15):

| Filter | Best `sigma` | PSNR at sigma 15: `sigma=σ` → best |
|---|---|---:|
| BM3D | 4/3 σ (its sigma follows bm3dcpu's convention) | 31.5 → 32.6 |
| LSSC, NLH | σ | — |
| NCSR | about 1.8 σ | 29.4 → 33.0 |
| WNNM | about 2.2 σ | 26.3 → 32.1 |
| MCWNNM | about 2.5 σ | 25.6 → 33.7 |
| TWSC | about 3.4 σ | 25.3 → 34.9 |
| NLM | `h` about 0.6 σ (`d=0`), 0.8 σ (`d` ≥ 1 on video) | 25.0 → 32.4 |

- The speed does not change with `sigma`.
- The factor of NCSR, WNNM, MCWNNM and TWSC falls as `group_size` and
  `iters` grow: their models take σ as is near the papers' geometry (TWSC at
  `group_size=90, iters=4`, NCSR at `group_size=32, iters=2`). WNNM runs one
  pass where the paper iterates, so it keeps a factor of about 1.8 even at
  group 32. The pages give the factor of each "closer" geometry.
- BM3D, NLH and NLM want a larger `sigma` (`h`) on correlated grain than on
  white noise; the four group filters above barely move.
- A larger factor also lowers the contrast of fine text a little (BM3D at
  4/3 σ: white text 1.6 levels darker at σ 15 than at σ).

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
| `core.nss_cuda.BM4D` | [bm4d.md](bm4d.md) | [bm4d.md](../bm4d.md) |
| `core.nss_cuda.DA3D` | [da3d.md](da3d.md) | [da3d.md](../da3d.md) |

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
| `num_streams` | per filter, see [default streams](#default-streams) | [1, 16] | Frames (or temporal chunks) in flight on the GPU at once. Each stream adds its own buffers. |
| `memory_limit_mb` | none | > 0 | Caps the instance's GPU memory, pinned host memory and the host memory it allocates for frames (output frames, temporal chunk stores); the work is batched to fit, the output does not change, and a limit too small for the clip is an error when the filter is created, never while frames are rendered. Without it the filter takes what it needs. |

Input conventions (`sigma`, `rclip`, `bm_range`, frame properties) are those
of the [CPU plugin](../README.md#conventions).

### Default streams

Without `num_streams`, each filter takes the fewest streams that were as fast
as any count up to three on our test GPU, and fewer where `memory_limit_mb`
does not hold them. Extra streams overlap one frame's transfers and host work
with another frame's kernels; they help where those leave the GPU idle, and
not where the kernels dominate.

| Filter | Default | Peak GPU memory, one stream → default |
|---|---|---|
| NLM | 2 | 529 → 759 MiB (gray, d = 1) |
| PNLM | 3 for d ≤ 1; 2 beyond | 182 → 325 MiB (gray, d = 1) |
| TWSC | 3 | 385 → 1084 MiB |
| NCSR | 2; 3 with direct transfers; 1 on GPU frames | 170 → 258 MiB (three: 346 MiB) |
| BM3D, WNNM | 1; 2 with direct transfers | YUV420 150 / 162 MiB, plus about 68 / 80 MiB per stream |
| MCWNNM | 1; 3 with direct transfers | 478 → 1009 MiB |
| DA3D | 3; 2 for joint RGB | |
| LSSC, NLH, BM4D | 1 | |

- "Direct transfers" means vapoursynth-classic's frame memory extension (see
  [below](#direct-transfers-with-vapoursynth-classic)), with spatial output or
  `temporal_mode="legacy"`.
- GPU frames (the R81 frontend below) keep one stream for BM3D, WNNM, MCWNNM
  and NCSR: a frame gives its stream back once its work is queued, so the next
  frame's work follows it on the device.
- Rolling temporal output keeps one stream.
- The NLM default applies to its tile kernels (`a`, `s` and `d` within their
  limits, as at the defaults); NLM's plane path keeps one.
- The output is the same at every stream count. Other GPUs and hosts may
  prefer other counts; `num_streams` overrides the default.

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
- BM3D, WNNM, MCWNNM and NCSR copy a chunk's frames to and from the GPU while
  its kernels run (one more source frame per plane stays on the GPU for that:
  about 8 MiB at 1080p gray).

### GPU frames

By default `nss_cuda` takes CPU frames only. On VapourSynth R80+ a GPU-resident
clip (`std.GPUUpload`, a GPU filter's output) passed to an `nss_cuda` filter is
downloaded by the core first, and the output is a CPU frame.

An **experimental, opt-in R81 frontend** covers every `nss_cuda` filter:
build with `-DNSS_CUDA_GPU_FRAMES=ON` and set `NSS_CUDA_GPU_FRAMES=1` before
loading the plugin. Filters then accept CPU or GPU frames (`vnode:all`) and
keep GPU input and output resident. R80 and untested newer cores explicitly
download GPU input instead. The frontend uses per-stream CUDA / Vulkan timeline
semaphores, bounded external-memory import caches, and the existing host wait
per frame. `NSS_CUDA_GPU_IMPORT_CACHE=2..16` selects the per-stream memory-import
cache capacity (default 8). Temporal rolling chunks stay on the device, and NLM uses its tile
path to avoid the initial GPU-node plane-path regression. `Backend()["gpu_frames"]`
reports the mode. The normal Vulkan GPU-chain implementation remains
[`nss_vk`'s](../vk/README.md#gpu-frames-vapoursynth-r80).

### Driver and GPU support

- The release packages are built with CUDA 13.3 and need an NVIDIA driver of
  the r580 branch (CUDA 13.0) or newer.
- Native code covers compute capability 7.5, 8.6, 8.9 and 12.0 (RTX 20, 30,
  40 and 50 series). Other GPUs of 7.5 or newer run through the driver's PTX
  JIT (untested, best effort; it needs a driver for CUDA 13.3 or newer);
  `Backend()` reports which one a device gets.
- Building is described in [Building](../building.md).

### Direct transfers with vapoursynth-classic

Under [vapoursynth-classic](https://github.com/AmusementClub/vapoursynth-classic)
[R57.A14](https://github.com/AmusementClub/vapoursynth-classic/releases/tag/R57.A14)
or newer, which provides its `VSFrameMemory.h` extension (frame pool blocks and
release hooks), every filter copies between the GPU and the VapourSynth
frames directly, without the pinned staging copies.

- **Where:** every input frame. Every output frame of spatial output,
  `temporal_mode="legacy"`, NLM, LSSC and DA3D. Rolling output and BM4D
  compute several frames at once, before their output frames exist: those
  are downloaded into the chunk cache and copied into each frame as it is
  asked for. Output is the same either way.
- **NLH and TWSC** still reject infinite and NaN input samples: a plane
  uploaded directly is read on the host for that, but not copied.
- **How:** a frame pool block is page-locked the second time one of its
  planes is transferred, and unlocked when the pool returns it to the system.
  At most half the core's frame cache (`core.max_cache_size`) is locked;
  further blocks are staged.
- **Effect:** the staging copies on the host go away, which helps most where
  they set the pace (spatial BM3D, WNNM and NLM) and little where the kernels
  do (NLH, TWSC, BM4D). `nss_vk.NLM` uses the same extension. Direct
  transfers raise the [default streams](#default-streams) of BM3D, WNNM,
  MCWNNM and NCSR. With a small frame cache the lock cap keeps most blocks
  staged, and the effect shrinks.
- **Environment:** `NSS_CUDA_DIRECT_IO=0` turns it off,
  `NSS_CUDA_DIRECT_IO_MAX_MB` sets the lock cap in MiB, and
  `NSS_CUDA_DIRECT_IO_STATS` prints counts when a core's last instance is freed.
- Upstream VapourSynth has no such extension: there nothing changes.

### Host waits

A VapourSynth thread that waits for its frame on the GPU sleeps until the GPU
is done, instead of spinning: every event the host waits for is created with
`cudaEventBlockingSync`, and stream waits go through such an event. Without
it `cudaEventSynchronize` spins in the driver whatever the context's
scheduling flags, so a few threads stayed at 100% CPU for nothing. Events
that only order streams on the GPU stay without the flag, because recording a
blocking event costs more.

The CPU this frees goes to the rest of the script (decoding, encoding, CPU
filters). The cost is the sleeping thread's wake-up, which is small except
where a frame is mostly waits.
