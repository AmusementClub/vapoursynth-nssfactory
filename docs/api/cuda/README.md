# nss_cuda.* Reference

`libnss_cuda` (namespace `nss_cuda`) runs the filters of the CPU plugin
(all but `VAggregate`) on NVIDIA GPUs. The arguments, defaults and models are the same; the
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
| [NLM](nlm.md) | 620 | 260 | (temporal by default, `d=1`) |
| [PNLM](pnlm.md) | 470 | 236 | (temporal by default, `d=1`; every plane) |
| [WNNM](wnnm.md) | 500 | 285 | 455 / 230 |
| [MCWNNM](mcwnnm.md) | — | 200 | — / 135 |
| [NCSR](ncsr.md) | 390 | 215 | 265 / 135 |
| [LSSC](lssc.md) | 305 | 165 | (no temporal mode) |
| [NLH](nlh.md) | 37 | 31 | 13 / 11 |
| [TWSC](twsc.md) | 77 | 44 | |
| [BM4D](bm4d.md) | 27 | — | (temporal by construction; gray 40) |
| [DA3D](da3d.md) | 37 / 70 | 5.6 / 14 | (sigma 3 / 10; a last step on a guide) |

- BM4D and DA3D are measured differently: BM4D on 160 frames of an anime episode
  with grain of sigma 5, DA3D on one test image at the sigma given, with
  `BM3D(final=1)` at 4/3 sigma as its guide (computed beforehand). DA3D's
  output is within a level of the CPU plugin's; BM4D's matching takes FP16
  distances (its PSNR is the CPU's).
- PNLM filters every plane by default; `NLM` filters only luma and measured
  711 / 269 fps the same way (`sigma=[3, 0]`: 606).
- NLH runs its default blind noise estimation; giving `sigma` skips it.
- TWSC's default is a speed setting. Its quality geometry (`block_step=1,
  group_size=90, iters=12`) denoises 8 to 10 dB better and runs at about
  0.012 fps for YUV420 and 0.022 fps for RGB at 1080p; its page has the trade-offs.
- MCWNNM takes RGB only.
- The release packages are built with CUDA 13.3. Up to r2 they were built with CUDA 12.9, which
  gives the same output and the same speed within 5% for every filter except NLM, about 20%
  fewer fps at its defaults (490 / 255 fps).
- Against BM3DCUDA, VapourSynth-WNNM, vs-nlm-ispc and vs-nlm-cuda at the same arguments, and
  r2 against r1: [Against other plugins](compare.md).
- The CPU plugin's speed is on the [CPU API guide](../README.md).

## Closer to the papers

The defaults are fast settings shared by most filters (`block_step=8`, small
groups). Each filter page has a "Closer to the papers" section with arguments
nearer its paper, their quality and their cost. In short, with σ the noise's
standard deviation in 8-bit units (one 1080p RGB test frame with Gaussian
noise added; PSNR against the frame without it, noisy input 24.97 dB):

| Filter | Arguments | PSNR at sigma 15: default → these | fps RGB: default → these |
|---|---|---:|---:|
| [BM3D](bm3d.md#closer-to-the-papers) | `final=1, block_step=3, group_size=32, group_size_basic=16, bm_range=19, sigma=4/3 σ` | 31.5 → 34.4 | 280 → 10.9 |
| [WNNM](wnnm.md#closer-to-the-papers) | `block_step=4, group_size=32, bm_range=15, sigma=1.41 σ` | 26.3 → 33.7 | 285 → 4.3 |
| [MCWNNM](mcwnnm.md#closer-to-the-papers) | `block_step=4, group_size=32, iters=4` | 25.6 → 33.7 | 200 → 0.38 |
| [NCSR](ncsr.md#closer-to-the-papers) | `block_step=4, group_size=16` | 29.4 → 33.4 | 215 → 18.5 |
| [LSSC](lssc.md#closer-to-the-papers) | `block_step=3` | 31.9 → 32.2 | 165 → 116 |
| [TWSC](twsc.md#closer-to-the-papers) | `block_step=4, group_size=90, iters=4` | 25.3 → 36.8 | 44 → 0.38 |
| [NLH](nlh.md#closer-to-the-papers) | defaults (or `sigma=σ`) | 35.2 (35.6) | 31 |
| [NLM](nlm.md#closer-to-the-papers) | `a=10, s=3, d=0, h=0.6 σ` | 25.0 → 33.3 | 260 → 81 |

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

Without `num_streams`, each filter takes the fewest streams that measured as
fast as any count up to three, and fewer where `memory_limit_mb` does not
hold them. Extra streams
overlap one frame's transfers and host work with another frame's kernels.
They pay where those leave the GPU idle, and not where the kernels dominate.

| Filter | Default | Gain over one stream | Peak GPU memory, one stream → default |
|---|---|---|---|
| NLM | 2 | 1.23× (YUV420, d = 1); with direct transfers 1.06–1.50× (three: no faster) | 529 → 759 MiB (gray, d = 1) |
| PNLM | 3 | 1.19× (gray, d = 1; two streams 1.13×) | 182 → 325 MiB (gray, d = 1) |
| TWSC | 3 | 1.22× (gray; two streams: 1.03×) | 385 → 1084 MiB |
| NCSR | 2; 3 with direct transfers | 1.10× (YUV420); with direct transfers 1.38–1.52× | 170 → 258 MiB (three: 346 MiB) |
| BM3D, WNNM | 1; 2 with direct transfers | 0.99× staged; with direct transfers 1.01–1.14× (three streams; two equal them on YUV420) | YUV420 150 / 162 MiB, plus about 68 / 80 MiB per stream |
| MCWNNM | 1; 3 with direct transfers | 0.99× staged; with direct transfers 1.46× (RGB) | 478 → 1009 MiB |
| DA3D | 3; 2 for joint RGB | see [da3d.md](da3d.md) | |
| LSSC, NLH | 1 | three streams: LSSC 0.73×, NLH 0.96× | |
| BM4D | 1 | computes one chunk at a time | |

- "Direct transfers" means vapoursynth-classic's frame memory extension (see
  [below](#direct-transfers-with-vapoursynth-classic)), with spatial output or
  `temporal_mode="legacy"`.
- Rolling temporal output keeps one stream; it has not been measured with
  more.
- The NLM default applies to its tile kernels (`a`, `s` and `d` within their
  limits, as at the defaults); NLM's plane path keeps one.
- Measured at 1080p on an RTX 5080 (Ryzen 9 5950X, VapourSynth R78 and a
  vapoursynth-classic build with the extension). Medians of three paired runs
  (classic: three runs in alternating order); outputs were identical at every
  stream count. Other GPUs
  and hosts may differ, and `num_streams` overrides the default.

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

An **experimental, opt-in R81 frontend** covers `nss_cuda.NLM`: build with
`-DNSS_CUDA_GPU_FRAMES=ON` and set `NSS_CUDA_GPU_FRAMES=1` before loading the
plugin. NLM then accepts CPU or GPU frames (`vnode:all`), keeps GPU input and
output resident, and uses its plane path with device-to-device plane copies.
Other filters keep their CPU-frame interface. On R80 and untested newer cores
NLM explicitly downloads GPU input instead. The frontend uses per-stream CUDA /
Vulkan timeline semaphores and a bounded external-memory import cache; it keeps
the existing host wait per frame. `Backend()["gpu_frames"]` reports the mode.
The normal Vulkan GPU-chain implementation remains
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
- **Speed** (1080p, RTX 5080, a decoder-like source, single runs, against
  staging): with one stream BM3D 1.46–1.65×, WNNM gray 1.63×, NCSR
  1.17–1.19×, MCWNNM RGB 1.13×; with `num_streams=3`, 1.53–1.67× across the
  four. NLM at its default two streams: gray 1.73×, YUV420 1.47×, RGB 1.67×,
  gray `d=0` 1.79× (two runs each). `nss_vk.NLM` (a preview since r3) uses the
  same extension and runs at 0.87–1.03× of `nss_cuda.NLM` on that core.
  Rolling `radius=1` (two runs, output hashed): BM3D gray 1.20×, YUV420 1.11×, WNNM gray 1.21×; NCSR, MCWNNM, LSSC, DA3D, BM4D, NLH and TWSC
  within ±3% (their kernels set the pace). Direct transfers raise the
  [default streams](#default-streams) of BM3D, WNNM, MCWNNM and NCSR. With a small frame cache the lock cap keeps most blocks
  staged, and the gain shrinks.
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

CPU time per frame, master `c161be3` against this behaviour (1080p, RTX
5080, a decoder-like source whose own work is in both columns, 30 s runs,
two each):

| Filter (vapoursynth-classic, direct transfers) | CPU ms/frame | fps |
|---|---|---|
| NLM gray `d=0` / RGB `d=1` | 5.10 → 1.85 / 15.75 → 5.55 | −0.2% / −0.6% |
| BM3D gray / YUV420 `radius=1` / gray `radius=1` legacy | 4.11 → 1.75 / 11.30 → 6.56 / 9.49 → 1.37 | −0.8% / −0.1% / −0.4% |
| WNNM gray / `radius=1` | 4.12 → 1.75 / 8.19 → 5.44 | −0.7% / −0.2% |
| NCSR gray / MCWNNM RGB / MCWNNM RGB `radius=1` | 3.74 → 1.13 / 13.94 → 4.57 / 24.30 → 11.01 | −0.7% / −0.2% / 0% |
| LSSC / DA3D | 10.31 → 1.01 / 68.4 → 1.44 | +0.6% / 0% |
| TWSC / NLH / BM4D | 17.63 → 17.05 / 33.6 → 21.4 / 46.9 → 33.7 | −0.2% / +0.2% / −0.5% |
| **VapourSynth R78** (staging): NLM gray / BM3D gray / BM3D YUV420 `radius=1` / LSSC | 10.80 → 9.22 / 7.99 → 6.56 / 17.05 → 14.63 / 10.71 → 2.91 | −0.1% / −0.2% / −0.2% / −0.4% |

The CPU this frees goes to the rest of the script (decoding, encoding, CPU
filters). The cost is the sleeping thread's wake-up where the GPU waits on
that thread: up to about 1% in the table, more where a frame is mostly waits
(NLM's plane path at 720p with `a=1`: −2.5% at a third of the CPU).
