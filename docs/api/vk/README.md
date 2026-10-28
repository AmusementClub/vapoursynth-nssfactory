# nss_vk.* Reference (preview)

`libnss_vk` (namespace `nss_vk`) runs filters of the CPU plugin on Vulkan
GPUs. It is a **preview**: two filters are ported so far, [BM3D](bm3d.md) and
[NLM](nlm.md), and it has been validated on NVIDIA GPUs only (AMD and Intel
GPUs are untested). The arguments, defaults and models are the CPU plugin's;
the output agrees with it to the CUDA plugin's 60 dB PSNR gate or better (not
bit for bit) and is identical from run to run on a given GPU, driver and
build. On VapourSynth R80 and later it also processes frames that are already
on the GPU ([GPU frames](#gpu-frames-vapoursynth-r80)), so a chain of GPU
filters keeps its frames in video memory.

## Speed

Frames per second at 1080p, default arguments (BM3D `sigma=3`), on an RTX 5080
(Ryzen 9 5950X host, driver 595.91.07), one instance, frames requested in
order and not read back by the consumer, median of three runs (2026-10-28):

| Filter | VapourSynth R81, CPU frames (gray / YUV420 / RGB) | vapoursynth-classic, [direct transfers](#direct-transfers-with-vapoursynth-classic) | R81, [GPU frames](#gpu-frames-vapoursynth-r80) |
|---|---:|---:|---:|
| [BM3D](bm3d.md) | 730 / 485 / 245 | 1165 / 756 / 375 | 1703 / 1051 / 586 |
| [NLM](nlm.md) | 804 / 710 / 209 | 851 / 836 / 381 | 841 / 839 / 505 |

- **CPU frames** are uploaded through resizable BAR memory (or staging) and
  downloaded through host memory on nss_vk's own device; that is also the path
  on R78 and older. **GPU frames** were made on the GPU (a noisy frame
  uploaded once, or `std.BlankClip(gpu=1)`), so the column is the filter alone:
  BM3D is then bound by its kernels (the GPU is 97% busy).
- A single filter between `std.GPUUpload` and `std.GPUDownload` runs at about
  the speed of its CPU-frame path (BM3D 843 / 544 / 279, NLM 840 / 560 / 276):
  the upload and download cost what the filter's own transfers did. The gain
  of GPU frames is in chains ([below](#when-gpu-frames-pay)).
- Against `nss_cuda` (the same host): BM3D 1.03–1.07× on classic with direct
  transfers and 1.19–1.27× on R78; NLM 0.87–1.03× on classic.

## Functions

| Function | GPU page | CPU page (algorithm, paper) |
|---|---|---|
| `core.nss_vk.BM3D` | [bm3d.md](bm3d.md) | [bm3d.md](../bm3d.md) |
| `core.nss_vk.NLM` | [nlm.md](nlm.md) | [nlm.md](../nlm.md) |

```python
core.nss_vk.Version()                 # version:data
core.nss_vk.Backend([int device_id])  # loader, device, support level, GPU frames
```

The other filters of the CPU plugin are not ported yet; `PNLM` is next.
`VAggregate` is the CPU plugin's alone: `core.nss.VAggregate` reduces the
legacy temporal output of `nss_vk.BM3D` (also when it is a GPU clip: the core
downloads it).

## Shared by every filter

Each filter takes the arguments of its `nss` counterpart (same names,
defaults, ranges and error messages, under the `nss_vk` name) plus these, as
in [`nss_cuda`](../cuda/README.md#shared-by-every-filter):

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `device_id` | 0 | device index | The Vulkan device of CPU-frame clips, in the loader's order. `core.nss_vk.Backend(device_id)` reports whether it is supported. GPU frames run on VapourSynth's device; there `device_id` may be omitted, or must name that device. |
| `num_streams` | 1 | [1, 16] | Frames (or temporal chunks) in flight at once; each adds its own buffers. Beyond them NLM keeps two more frames in flight and BM3D one (spatial and legacy), as far as `memory_limit_mb` allows. |
| `memory_limit_mb` | none | > 0 | Caps the instance's device memory and the host memory it allocates; the work is batched to fit, the output does not change, and a limit too small for the clip is an error when the filter is created. |

Temporal output (`radius > 0`) works as in `nss_cuda`: `temporal_mode`
`"rolling"` (default, finished frames) or `"legacy"` (the CPU plugin's
intermediate for `VAggregate`), with `rolling_chunk`, `rolling_cache_chunks`
and `rolling_cache_limit` ([temporal output](../cuda/README.md#temporal-output)).

## Device requirements

A device runs `nss_vk` when it has Vulkan 1.3 with:

- buffer device address, timeline semaphores, 64-bit integers in shaders and
  64-bit integer atomics on buffers (the sums are fixed point);
- subgroup basic, vote, ballot, shuffle and arithmetic operations in compute
  shaders, subgroup size control with full subgroups;
- 48 KiB of shared memory per workgroup.

`core.nss_vk.Backend(device_id)` reports it:

| Key | Meaning |
|---|---|
| `support` | `native`: every kernel runs on subgroups of 32 lanes. `emulated`: the widest subgroup is below 32; kernels with more lanes per reference run on an emulated warp (same results, slower). `unsupported`: `missing` lists what the device lacks. |
| `name`, `driver_name`, `driver_info`, `api_version` | The device and its driver. |
| `optional` | Optional features present: `external_memory_host` (direct transfers), `transfer_queue` (copies beside the kernels), `host_visible_device_memory` (resizable BAR uploads). |
| `probe_ok`, `probe_ms` | A kernel run at load. |
| `gpu_frames` | Whether GPU frames are processed on this core: `yes (R81)`, or why not (see below). |

- **Vendors.** Validated on NVIDIA (RTX 5080, driver 595.91.07, Linux).
  Mesa's software device (lavapipe) has 32 KiB of shared memory and is
  `unsupported`. AMD and Intel GPUs that meet the list should run, but are
  untested, as are Windows drivers at run time.
- **No loader needed to load.** The plugin finds the Vulkan loader when a
  filter is created; without one, `Backend()` says so.
- **Pipeline cache.** Compiled kernels are kept on disk per device and driver
  (`~/.cache/nssfactory/vk`, `%LOCALAPPDATA%\nssfactory\vk` on Windows), so
  only the first filter creation after an update compiles them.
  `NSS_VK_PIPELINE_CACHE` names another directory, or `off`.

## GPU frames (VapourSynth R80+)

On VapourSynth R80 and later (API 4.3), every clip argument of an `nss_vk`
filter is declared `vnode:all`: it takes CPU frames and GPU frames, and its
output is where its input is.

```python
gpu = clip.std.GPUUpload()                                  # once, at the head of the chain
den = core.nss_vk.BM3D(gpu, sigma=3, final=1)               # on the GPU
den = core.nss_vk.NLM(den, d=1, h=2.4)                      # still on the GPU
out = den.std.GPUDownload()                                 # or leave it: output() downloads
```

- **Same output.** A filter gives the same bits on GPU frames as on CPU
  frames (the same kernels and plan; the planes are copied on the device
  instead of uploaded).
- **The device is VapourSynth's.** GPU frames are processed on the core's
  Vulkan device (plugins may not create their own for them); choose it with
  `core.set_vulkan_device(index)` before the first GPU filter.
  `device_id` may be omitted, or must name the same device in nss_vk's order
  (an error otherwise). CPU-frame clips keep running on `device_id`.
- **Residency.** A filter's clips must all be CPU frames or all GPU frames
  (`clip` and `ref` / `rclip`); mixing them is an error naming both (pass one
  through `std.GPUUpload` or `std.GPUDownload`).
- **VRAM.** nss_vk's buffers on the core's device are reported to its VRAM
  budget (`core.max_vram_cache_size`), so VapourSynth makes room for them by
  evicting cached GPU frames; `memory_limit_mb` still caps nss_vk's own
  buffers.
- **Planes left alone** (NLM's `channels="Y"` or `"UV"`, BM3D planes with
  `sigma=0`) share the source's GPU planes: no copy.
- **Legacy temporal output** stays on the GPU too; `core.nss.VAggregate`
  (a CPU filter) gets it downloaded by the core.

### Tested cores

The GPU interface of VapourSynth (`VSVulkan4.h`) has no version of its own
and changes with the core, so nss_vk processes GPU frames only on the cores
it was built and tested against: **R80 and R81**. On a newer core the same
binary still loads; GPU clips are then downloaded (`std.GPUDownload`) and run
on CPU frames, with one warning per core, until a release tested against that
core. `Backend()["gpu_frames"]` says which applies:

| `gpu_frames` | Meaning |
|---|---|
| `yes (R81)` | GPU frames are processed on the GPU. |
| `downloaded (R82 untested; ...)` | An untested core: GPU clips run on CPU frames. `NSS_VK_GPU_FRAMES=force` processes them on the GPU anyway, at your own risk. |
| `off (NSS_VK_GPU_FRAMES=0)` | Turned off by the environment. |
| `no (VapourSynth API R4.2: R80 or later)` | R78, vapoursynth-classic and older: CPU frames only, plain `vnode` arguments. |

### When GPU frames pay

- **Chains on the GPU.** Frames stay in video memory between GPU filters
  (`nss_vk`, the core's GPU filters, other GPU plugins): one upload at the
  head of the chain and one download at the end, instead of one each per
  filter. A resident BM3D runs at 1703 fps (gray 1080p) against 730 on CPU
  frames.
- **Not a single filter.** Between `std.GPUUpload` and `std.GPUDownload`
  alone, a filter runs at about its CPU-frame speed: the core's transfers
  replace the filter's own.
- **Kernel-bound filters gain less.** NLM's defaults and BM3D with `final=1`
  are bound by their kernels; for them a chain saves the transfers' CPU and
  bus time rather than adding frames per second.
- **vapoursynth-classic** has no GPU frames; there
  [direct transfers](#direct-transfers-with-vapoursynth-classic) are the fast
  path for CPU frames.

## Direct transfers with vapoursynth-classic

Under [vapoursynth-classic](https://github.com/AmusementClub/vapoursynth-classic)
[R57.A14](https://github.com/AmusementClub/vapoursynth-classic/releases/tag/R57.A14)
or newer, nss_vk imports the frames' memory into the device
(`VK_EXT_external_memory_host`, where the device has it) and its copy queues
read the input planes from and write the output planes into the frames
themselves, as `nss_cuda` does with page-locked memory
([details](../cuda/README.md#direct-transfers-with-vapoursynth-classic)). At
most half the core's frame cache is imported.
`NSS_VK_DIRECT_IO=0` turns it off, `NSS_VK_DIRECT_IO_MAX_MB` sets the cap and
`NSS_VK_DIRECT_IO_STATS` prints counts when a core's last instance is freed.

## Environment

| Variable | Meaning |
|---|---|
| `NSS_VK_GPU_FRAMES` | `force`: GPU frames on any core with API 4.3; `0`: never (GPU clips are downloaded). |
| `NSS_VK_PIPELINE_CACHE` | Pipeline cache directory, or `off`. |
| `NSS_VK_DIRECT_IO`, `NSS_VK_DIRECT_IO_MAX_MB`, `NSS_VK_DIRECT_IO_STATS` | Direct transfers with vapoursynth-classic (above). |
| `NSS_VK_WAIT_TIMEOUT_MS` | How long a frame may take on the device before it is an error (default 60000). |
| `NSS_VK_REBAR=0`, `NSS_VK_COPY_QUEUES=0/1/2` | Diagnostics: no resizable BAR uploads; fewer copy queues. |
