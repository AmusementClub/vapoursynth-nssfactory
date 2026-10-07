# nss_cuda.* API Reference

`libnss_cuda` is the CUDA plugin (namespace `nss_cuda`), built with
`-DNSS_ENABLE_CUDA=ON`. It has all nine filters of the CPU plugin, one page
each below: the call signature and what is specific to the device (memory,
numerics, measured speed). This page has what they share. The meaning of the
parameters, the papers and the pitfalls are on the CPU pages and hold for
both plugins.

| Function | CUDA page | CPU page (parameters) |
|---|---|---|
| `core.nss_cuda.BM3D` | [bm3d.md](bm3d.md) | [bm3d.md](../bm3d.md) |
| `core.nss_cuda.VAggregate` | [vaggregate.md](vaggregate.md) | [bm3d.md](../bm3d.md) |
| `core.nss_cuda.NLM` | [nlm.md](nlm.md) | [nlm.md](../nlm.md) |
| `core.nss_cuda.WNNM` | [wnnm.md](wnnm.md) | [wnnm.md](../wnnm.md) |
| `core.nss_cuda.MCWNNM` | [mcwnnm.md](mcwnnm.md) | [mcwnnm.md](../mcwnnm.md) |
| `core.nss_cuda.NCSR` | [ncsr.md](ncsr.md) | [ncsr.md](../ncsr.md) |
| `core.nss_cuda.NLH` | [nlh.md](nlh.md) | [nlh.md](../nlh.md) |
| `core.nss_cuda.TWSC` | [twsc.md](twsc.md) | [twsc.md](../twsc.md) |
| `core.nss_cuda.LSSC` | [lssc.md](lssc.md) | [lssc.md](../lssc.md) |

```python
core.nss_cuda.Version()               # version:data
core.nss_cuda.Backend([int device_id])  # device, driver/runtime versions, support level
```

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
returns normal-height frames; LSSC has no temporal mode.)

| Parameter | Default | Meaning |
|---|---|---|
| `temporal_mode` | `"rolling"` | `"rolling"` returns finished frames; `"legacy"` returns the fat intermediate for `VAggregate`, exactly as the CPU plugin does. |
| `rolling_chunk` | 4 | Frames accumulated per rolling chunk, 1 to 64. |
| `rolling_cache_chunks` | 1 | Finished chunks kept for later frame requests at first, 1 to 64. Given alone, the cache stays at this size. |
| `rolling_cache_limit` | 16 | What the cache may grow to, 1 to 64 and at least `rolling_cache_chunks`. It keeps one more chunk each time two requests miss on chunks it dropped recently, so alternating between positions settles after a few misses. Under `memory_limit_mb` it grows only into what the limit leaves. (On the CPU the two arguments name one fixed size.) |

- BM3D, WNNM and single-round NCSR keep the temporal accumulation on the
  device and copy back only final frames. MCWNNM, NLH and TWSC (and NCSR with
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
