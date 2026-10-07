# nss_cuda.BM3D

Block matching and 3D collaborative filtering on the device: spatial, `ref` (Wiener), temporal, `chroma` (CBM3D) and `final` (both stages in one call).

The parameters mean what they mean in [`nss.BM3D`](../bm3d.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

```python
basic = core.nss_cuda.BM3D(clip, sigma=25)
final = core.nss_cuda.BM3D(clip, ref=basic, sigma=25)
both = core.nss_cuda.BM3D(clip, sigma=25, final=1)        # the two lines above, estimate kept on the device
color = core.nss_cuda.BM3D(yuv444, sigma=[25, 15, 15], chroma=1)
# Temporal: finished frames by default (temporal_mode="rolling").
temporal = core.nss_cuda.BM3D(clip, sigma=25, radius=1)
# The CPU plugin's two-step form, for mixing backends or debugging.
fat = core.nss_cuda.BM3D(clip, sigma=25, radius=1, temporal_mode="legacy")
temporal = core.nss_cuda.VAggregate(fat, clip, radius=1)
```

## Signature

```python
core.nss_cuda.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8,
                   int[] group_size = 8, int[] block_step, int[] bm_range = 7,
                   int radius = 0, int[] ps_num, int[] ps_range = 4, int chroma = 0,
                   int final = 0, float[] sigma_basic, int[] block_size_basic,
                   int[] group_size_basic,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments). `temporal_mode` and the `rolling_*`
arguments are described under [temporal output](README.md#temporal-output).

## On the device

- **`num_streams`.** One stream already keeps the device busy; more add
  little (about 15% for the rolling mode).
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
- **Reducing the legacy intermediate.** See [VAggregate](vaggregate.md).
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
