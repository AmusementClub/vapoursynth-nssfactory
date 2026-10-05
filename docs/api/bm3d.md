# nss.BM3D

The reference block-matching 3D filter: similar patches are grouped into 3D
stacks, collaboratively filtered in a transform domain (hard thresholding in
the Basic stage, Wiener filtering in the second stage), and aggregated back.
Fast, mature, and the quality baseline every other filter here is compared
against.

```python
out = core.nss.BM3D(clip, sigma=25)                       # spatial default
fine = core.nss.BM3D(clip, sigma=25, block_size=8, block_step=4, bm_range=10)
temporal = core.nss.BM3D(clip, sigma=25, radius=2)        # see radius note below
```

## Signature

```python
core.nss.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8,
              int[] group_size = 8, int[] block_step, int[] bm_range = 7,
              int radius = 0, int[] ps_num, int[] ps_range = 4,
              int memory_limit_mb])
```

Array-typed parameters take one value per plane; a single value broadcasts.
Omitted `block_step` adapts to `min(8, block_size)` per plane; omitted
`ps_num` adapts to `min(2, group_size)`.

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise standard deviation per plane. Controls threshold/Wiener strength. Note the input is remapped through the BM3D effective-noise profile, so the response tracks the paper's expected behaviour. `sigma=0` bypasses the plane. |
| `block_size` | 8 | {1,2,4,8,12,16,32} | Patch edge. 8 is the general default; 12 exists for high-noise DCT profiles. Larger blocks cost quadratically per patch. |
| `block_step` | min(8, block) | [1, block] | Stride between reference patches. Halving it roughly quadruples positions (2D) — the main quality/speed trade. |
| `group_size` | 8 | {1,2,4,8,16,32,64} | Maximum patches per 3D stack. More matches help at high noise; marginal at low noise. |
| `bm_range` | 7 | [1, 64] | Search window radius (`window = 2*bm_range + 1`). Cost grows quadratically; motion/texture may justify a larger window. |
| `radius` | 0 | [0, 16] | Temporal radius in frames. >0 switches output to the weighted intermediate for `VAggregate`. |
| `ref` | none | clip | External reference clip guiding both stages' matching (paper-style two-stage usage). |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `ps_num` | min(2, group) | [1, group] | Predictive-search seeds kept per temporal step (only used with `radius > 0`). |
| `ps_range` | 4 | [1, 64] | Predictive-search window radius around each seed (temporal mode). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Algorithm and paper

Dabov, Foi, Katkovnik & Egiazarian, *Image Denoising by Sparse 3D
Transform-Domain Collaborative Filtering*, IEEE TIP 2007. Two stages (Basic
hard-threshold, Wiener) with the standard `2.7 * sigma` hard-threshold
calibration, DCT/block transforms as applicable, and the plugin's deterministic
block-matching underneath. The paper's pipeline is followed; the numerics are
pinned down (fixed reduction orders, deterministic matcher tie-breaking) so the
same build is reproducible bit-for-bit on a given toolchain.

## Defaults rationale and performance

`block=8, step=8, group=8, bm_range=7, sigma=3` is the balanced point chosen
from this repository's calibration matrix. ~59 ms per 1080p GRAYS frame
on a single Emerald Rapids core (~17 fps) — the fastest filter here. The
dominant cost driver is `block_step` (positions scale as `1/step^2`), then
`bm_range` and `group_size`.

## Pitfalls

- With `radius > 0` the output is a **taller intermediate**, not a viewable
  frame — pipe it through `core.nss.VAggregate(out, clip, radius=radius)`.
- `sigma` is in 8-bit units even though the clip is float32; 25 means the
  usual "25/255" noise.
- Array parameters are per-plane: `block_size=[8,16]` is legal on YUV.

## CUDA (`core.nss_cuda.BM3D`, `core.nss_cuda.VAggregate`)

The CUDA plugin (`libnss_cuda`, built with `-DNSS_ENABLE_CUDA=ON`) takes the
same arguments and gives the same errors as `nss.BM3D` / `nss.VAggregate`. It
adds two GPU-only arguments at the end of the argument list. Its temporal
arguments, shared with the CPU signature, are the ones to know on the device:

| Parameter | Default | Meaning |
|---|---|---|
| `temporal_mode` | `"rolling"` | With `radius > 0`: `"rolling"` returns finished, normal-height frames; `"legacy"` returns the fat intermediate for `VAggregate`, as the CPU plugin does. |
| `rolling_chunk` | 4 | Frames accumulated per rolling chunk, 1 to 64. |
| `rolling_cache_limit` (or `rolling_cache_chunks`) | 1 | Finished chunks kept for later frame requests, 1 to 64. |
| `device_id` | 0 | CUDA device index. |
| `num_streams` | 1 | How many frames (or rolling chunks) can be in flight on the device at once, 1 to 16. Each stream has its own device buffers. One stream already keeps the device busy; more add little (about 15% for the rolling mode). With `memory_limit_mb`, a value that does not fit is a creation error. `VAggregate` accepts both arguments but does not use a device. |

```python
basic = core.nss_cuda.BM3D(clip, sigma=25)
final = core.nss_cuda.BM3D(clip, ref=basic, sigma=25)
# Temporal: finished frames by default (temporal_mode="rolling").
temporal = core.nss_cuda.BM3D(clip, sigma=25, radius=1)
# The CPU plugin's two-step form, for mixing backends or debugging.
fat = core.nss_cuda.BM3D(clip, sigma=25, radius=1, temporal_mode="legacy")
temporal = core.nss_cuda.VAggregate(fat, clip, radius=1)
```

- **Device-resident.** Matching (spatial or predictive temporal), collaborative
  filtering and aggregation all run on the device.
  - Spatial and rolling output copy only final frames back.
  - Legacy `radius > 0` copies back the fat intermediate, because it must
    exist as a VS frame.
- **Rolling.**
  - Each chunk of frames keeps a ring of the temporal window on the device.
  - Each center frame's contributions are added into the chunk's target
    frames in ascending order, as the CPU does.
  - It is bit-identical to `nss_cuda.VAggregate(nss_cuda.BM3D(..., radius=R))`.
- **`nss_cuda.VAggregate` runs on the host.** Its inputs are already host
  frames, and uploading 2(2R+1) planes per frame costs several times more than
  the sum itself. It sums slices in the CPU's order and divides with IEEE
  rounding.
- **Interop.** The fat intermediate is a plain VS frame with versioned
  properties, so `nss.VAggregate` and `nss_cuda.VAggregate` accept each other's
  BM3D output. They agree within a few ulp: the CPU's fast-math division may
  be up to 2 ulp off IEEE rounding.
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
    gate in `tests/data/cuda_tolerances_v1.json`.
  - The output is run-to-run identical on a given GPU, driver and build.
    Spatial filtering of most shapes aggregates from inside the filter kernel
    with integer atomics on fixed-point sums (exact, so independent of the
    order); temporal filtering and the remaining shapes sort their patches
    and sum them in a fixed order.
- **Memory.** `nss_cuda` has no default `memory_limit_mb`: the filter takes
  what its plan needs and a failed device allocation is reported as the CUDA
  out-of-memory error. With `memory_limit_mb`, the value also caps device
  memory, pinned staging and the rolling chunk cache, and the filter runs
  with smaller internal batches to fit (the output does not change). The
  smallest limit one stream accepts at 4K, for GRAYS / YUV420 / YUV444 or RGB:
  - Spatial: about 300 / 310 / 350, and 350 / 370 / 420 with `ref`.
  - Rolling, `radius = 1`: 1280 / 1440 / 1910, and 1630 / 1780 / 2260 with `ref`.
  - Rolling, `radius = 2`: 1600 / 1750 / 2230, and 2130 / 2290 / 2770 with `ref`.
  - Legacy with `ref`: 990 / 1090 / 1370 at `radius = 1`, 1630 / 1790 / 2260
    at `radius = 2`.
- **Performance.** At 1080p GRAYS on an RTX 5080 (`num_streams=3` for temporal):
  - Spatial runs at bm3dcuda's speed or slightly faster.
  - Temporal runs at about 0.85 to 0.95x bm3dcuda. It uses the CPU's
    predictive search and deterministic aggregation.
