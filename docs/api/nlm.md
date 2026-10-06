# nss.NLM

Classic Non-Local Means: every pixel is replaced by a weighted average of
pixels whose surrounding patch resembles it, searched in a window around (and,
with `d > 0`, before/after) the current position. The lightest filter in the
plugin; useful for mild noise and as a reference point for the heavier
group-based methods.

```python
out = core.nss.NLM(clip, h=1.2)          # defaults
denoised = core.nss.NLM(clip, d=1, a=4, s=3, h=1.5, channels="Y")
```

## Signature

```python
core.nss.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2,
             string channels = "AUTO", int wmode = 0, float wref = 1.0,
             clip rclip = None, int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `h` | 1.2 | > 0 | Filtering strength. Larger values smooth more aggressively. This is the knob to turn first; the default is intentionally conservative (earlier calibration found `h=1.2` barely denoises strong noise — raise it with the noise level). |
| `d` | 1 | [0, 256] | Temporal radius in frames. `d=0` is purely spatial; `d>0` pools matches from `d` neighbouring frames on each side, improving quality on static regions at a linear cost in frames searched. |
| `a` | 2 | [1, 64] | Search radius in pixels around the anchor. Cost grows as `(2a+1)^2`; quality saturates quickly. |
| `s` | 4 | [0, 1024] | Patch radius for the similarity comparison (patch size `2s+1`). Larger patches are more noise-robust but blur fine structure. |
| `channels` | `"AUTO"` | Y / UV / YUV / RGB / AUTO | Which planes to process. AUTO selects by color family. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `wref` | 1.0 | > 0 | Weight of the reference (center) pixel's own contribution. Below 1 trusts neighbours more. |
| `wmode` | 0 | 0 only | Weight kernel. Only `0` (Welsch) is implemented; other values are rejected. |
| `rclip` | none | clip | Reference clip for similarity search (same format/size). |
| `memory_limit_mb` | none | — | Workspace cap; the call fails instead of degrading output. |

## Algorithm and paper

Buades, Coll & Morel, *A Non-Local Algorithm for Image Denoising*, CVPR 2005.
The implementation follows the patch-distance formulation directly; distances
use the fixed reduction kernels shared with the other filters, and the temporal
pooling (`d`) is the standard extension. There is no deliberate departure to
document — the model is the paper's.

## Defaults rationale and performance

Defaults (`d=1, a=2, s=4, h=1.2`) are the classic mild-denoise configuration.
~131 ms per 1080p GRAYS frame on a single Emerald Rapids core (~7.6 fps).
Cost scales with `(2a+1)^2 * (2d+1)`; `s` has a smaller linear effect.

## Pitfalls

- `h` is **not** in sigma units and does not track the noise level by itself;
  retune it when noise changes materially.
- `radius`-style fat-frame output does not exist here: temporal support is the
  `d` parameter and the result stays normal-height.
- `a` must be smaller than the processed plane width (creation-time error).
- `d` pins `2d+1` input frames; creation rejects radii whose pinned footprint exceeds 2 GiB for the actual frame size.

## CUDA (`core.nss_cuda.NLM`)

`core.nss_cuda.NLM` (from `libnss_cuda`, built with `-DNSS_ENABLE_CUDA=ON`)
takes the same arguments and gives the same errors as `nss.NLM`. It adds
`device_id` (default 0) and `num_streams` (default 1) at the end of the argument list.

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
