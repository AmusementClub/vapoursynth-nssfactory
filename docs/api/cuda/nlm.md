# nss_cuda.NLM

Non-local means on the device: all channel modes, temporal `d`, and `rclip`. The output is always normal-height.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.NLM`](../nlm.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.NLM(clip, h=1.2)          # defaults
denoised = core.nss_cuda.NLM(clip, d=1, a=4, s=3, h=1.5, channels="Y")
```

## Signature

```python
core.nss_cuda.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2,
                  string channels = "AUTO", int wmode = 0, float wref = 1.0,
                  clip rclip = None,
                  int memory_limit_mb, int device_id = 0, int num_streams = 1])
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

## Device parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `device_id` | 0 | device index | CUDA device the instance runs on. One instance uses one device; `core.nss_cuda.Backend(device_id)` reports whether it is supported. |
| `num_streams` | 1 | [1, 16] | Frames (or temporal chunks) the instance has in flight on the device at once. Each stream owns its device buffers, so memory grows with it. |
| `memory_limit_mb` | none | > 0 | Caps the instance's device and pinned host memory. The internal batches are fitted to it and the output does not change; a limit that cannot hold the streams is a creation error. Without it the filter takes what its plan needs. |

How the streams and the memory limit behave across filters is in the [shared notes](README.md#shared-by-every-filter).

## On the device

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
- **Weight maps are computed once with `d = 1`.** The map of a frame against
  the next one is also the next frame's map against it, so a stream keeps
  what a frame computed for the frame after it. This helps requests in
  order only and costs `2 (2a + 1)²` planes per stream (396 MiB at 1080p
  with the default `a = 2`); it is the first thing `memory_limit_mb` takes
  away, and it is left out when it would need more than a quarter of the
  device's memory. The output is the same with and without it.
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

## Pitfalls

- `h` is **not** in sigma units and does not track the noise level by itself;
  retune it when noise changes materially.
- There is no `radius` and no fat intermediate: temporal support is the `d`
  parameter and the result is always normal-height.
- `a` must be smaller than the processed plane width (creation-time error).
- `d` pins `2d + 1` input frames; creation rejects radii whose pinned footprint
  exceeds 2 GiB for the actual frame size.
