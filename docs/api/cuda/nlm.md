# nss_cuda.NLM

Non-local means on the device: all channel modes, temporal `d`, and `rclip`. The
output is always normal-height.

Algorithm, paper and the reasoning behind the defaults: [`nss.NLM`](../nlm.md).

```python
out = core.nss_cuda.NLM(clip, h=1.2)          # defaults
denoised = core.nss_cuda.NLM(clip, d=1, a=4, s=3, h=1.5, channels="Y")
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default (`d=1`) | 620 | 260 |

The release packages are built with CUDA 13.3. A CUDA 12.9 build (the toolkit of r2 and earlier) gives the same
output at 490 / 255 fps: its code generation suits this kernel less
([toolkit note](compare.md#notes)).

## Closer to the papers

The defaults are fast settings. Settings closer to the paper, with their quality and cost. Measured on one 1080p RGB frame (an anime credits frame with its own film grain) with Gaussian noise added, against the frame without it; the noisy input is 24.97 dB at sigma 15 and 20.98 dB at sigma 25. One image: take these as starting points, not tuned presets. σ is the standard deviation of the noise in 8-bit units; speed at 1080p on an RTX 5080 (the second row's on the test frame).

| Arguments | PSNR, sigma 15 | PSNR, sigma 25 | fps YUV420 / RGB |
|---|---:|---:|---:|
| defaults (`h=1.2`) | 24.97 dB | 20.98 dB | 620 / 260 |
| `d=0, h=0.6 σ` | 32.44 dB | 28.94 dB | 780 / 280 |
| `a=10, s=3, d=0, h=0.6 σ` | 33.31 dB | 30.53 dB | 130 / 81 |

- `h` is the KNLMeansCL scale (the weight is exp(-mean squared patch difference / h^2), without the paper's -2 sigma^2 offset), so it has to grow with the noise: the default `h=1.2` leaves noise of sigma 3 or more untouched. About 0.6 σ is best on a single frame (`d=0`), about 0.8 σ with `d` ≥ 1 on video, where the neighbouring frames carry their own noise.
- The 21 x 21 window and 7 x 7 patches are the paper's. They gain 0.9 to 1.6 dB on this grainy frame, but within 0.1 dB on 11 cleaner images (DIV2K and crops), at 3 (RGB) to 6 (YUV420) times the cost.
- An earlier version of this table gave `h=1.3 σ` with `d=1` on a clip of one frame: its "neighbours" were the same frame with the same noise, which favours a larger `h`. With `d=0`, `h=1.3 σ` gives 30.21 / 28.07 dB.

## Signature

```python
core.nss_cuda.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2,
                  string channels = "AUTO", int wmode = 0, float wref = 1.0,
                  clip rclip = None,
                  int memory_limit_mb, int device_id = 0, int num_streams])
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
| `wmode` | 0 | [0, 3] | Weight of `u`, the mean squared patch difference in 8-bit units over `h²` (the KNLMeansCL / nlm-ispc modes): `0` Welsch `exp(-u)`; `1`, `2`, `3` Bisquare A, B, C, `max(1-u, 0)` to the power 1, 2, 8. Bisquare weights are exactly zero past `u = 1` and need a much larger `h` (see Pitfalls). |
| `rclip` | none | clip | Reference clip for similarity search (same format/size). |

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 2, none | The GPU, the frames in flight ([default streams](README.md#default-streams)), and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |

## Notes

- **R81 GPU frames (experimental).** With the build- and run-time switches in
  [GPU frames](README.md#gpu-frames), NLM accepts and returns GPU-resident
  frames. This uses the plane path and device-to-device copies; the tile cache
  and weight-map reuse described below remain on the CPU-frame path.
- **Accuracy.** Above 144 dB against the CPU plugin (it is the same model with
  exact box sums); identical from run to run.
- **Frames are uploaded once**, however many windows they are part of. With
  `d=1` and frames rendered in order, the weights of a frame pair are computed
  once for both frames, whichever streams they run on: one set of maps more
  than there are streams, 198 MiB each at 1080p with the default `a=2` (594 MiB
  at the default two streams). They are the first thing `memory_limit_mb`
  takes away (the output is the same without them).
- **Memory.** About `2d + 4` frames of the clip per stream. Large `d`, patches
  or search windows fall back to a slower path with fewer frames resident.

## Pitfalls

- `h` is **not** in sigma units and does not track the noise level by itself;
  retune it when noise changes materially.
- With `wmode` 1–3 even a perfect match under Gaussian noise of σ (8-bit units)
  has `u ≈ 2σ²/h²`, so every weight is zero and the output equals the input
  until `h` exceeds about `1.4σ`. Welsch never reaches zero, so the same `h`
  is not comparable across modes.
- There is no `radius` and no fat intermediate: temporal support is the `d`
  parameter and the result is always normal-height.
- `a` must be smaller than the processed plane width (creation-time error).
- `d` pins `2d + 1` input frames; creation rejects radii whose pinned footprint
  exceeds 2 GiB for the actual frame size.
