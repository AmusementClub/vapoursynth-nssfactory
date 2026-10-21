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
| default (`d=1`) | 565 | 235 |

## Closer to the papers

The defaults are fast settings. Settings closer to the paper, with their quality and cost. Measured on one 1080p RGB frame (an anime credits frame with its own film grain) with Gaussian noise added, against the frame without it; the noisy input is 24.97 dB at sigma 15 and 20.98 dB at sigma 25. One image: take these as starting points, not tuned presets. σ is the standard deviation of the noise in 8-bit units; speed at 1080p on an RTX 5080 (the second row's on the test frame).

| Arguments | PSNR, sigma 15 | PSNR, sigma 25 | fps YUV420 / RGB |
|---|---:|---:|---:|
| defaults (`h=1.2`) | 24.97 dB | 20.98 dB | 565 / 235 |
| `d=0, h=0.6 σ` | 32.44 dB | 28.94 dB | 760 / 235 |
| `a=10, s=3, d=0, h=0.6 σ` | 33.31 dB | 30.53 dB | 121 / 78 |

- `h` is the KNLMeansCL scale (the weight is exp(-mean squared patch difference / h^2), without the paper's -2 sigma^2 offset), so it has to grow with the noise: the default `h=1.2` leaves noise of sigma 3 or more untouched. About 0.6 σ is best on a single frame (`d=0`), about 0.8 σ with `d` ≥ 1 on video, where the neighbouring frames carry their own noise.
- The 21 x 21 window and 7 x 7 patches are the paper's. They gain 0.9 to 1.6 dB on this grainy frame, but within 0.1 dB on 11 cleaner images (DIV2K and crops), at 3 (RGB) to 6 (YUV420) times the cost.
- An earlier version of this table gave `h=1.3 σ` with `d=1` on a clip of one frame: its "neighbours" were the same frame with the same noise, which favours a larger `h`. With `d=0`, `h=1.3 σ` gives 30.21 / 28.07 dB.

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

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |

## Notes

- **Accuracy.** Above 144 dB against the CPU plugin (it is the same model with
  exact box sums); identical from run to run.
- **Frames are uploaded once**, however many windows they are part of. With
  `d=1` and frames rendered in order, the weights of a frame pair are computed
  once for both frames: 396 MiB more at 1080p with the default `a=2`, the first
  thing `memory_limit_mb` takes away (the output is the same without it).
- **Memory.** About `2d + 4` frames of the clip per stream. Large `d`, patches
  or search windows fall back to a slower path with fewer frames resident.

## Pitfalls

- `h` is **not** in sigma units and does not track the noise level by itself;
  retune it when noise changes materially.
- There is no `radius` and no fat intermediate: temporal support is the `d`
  parameter and the result is always normal-height.
- `a` must be smaller than the processed plane width (creation-time error).
- `d` pins `2d + 1` input frames; creation rejects radii whose pinned footprint
  exceeds 2 GiB for the actual frame size.
