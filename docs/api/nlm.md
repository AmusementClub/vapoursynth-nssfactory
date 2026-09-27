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
| `a` | 2 | [1, plane width) | Search radius in pixels around the anchor. Cost grows as `(2a+1)^2`; quality saturates quickly. |
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
