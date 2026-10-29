# nss.* API Guide

`libnss` is the CPU plugin (namespace `nss`). Each page documents one filter:
its signature, the parameters (the ones you usually touch first), the paper
and where this implementation departs from it, the defaults, speed and
pitfalls. The GPU plugins have the same filters and arguments; see the
[CUDA reference](cuda/README.md) and the [Vulkan reference](vk/README.md)
(a preview: BM3D, NLM and PNLM so far, with VapourSynth R80+ GPU frames).

## Choosing a filter

One CPU core (Emerald Rapids), 1920x1080 gray, sigma 3, default arguments
(MCWNNM: all three planes of an RGB clip). Treat the numbers as cost tiers:
they scale with the pixel count, and VapourSynth runs frames in parallel on
more cores.

| Filter | fps | Temporal | Good for |
|---|---:|---|---|
| [BM3D](bm3d.md) | ~17† | yes | General use; the fastest |
| [NLM](nlm.md) | ~7.6 | yes (`d`) | Light noise |
| [PNLM](pnlm.md) | ~4.0§ | yes (`d`) | Light noise; edges and line art |
| [WNNM](wnnm.md) | ~7.6 | yes | Texture and edge retention |
| [NCSR](ncsr.md) | ~3.7 | yes | Sparse-coding quality at moderate cost |
| [LSSC](lssc.md) | ~2.7‡ | no | Structured noise, film grain, text |
| [TWSC](twsc.md) | ~0.75 | yes | Real-world noise |
| [NLH](nlh.md) | ~0.15* | yes | Real-world noise of unknown level |
| [MCWNNM](mcwnnm.md) | ~0.43 | yes | RGB color noise |

[BM4D](bm4d.md) (about 0.15 fps, temporal by construction) treats the
clip as a volume of 3D cubes; it beats temporal BM3D on still and slowly
moving content, and is the slowest. [DA3D](da3d.md) is not a denoiser on its own: it takes the noisy clip and
another filter's output and returns a refined result (see its page for its
gain and cost).

\* NLH measured with `block_size=8, block_step=8` in both stages and a given
sigma; its default (smaller steps, blind estimation) is much heavier (see its
page).

† BM3D measured with `bm_range=7`; the current default is 9, about 1.6x the
candidates of the matching step.

§ PNLM measured on a Ryzen 9 5950X core: NLM 8.8 fps; spatial (`d=0`) PNLM
7.4 fps against NLM's 35.7. On the GPU it is the faster of the two from `a=2` up.

‡ LSSC measured with model version 3 at `block_step=8`. Version 4 at its
default `block_step=4` took about 7% less time at sigma 3 and about half at
sigma 25 on a Ryzen 9 5950X core.

## Conventions

- **Input**: constant-format 32-bit float Gray, YUV or RGB. Integer or
  variable-format input is rejected when the filter is created.
- **`sigma`**: the noise standard deviation in 8-bit units (0–255), one value
  per plane; a single value applies to all. `sigma=0` passes the plane through
  unchanged (DA3D: passes its guide's plane through). NLH estimates sigma per frame when it is omitted, TWSC with
  `estimate_sigma=1`; the other filters take it as given. It sets the
  filter's strength, which matches the noise's σ only for some filters and
  settings: at the defaults BM3D wants about 4/3 σ, NCSR 1.8 σ, WNNM 2.2 σ,
  MCWNNM 2.5 σ and TWSC 3.4 σ, PNLM σ itself, and NLM `h` about 0.6 to
  0.8 σ (the models
  are the same on both backends; see
  [Sigma at the default settings](cuda/README.md#sigma-at-the-default-settings)).
- **`rclip` / `ref`**: an optional clip of the same format and size that guides
  the matching; denoising still applies to `clip`.
- **`bm_range` and `search_window`**: `search_window = 2 * bm_range + 1`.
  TWSC and NLH accept either (not both); the others take `bm_range`.
- **`memory_limit_mb`**: caps the filter's working memory; without it there is
  no cap. Exceeding it is an error, never a silent change of output.
- **Frame properties**: NLH and TWSC stamp per-frame `_NSS*` properties (the
  resolved shapes, sigma, model version, solver statistics), DA3D its patch
  counts; the other filters stamp none.
- **Threads**: VapourSynth runs frames in parallel; each frame is computed on
  one thread.

## Temporal output

With `radius > 0` a filter also uses the `radius` frames on each side
(BM3D, WNNM, MCWNNM, NCSR, TWSC, NLH). `temporal_mode` chooses what it
returns:

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `temporal_mode` | `"legacy"` | legacy / rolling | `"legacy"` returns the weighted intermediate (numerator and denominator slices in a taller frame) for `VAggregate`; `"rolling"` returns finished frames. |
| `rolling_chunk` | 4 | [1, 64] | Frames computed together in rolling mode. |
| `rolling_cache_chunks`, `rolling_cache_limit` | 1 | [1, 64] | Finished chunks kept for later frame requests in rolling mode. Give one of the two. |

```python
fat = core.nss.WNNM(clip, sigma=5, radius=1)                            # intermediate
out = core.nss.VAggregate(fat, clip, radius=1)                          # finished frames
out = core.nss.WNNM(clip, sigma=5, radius=1, temporal_mode="rolling")   # the same in one call
```

On the CPU only BM3D computes rolling chunks itself. For WNNM, MCWNNM, NCSR,
TWSC and NLH, `"rolling"` is the intermediate reduced by `VAggregate` inside
the call: same result, same cost and memory as the two calls above, and
`rolling_chunk` and the cache arguments are checked but change nothing.

NLM is temporal through its own `d` argument and always returns finished
frames; BM4D is temporal by construction (no `radius`) and returns finished
frames; LSSC has no temporal mode. The GPU plugin returns finished frames by
default and rolls every temporal filter on the device.

### VAggregate

```python
core.nss.VAggregate(clip clip, clip src[, int radius = 0, int[] planes,
                    int allow_legacy = 0, int memory_limit_mb])
```

| Parameter | Default | Meaning |
|---|---|---|
| `clip` | required | The intermediate: a filter's output with `radius > 0` in legacy mode, from either plugin. Its height is `src`'s x (2 x radius + 1) x 2. |
| `src` | required | The clip the intermediate was computed from; it gives the output format and the planes that are not aggregated. |
| `radius` | 0 | The radius the intermediate was made with (checked against its frame properties). |
| `planes` | all | Planes to aggregate; the others are copied from `src`. |
| `allow_legacy` | 0 | 1 also accepts an intermediate without identity properties (old builds). |
| `memory_limit_mb` | none | Caps the working memory. |

The intermediate records its model, model version, radius and layout as
frame properties; one made by a different model version is rejected and must
be regenerated.
