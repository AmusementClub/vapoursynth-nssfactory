# nss.* API Guide

`libnss` is a VapourSynth CPU plugin (namespace `nss`) implementing eight classical
non-local self-similarity (NSS) denoisers. Each page below documents one filter:
its call signature, parameters in the order you will usually touch them
(primary first, secondary/solver knobs last), the paper it implements and where
this implementation deliberately departs from it, the defaults and why they were
chosen, measured speed, diagnostics, and pitfalls.

## Choosing an algorithm

Measured on a single Emerald Rapids core, 1920x1080 GRAYS float, synthetic sigma=3
(MCWNNM processes all three RGB planes). Absolute numbers scale roughly with pixel
count; treat them as cost tiers, not promises.

| Filter | Cost (ms/frame) | fps | Temporal (radius>0) | Best for |
|---|---:|---:|---|---|
| [BM3D](bm3d.md) | ~59 | ~17 | yes | General-purpose baseline; fast |
| [NLM](nlm.md) | ~131 | ~7.6 | yes (d) | Classic non-local means; light noise |
| [WNNM](wnnm.md) | ~131 | ~7.6 | yes | Stronger texture retention than BM3D |
| [NCSR](ncsr.md) | ~268 | ~3.7 | yes | Sparse-coding quality at moderate cost |
| [LSSC](lssc.md) | ~369 | ~2.7 | yes | Structured sparse coding, film grain |
| [TWSC](twsc.md) | ~1337* | ~0.75* | yes | Real-world noise, three-weight ADMM |
| [NLH](nlh.md) | ~6621* | ~0.15* | yes | Blind real-world noise (sigma estimated) |
| [MCWNNM](mcwnnm.md) | ~2326 | ~0.43 | yes | RGB color denoising, joint channels |

\* TWSC and NLH rows were measured at the profiler's lighter geometry
(`block_step=8, group_size=8, iters=2`); their true defaults are substantially
heavier (see the per-page performance notes). All other rows are exact defaults.

## Conventions shared by every filter

- **Input**: constant-format 32-bit float Gray, YUV, or RGB clips only. Integer
  or variable-format input is rejected at creation time.
- **`sigma` units**: always the 8-bit noise standard deviation (0–255 scale),
  one value per processed plane; a single value broadcasts. `sigma=0` bypasses
  that plane (exact identity). NLH estimates sigma per frame when it is omitted;
  TWSC does the same with `estimate_sigma=1`. All other filters require an
  explicit sigma and treat it as known.
- **`radius > 0` (temporal)**: the filter then returns the *weighted
  intermediate* (a taller frame stack; numerator/denominator per temporal
  slice) for an explicit `VAggregate` call. The device plugin (`nss_cuda`)
  returns finished frames by default instead; see each filter's CUDA section. With `radius = 0` every filter returns a normal-height frame.
- **`rclip` / `ref`**: an optional reference clip (same format/size) that guides
  matching; denoising still targets `clip`.
- **`bm_range` vs `search_window`**: `search_window = 2 * bm_range + 1`.
  Passing both is an error. BM3D/NLM-family filters expose `bm_range` only;
  TWSC/NLH accept either.
- **`memory_limit_mb`**: caps internal workspace; exceeding it fails the call
  rather than silently degrading output.
- **Diagnostics**: NLH and TWSC stamp per-frame `_NSS*` frame properties
  (resolved block/group/step/window/iterations, sigma, model version, solver
  stats). Other filters do not currently emit frame properties.
- **Multi-threading**: all parallelism is frame-level and owned by VapourSynth;
  the kernels are single-threaded by contract. The numbers above are per-core.
