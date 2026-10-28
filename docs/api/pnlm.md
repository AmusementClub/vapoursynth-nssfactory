# nss.PNLM

Patchwise non-local means: for every reference patch on a grid, the most
similar patches in a window around it (and, with `d > 0`, in the frames
before and after) are averaged with weights from their distance, the whole
patch is estimated at once, and the overlapping estimates are blended. It is
NLM as Davy and Ehret run it on the GPU: better output than [`NLM`](nlm.md)
at the same window, and on the GPU its kernels are faster than `NLM`'s from
`a = 2` up (the CPU takes about twice `NLM`'s time).

```python
out = core.nss.PNLM(clip, sigma=3)                 # defaults: d=1, a=2, block_size=6
spatial = core.nss.PNLM(clip, sigma=5, d=0, a=4)
luma = core.nss.PNLM(yuv, sigma=[4, 0])            # luma only
```

## Signature

```python
core.nss.PNLM(clip clip[, float[] sigma = 3.0, int d = 1, int a = 2,
              int block_size = 6, int group_size = 16, int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | Noise standard deviation in 8-bit units, one value per plane (the last value repeats). It is the noise's σ itself: the strength is calibrated so that the true σ is about the best setting (see Defaults). `sigma=0` passes the plane through. |
| `d` | 1 | [0, 31] | Temporal radius in frames. The frames `n - d` .. `n + d` of the clip are searched (fewer at the ends of the clip: no frame is repeated). `d=0` is purely spatial. |
| `a` | 2 | [1, 64] | Search radius in pixels: candidates within ±`a` along x and y, in every frame searched. Cost grows as `(2a+1)^2`; quality grows slowly with it. |
| `block_size` | 6 | 4, 6, 8 | Patch side. References sit every `block_size / 2` pixels. 4 keeps the most detail at light noise; 8 is the fastest; 6 is the balance. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `group_size` | 16 | [2, 32] | Patches averaged per reference, the reference included. More only helps with `d >= 2` on still content at stronger noise. |
| `memory_limit_mb` | none | — | Workspace cap; the call fails instead of degrading output. |

## Algorithm and paper

Davy & Ehret, *GPU acceleration of NL-means, BM3D and VBM3D*, J. Real-Time
Image Processing 18 (2021), section 2.1: NL-means as improved by Buades, Coll
& Morel (IPOL 2011), patchwise:

1. References: the `p x p` patches (`p = block_size`) at every `p / 2` pixels,
   and the last row and column of patches at the plane's edge.
2. For each reference, the `group_size - 1` candidates of least squared
   distance within the window (in every frame searched, the reference itself
   excepted), ties broken by frame, row and column.
3. Weights `exp(-max(D / p² - 2σ², 0) / h²)` with `h = 2σ`; the reference
   weighs 1. The patch estimate is the weighted mean of the patches.
4. The flat-patch rule: where the variance of all the matched patches'
   pixels is close to σ² (a flat area), the estimate moves to their plain
   mean.
5. Every output pixel is the mean of the estimates of the patches that cover
   it, weighted by a tent (highest at a patch's centre).

Departures from the paper, all measured on anime frames (σ 1 to 30):

- The weights' `h = 2σ` (the paper: `h = σ`) and a soft flat-patch rule (a
  ramp over a variance of 0.9 to 1.6 σ² instead of a hard threshold at
  1.05 σ²). Both make the result much less sensitive to a wrongly set
  `sigma` at the cost of 0.0 to 0.2 dB at the right one.
- Even patches at a step of half the patch (the paper: 9 x 9 or 7 x 7 at
  about a third), the shape whose distances the GPU shares between
  references; 8 x 8 at step 4 is as good as the paper's 9 x 9 or better
  (+0.00 to +0.09 dB).
- The temporal window (`d`): the paper's NL-means is spatial.

The model (search order, weights, sums and their order) is fixed in
`include/nss/params/pnlm.hpp`, model version 1: x86 (AVX2, AVX-512), arm64
(NEON) and the CUDA plugin give the same bits.

## Defaults rationale and performance

`sigma` is the noise's σ in 8-bit units, as measured or estimated: on 10
anime frames (gray 1080p, Gaussian noise of σ 1 to 30) the best setting was
the true σ within ±25% from σ 3 up, and about 0.65 σ at σ 1. A `sigma` of
half the noise's loses 1.2 dB at σ 2 to 4 dB at σ 10 (up to 6 dB at `a = 8`),
twice the noise's 1.1 to 3.9 dB (see Pitfalls).

Against `NLM` at its best `h`, each at its best strength and setting (PSNR;
`d = 0`: the same `a`, 1 to 8, `NLM`'s best `s`; `d = 1`: `a = 2`,
`block_size=6` against `NLM`'s `s = 4`, on 7-frame clips of the same frames,
still, panning 1 px and 3 px a frame):

| Noise σ | `d = 0` | `d = 1`: still / 1 px / 3 px a frame |
|---|---|---|
| 1 | +0.7 to +0.9 dB | |
| 2 | +0.7 to +1.2 dB | +0.2 / +0.3 / +1.1 dB |
| 3 | +0.8 to +1.7 dB | |
| 5 | +0.6 to +2.1 dB | +0.5 / +0.5 / +1.5 dB |
| 10 | +0.6 to +2.8 dB | +0.7 / +0.8 / +1.0 dB |

The largest gains are at `a = 1`. On edges it leaves 15 to 40% less of the
largest errors (halos, bleeding across lines) and 5 to 20% less error on
average; at σ 10 it softens edges more than `NLM` at small windows. `d = 1`
gains 1.8 to 2.6 dB over `d = 0` on still and slowly moving content and never
lost in the tests.

`block_size`: at `a = 2`, 4 / 6 / 8 give 45.86 / 45.80 / 45.66 dB at σ 3
and 39.32 / 39.49 / 39.36 at σ 10; 4 keeps the most edge contrast up to σ 5
and falls behind above σ 10.

One Ryzen 9 5950X core, 1080p gray, defaults: about 250 ms a frame
(`d = 0`: 135 ms); `NLM` at its defaults takes 113 ms (`d = 0`: 28 ms). The
matching grows with `(2a+1)^2 (2d+1)`, the rest (about 65 ms) does not; the
cost is highest at `block_size = 4`.

## Pitfalls

- A `sigma` far below the noise leaves noise and blotches (half the noise's
  σ: about -2 dB at σ 3, -4 dB at σ 10); far above, it flattens texture and
  gradients. Both cost more than a wrong `h` costs `NLM` from σ 3 up. Measure
  or estimate σ rather than tuning it like `NLM`'s `h`.
- At large windows and light noise, smooth gradients can flatten into steps
  more than with `NLM` (`a = 10`, σ 1: 4.6% of gradient pixels against under
  1%).
- Motion beyond `a` pixels a frame is not matched across frames: `d` then
  helps little (the 3 px a frame pans above, at `a = 2`).
- Every plane with a sigma is filtered on its own (no joint YUV distance as
  in `NLM`'s `channels`); `sigma=[s, 0]` filters luma only.
- Planes up to 8192 x 8192 pixels; a filtered plane must be at least
  `block_size` wide and high. `d` pins up to `2d+1` input frames; creation
  rejects radii whose footprint exceeds 2 GiB.

## CUDA

`core.nss_cuda.PNLM` takes the same arguments and gives the same bits. What
is specific to the device (extra arguments, memory, speed) is in the
[CUDA page](cuda/pnlm.md).
