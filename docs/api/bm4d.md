# nss.BM4D

Block matching and 4D filtering: BM3D with cubes. The clip is treated as a
volume (frames as the third dimension); similar 4 x 4 x 4 (then 5 x 5 x 5)
cubes, searched up to 7 pixels and 7 frames away, are stacked into groups and
filtered together in two stages, hard thresholding then Wiener. Held frames
and still or slowly moving content are where it gains most. Heavy on the CPU;
[`nss_cuda.BM4D`](cuda/bm4d.md) gives the same quality (PSNR) about 40 times faster.

```python
out = core.nss.BM4D(clip, sigma=5)        # the noise's true sigma
```

## Signature

```python
core.nss.BM4D(clip clip[, float[] sigma = 3.0, int memory_limit_mb])
```

## Parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | The noise's standard deviation in 8-bit units, per plane: the **true** sigma (unlike BM3D, which wants about 4/3 of it). `sigma=0` passes the plane through. |
| `memory_limit_mb` | none | — | Caps the working memory; fails instead of degrading. |

Each plane is filtered on its own (luma and chroma of YUV, or R, G and B).
The cube sizes, steps, group sizes (16, 32), search range (7) and matching
thresholds are the paper's defaults and fixed (`include/nss/params/bm4d.hpp`).

## Algorithm and paper

Maggioni, Katkovnik, Egiazarian and Foi, *Nonlocal Transform-Domain Filter for
Volumetric Data Denoising and Reconstruction*, IEEE TIP 22(1), 2013, with the
parameters of the authors' published default profile. Written from the paper
(the authors' software is not open source and is not used):

- Hard stage: reference cubes of 4 x 4 x 4 every 3 voxels along each axis;
  the 16 closest cubes within +-7 along every axis whose mean squared
  difference is at most 2.9527 sigma^2; a group transform (Haar along the
  group, 3D DCT of each cube), coefficients below 2.7 sigma set to zero,
  aggregation weighted by the coefficients kept and a Kaiser window.
- Wiener stage: the same on the basic estimate with 5 x 5 x 5 cubes, 32 per
  group and threshold 0.7689 sigma^2; the noisy group shrunk by the
  empirical Wiener filter.
- Departure: the hard stage's cube transform is the DCT (the authors' profile
  uses the bior1.5 wavelet there).

Video is not a medical volume: there is no motion compensation (that is
V-BM4D, a different method). Objects moving fast within the 4 or 5 frames of
a cube match poorly.

## Volumes

The clip is filtered in chunks of 32 frames, each with up to 8 frames of the
neighbouring chunks on each side, and every plane in tiles of about 256 x 256
with 16 pixels of overlap; only each chunk's frames and each tile's pixels are
kept. The output does not depend on the order frames are asked for or on the
number of threads; asking for one frame computes its whole chunk (on all of
VapourSynth's threads) and keeps it for the chunk's other frames. Scene
changes are not marked: the matching threshold keeps cubes from matching
across a cut.

## Quality

One anime episode (1080p web source taken as clean), 12 segments of 16
frames (384 x 256 crops) chosen from still to fast action, Gaussian noise of
sigma 5 and 10 added. PSNR of BM4D (the prototype this filter was checked
against, on the whole segment) minus `nss.BM3D(final=1, radius=2,
temporal_mode="rolling")` at 4/3 sigma:

| Motion (mean frame difference) | Segments | sigma 5 | sigma 10 |
|---|---:|---|---|
| still (0.01) to slow (1.8) | 3 | +1.7 to +5.4 dB | +2.5 to +6.1 dB |
| with a scene cut (3.1, 5.0) | 2 | +2.9, +4.5 dB | +3.6, +4.8 dB |
| steady pans (5.9 to 13.4) | 4 | +0.2 to +1.6 dB | +0.3 to +2.4 dB |
| fast action (17 to 46) | 3 | -0.6 to -1.3 dB | -0.15 to -1.0 dB |

On the pixels that move, BM4D gains as well, except in fast action and in
one nearly still segment with a small moving object (-0.7 / -0.3 dB). Use
temporal BM3D for fast action.

## Speed and memory

1080p gray, one Ryzen 9 5950X with 16 threads: 1.2 frames per second (about
6.6 s of one core per frame). YUV420 adds half for the chroma planes. Each
worker thread holds about 100 MB, and the filter keeps up to two finished
chunks (32 frames each) plus the 48 input frames of the chunk being
computed: about 2 GB at 1080p with 16 threads.

## Pitfalls

- Give the true sigma. The filter assumes white Gaussian noise.
- The first frame of each chunk takes the whole chunk's time; later frames of
  the chunk are immediate.
- Fast action loses to temporal BM3D (see Quality).
- On the GPU, [`nss_cuda.BM4D`](cuda/bm4d.md) gives the same quality (1080p gray 40 fps on an RTX 5080).
