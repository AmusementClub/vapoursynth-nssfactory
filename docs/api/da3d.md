# nss.DA3D

Data-adaptive dual-domain denoising: a last step on top of another denoiser.
It takes the noisy clip and that denoiser's output (the guide) and, patch by
patch, shrinks the noisy clip's Fourier coefficients where the guide says
there is no signal, on patches shaped by the guide (a bilateral weight around
a fitted plane). It recovers detail and flattens residual noise the first
denoiser left. CPU only for now.

```python
basic = core.nss.BM3D(clip, sigma=20, final=1)      # sigma x 4/3 suits BM3D
out = core.nss.DA3D(clip, basic, sigma=15)          # the noise's true sigma
```

## Signature

```python
core.nss.DA3D(clip clip, clip ref[, float[] sigma = 3.0, int memory_limit_mb])
```

## Parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `clip` | required | — | The noisy clip. |
| `ref` | required | same format, size and length as `clip` | The guide: another denoiser's output for `clip`. DA3D is only as good as it: it keeps what the guide shows and removes what the guide does not. |
| `sigma` | 3.0 | >= 0 | The noise's standard deviation in 8-bit units, per plane: the **true** sigma, not a strength (unlike BM3D, which wants about 4/3 of it). `sigma=0` returns `ref`'s plane. |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. About (planes + 1) x 4 bytes per pixel plus 3 MB: 33 MB for a 1080p RGB frame. |

Planes: an RGB clip with one sigma for all three planes is filtered as one
color image (the paper's color version, after an orthonormal color
transform). Every other plane (Gray, each YUV plane, RGB with different
sigmas) is filtered on its own.

Patch size (64 x 64), the weights (sigma_s = 14, gamma_r = 0.7, gamma_f =
0.8) and the stopping weight (2) are the paper's and fixed
(`include/nss/params/da3d.hpp`).

## Algorithm and paper

Pierazzo and Facciolo, *Data Adaptive Dual Domain Denoising: a Method to
Boost State of the Art Denoising Algorithms*, IPOL 7 (2017), and the authors'
reference implementation (BSD-2-Clause; see NOTICE). Until every pixel has an
aggregation weight of at least 2, the pixel of least weight is taken as the
centre of a 64 x 64 patch; a plane fitted to the guide around it is removed;
a weight mask k from the guide's similarity to the centre (and the distance)
selects the part of the patch that belongs with it; the masked noisy patch is
shrunk in the Fourier domain by exp(-0.8 sigma_f^2 / |G|^2), with G the
guide's coefficients; the result is added back with weight k^2. Where the mask
is too small (weight mass below 10) the guide itself is added.

This implementation:

- **Fixed tiles.** The image is cut into tiles of about 128 x 128, each
  processed on its own, independent of the number of threads (the reference
  makes one tile per thread, so its output changes with the machine). The
  quality is the same within 0.01 dB as one tile for the whole image.
- **Reproducible.** The same bits whatever the SIMD width (checked on AVX2
  and NEON hosts): sums in a fixed lane order, an own radix-4 FFT instead of
  FFTW.
- **Against the reference**, with the same inputs and tiles: the same PSNR
  to 0.001 dB, the outputs 96 to 138 dB apart on five 1080p cases.

## Quality and cost

9 anime frames (1080p, `ds/`) with Gaussian noise added; the guide is
`nss.BM3D` with both stages (`final=1`) at 4/3 of sigma. PSNR gain of DA3D
over its guide, mean (worst to best), and the time for one frame on one
Ryzen 9 5950X core (median):

| sigma | gray: gain | gray: s / frame | RGB: gain | RGB: s / frame |
|---:|---:|---:|---:|---:|
| 2 | +0.19 dB (-0.01 to +0.42) | 3.0 | +0.30 dB (+0.12 to +0.47) | 26 |
| 5 | +0.75 dB (+0.43 to +1.34) | 1.7 | +0.94 dB (+0.60 to +1.33) | 12 |
| 10 | +1.34 dB (+0.74 to +2.02) | 1.1 | +1.87 dB (+1.40 to +2.46) | 8.3 |
| 25 | +2.19 dB (+1.28 to +2.97) | 0.65 | +3.37 dB (+2.57 to +4.37) | 4.3 |

- The gain grows with the noise, and so does the speed: at low sigma the
  masks are narrow and far more patches are needed (gray: 8% of the pixels
  are patch centres at sigma 2, 1.3% at sigma 25; RGB: 37% and 3.6%).
- Anime's flat areas and gradients suit the method; the paper reports
  smaller gains on photographs (+0.1 to +0.7 dB over BM3D).
- RGB costs 3 to 9 times gray: three channels per patch, and the color
  similarity makes the masks narrower, so there are more patches.
- VapourSynth runs frames in parallel; each frame runs on one thread.

## Pitfalls

- Give the true sigma of the noise, also when the guide needed another
  value (BM3D about 4/3 sigma, see [Sigma at the default
  settings](cuda/README.md#sigma-at-the-default-settings)). The model assumes
  white Gaussian noise of that sigma; on real grain or compression noise the
  gain is not measured.
- At sigma 2 or less the gain is small and the cost is high.
- Frame properties: `_NSSDA3DPatches` and `_NSSDA3DGuideOnly` count the
  patches of the frame (all planes) and those that took the guide.
- No GPU version yet (`nss_cuda` has no DA3D).
