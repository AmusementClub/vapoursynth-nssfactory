# nss.LSSC

Learned sparse coding: one dictionary is learned per plane (DCT atoms plus
sampled patches, one K-SVD round), and every overlapping patch is coded on its
own by orthogonal matching pursuit until its residual is at the noise level.
The coefficients are a least-squares fit, not shrunk, so high-contrast detail
(text, line art) keeps its level. Moderate cost.

```python
out = core.nss.LSSC(clip, sigma=25)                    # defaults
fine = core.nss.LSSC(clip, sigma=25, block_step=3)     # denser overlap
```

## Signature

```python
core.nss.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8,
              int block_step = block_size / 2, int radius = 0,
              int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Sets the residual at which a patch's coding stops and the weight of the noisy plane in the average. `sigma=0` bypasses the plane. |
| `block_size` | 8 | {1,2,4,8,16} | Patch edge (power-of-two family only). Dictionary dimension is `block^2`. |
| `block_step` | block / 2 (4) | [1, block] | Reference-patch stride; positions scale as `1/step^2`. The model needs overlap: at `block_step=block` (no overlap) it loses 1.2 to 1.8 dB. |
| `radius` | 0 | 0 only | Kept for the common signature: LSSC has no temporal mode, and `radius > 0` is rejected. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

The dictionary size (256 atoms), the residual factor (1.15), the support cap (48
atoms a patch) and the weight of the noisy plane are internal constants
(`include/nss/params/lssc.hpp`); they are not exposed.

## Algorithm and paper

Mairal, Bach, Ponce, Sapiro & Zisserman, *Non-local Sparse Models for Image
Restoration*, ICCV 2009, without the simultaneous (grouped) coding, and the
sparse-coding denoiser of Elad & Aharon (K-SVD, 2006). This implementation:

- One dictionary per plane: DCT atoms plus sampled patches, then one K-SVD
  round with OMP at sparsity 8 (FP32 correlation screen, ordered FP64
  refinement and residual). This is the repository's own learning step, not
  Mairal's online dictionary learning.
- Every patch on the `block_step` grid, centred on its mean, coded by
  error-constrained Batch-OMP: atoms are added (the largest normalized
  correlation with the residual) until the residual energy is at most
  `m (1.15 sigma)^2`, or 48 atoms. The coefficients are the least-squares fit
  on the chosen atoms.
- The patches are averaged, with the noisy plane added at the weight
  `2.5 (block / step)^2 / sigma` (sigma in 8-bit units), the image term of
  K-SVD denoising. It keeps a low-noise plane close to its input.

The paper's simultaneous coding of similar patches added little in this
repository's paper-track study (`tests/lssc_paper/`, +0.09 dB on average) and
needs the authors' prelearned dictionary.

## Model version 4

Model version 3 coded k-means clusters of patches jointly with an
l1,2 group soft threshold. The threshold shrank every coefficient row by an
amount proportional to sigma: high-contrast patches lost contrast, and white
text on a dark ground darkened by about 4 levels at the default sigma (about
21 at sigma 15). Neither more rounds, more clusters, small nonlocal groups, a
least-squares refit of the kept rows nor a firm or hard row threshold fixed it
without losing the denoising at higher sigma.

Against model 3 at its default step 8, on 11 RGB images (six DIV2K, three
control crops, a synthetic credits frame and a real one), mean (worst) PSNR:

| sigma | 3 | 10 | 25 |
|---|---:|---:|---:|
| version 4 against version 3 | +1.89 (+1.35) dB | +1.27 (+0.83) dB | +0.83 (+0.38) dB |
| images below the noisy input, version 3 | 8 of 11 | 0 | 0 |
| images below the noisy input, version 4 | 0 of 11 | 0 | 0 |

On the synthetic credits frame the white text moves by -0.2 / -2.3 / -12.5
levels at sigma 3 / 10 / 25 (version 3: -4.1 / -13.9 / -29.9).

The arithmetic is FP32 with a fixed order of every sum (it is the CUDA
kernels' order), so the two plugins agree closely. The model semantic version
(`include/nss/contracts.hpp`) is 4; LSSC stamps no frame properties, so
outputs of the two versions are told apart only by the plugin build.

## Defaults rationale and performance

`block=8, step=4, sigma=3`. About 3.2 fps at sigma 3, 5.0 at sigma 10 and
6.3 at sigma 25 for a 1080p gray plane on one Ryzen 9 5950X core (patches need
fewer atoms as the noise grows); version 3 ran at 3.0 fps at its default step 8
and 0.63 at step 4. A denser step costs about `1/step^2`: on
frames small enough for it, `block_step=2` added about 1 dB at sigma 3 in the
evaluation above.

Settings closer to the paper, with their measured quality and GPU speed, are on the [CUDA page](cuda/lssc.md#closer-to-the-papers); the CPU takes the same arguments.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- The patch grid is capped at 16M floats (patches x block^2): at 1080p and
  block 8, `block_step=2` fails at creation ("grid too large"); 3 is accepted.
- No `_NSS*` frame properties on this filter.

## CUDA

`core.nss_cuda.LSSC` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/lssc.md).
