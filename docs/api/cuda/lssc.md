# nss_cuda.LSSC

Learned sparse coding on the device. Spatial only: `radius > 0` is rejected,
as on the CPU.

Algorithm, paper and the reasoning behind the defaults: [`nss.LSSC`](../lssc.md).

```python
out = core.nss_cuda.LSSC(clip, sigma=25)                    # defaults
fine = core.nss_cuda.LSSC(clip, sigma=25, block_step=3)
```

## Cost

The cost of a patch grows with the atoms it needs, so it depends on the
content and the noise: grainy, text-heavy frames at low sigma are the
slowest.

## Closer to the papers

The defaults are fast settings. Settings closer to the paper, with their quality; they are much slower than the defaults. Measured on one 1080p RGB frame (an anime credits frame with its own film grain) with Gaussian noise added, against the frame without it; the noisy input is 24.97 dB at sigma 15 and 20.98 dB at sigma 25. One image: take these as starting points, not tuned presets. σ is the standard deviation of the noise in 8-bit units.

| Arguments | PSNR, sigma 15 | PSNR, sigma 25 |
|---|---:|---:|
| defaults, `sigma=σ` | 31.86 dB | 28.75 dB |
| `block_step=3, sigma=σ` | 32.19 dB | 29.17 dB |

- The defaults are close to the paper's sparse coding already (overlapping patches, error-constrained OMP); a denser step adds a little. Model version 3 gave 29.15 / 26.28 dB here.

## Signature

```python
core.nss_cuda.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = block_size / 2, int radius = 0,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Sets the residual at which a patch's coding stops and the weight of the noisy plane in the average. `sigma=0` bypasses the plane. |
| `block_size` | 8 | {1,2,4,8,16} | Patch edge (power-of-two family only). Dictionary dimension is `block^2`. |
| `block_step` | block / 2 (4) | [1, block] | Reference-patch stride; positions scale as `1/step^2`. The model needs overlap. |
| `radius` | 0 | 0 only | Kept for the common signature. LSSC has no temporal mode: `radius > 0` is rejected. |

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |

## Notes

- **How it runs.** Each patch is coded by one warp (patches taken from a
  counter, so busy patches do not hold up the rest); the correlations of a
  batch of patches with the dictionary come from one tiled product and stay
  in L2. The arithmetic and the order of every sum are the CPU's.
- **Accuracy.** 97–150 dB against the CPU plugin on the reference set; the
  dictionary's K-SVD round runs in FP32 on the device, so near-ties between
  atoms can resolve differently. Identical from run to run on a given GPU.
- **Memory.** About 220 MB per stream at 1080p gray with the defaults: the
  patches and their reconstructions, the correlations of a batch of 32768
  patches, and a coding scratch of `48 x min(256, patches)` floats per
  coding warp. Under a tight `memory_limit_mb` fewer warps code (slower, the
  same output).

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- `radius > 0` is rejected: LSSC has no temporal mode.
- No `_NSS*` frame properties on this filter.
