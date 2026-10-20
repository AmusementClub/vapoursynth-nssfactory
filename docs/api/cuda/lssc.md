# nss_cuda.LSSC

Learned simultaneous sparse coding on the device. Spatial only: `radius > 0` is
rejected, as on the CPU.

Algorithm, paper and the reasoning behind the defaults: [`nss.LSSC`](../lssc.md).

```python
out = core.nss_cuda.LSSC(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.LSSC(clip, sigma=25, block_step=4)
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default | 205 | 145 |

These are model version 3 figures (the preconditioned FISTA coding, see
[`nss.LSSC`](../lssc.md#model-version-3-the-cluster-coding)). Its 12 rounds read
the codes of the two previous rounds; where those do not fit half of the GPU's
L2 cache (1080p and above on an RTX 5080), the rounds run over batches of whole
clusters so that a batch's codes stay in L2. The output does not depend on the
batches.

## Signature

```python
core.nss_cuda.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int radius = 0,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Drives the sparse-coding regularization. `sigma=0` bypasses the plane. |
| `block_size` | 8 | {1,2,4,8,16} | Patch edge (power-of-two family only). Dictionary dimension is `block^2`. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. The main quality/speed trade. |
| `radius` | 0 | 0 only | Kept for the common signature. LSSC has no temporal mode: `radius > 0` is rejected. |

The dictionary size (256 atoms) and cluster count (64) are internal constants,
capped by the actual patch count; they are not exposed.

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |

## Notes

- **Accuracy.** On GPUs of compute capability 8.0 or newer the coding
  round's products run on TF32 tensor cores: 103–112 dB against the CPU
  plugin (model version 3 references), with the same denoising quality to
  1e-4 dB. Compute capability 7.5 keeps FP32 products: 137.0–140.8 dB (the
  sparse coding runs in FP32 where the CPU uses FP64 in places). Identical
  from run to run on a given GPU. The frozen C1 references (`tests/data/cuda_refs_v1.json`)
  are version 2 outputs; regenerate LSSC references before comparing.
- **Memory.** The codes take `2 min(256, patches)` floats per patch (the
  current and the previous round), so a dense `block_step` on a large frame
  takes that much GPU memory: about 66 MB at 1080p with the default step, 264 MB
  with `block_step=4`.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- `radius > 0` is rejected: LSSC has no temporal mode.
- No `_NSS*` frame properties on this filter.
