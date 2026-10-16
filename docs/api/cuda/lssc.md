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
| default | 190 | 125 |

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

- **Accuracy.** 137–141 dB against the CPU plugin on the reference set (the
  sparse coding runs in FP32 where the CPU uses FP64 in places); identical
  from run to run.
- **Memory.** The codes take `min(256, patches)` floats per patch, so a dense
  `block_step` on a large frame takes that much GPU memory.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- `radius > 0` is rejected: LSSC has no temporal mode.
- No `_NSS*` frame properties on this filter.
