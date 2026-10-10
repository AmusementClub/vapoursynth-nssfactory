# nss_cuda.LSSC

Learned simultaneous sparse coding on the device. Spatial only: `radius > 0` is rejected, as on the CPU.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.LSSC`](../lssc.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.LSSC(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.LSSC(clip, sigma=25, block_step=4)
```

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

## Device parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `device_id` | 0 | device index | CUDA device the instance runs on. One instance uses one device; `core.nss_cuda.Backend(device_id)` reports whether it is supported. |
| `num_streams` | 1 | [1, 16] | Frames (or temporal chunks) the instance has in flight on the device at once. Each stream owns its device buffers, so memory grows with it. |
| `memory_limit_mb` | none | > 0 | Caps the instance's device and pinned host memory. The internal batches are fitted to it and the output does not change; a limit that cannot hold the streams is a creation error. Without it the filter takes what its plan needs. |

How the streams and the memory limit behave across filters is in the [shared notes](README.md#shared-by-every-filter).

## On the device

- **Device-resident.** Patch packing, k-means, the dictionary (DCT atoms,
  sampled patches, one K-SVD round), the per-cluster sparse coding, the
  reconstruction and the aggregation run on the device. The host only
  sequences the steps and keeps index bookkeeping (cluster member lists and
  the k-means round control).
- **Numerics.** The K-SVD sparse codes use FP64 correlations and solves as on
  the CPU; an atom's rank-one update takes the dominant singular triplet from
  an FP64 power iteration instead of a full SVD. Everything else is FP32.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (139–141 dB on the frozen references) and run-to-run identical.
- **Memory.** The coefficients cost `min(256, patches)` floats per grid patch,
  so a dense `block_step` on a large frame takes that much device memory.
- **Speed.** The dictionary update is sequential over the atoms, so the gain
  over the CPU is small: about 1–2x the CPU plugin at 16 threads for 1080p.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- `radius > 0` is rejected: LSSC has no temporal mode.
- No `_NSS*` frame properties on this filter.
