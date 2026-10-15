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
  reconstruction and the aggregation run on the device, the k-means rounds and
  cluster member lists included. The host only queues the steps.
- **Numerics.** Everything is FP32. The K-SVD sparse codes (OMP) take their
  correlations and solves in FP32 where the CPU uses FP64, and an atom's
  rank-one update takes the dominant singular triplet from an FP32 power
  iteration instead of the CPU's FP32 SVD. The group soft threshold's row
  norms are summed by tiles of a cluster's members, not in the CPU's patch
  order.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (137–141 dB on the frozen references) and run-to-run identical.
- **Memory.** The coefficients cost `min(256, patches)` floats per grid patch,
  so a dense `block_step` on a large frame takes that much device memory.
  Without a limit (or where the limit leaves room) a stream also holds up to
  two more sets of its input-side buffers (input plane, patches, assignments,
  dictionary, k-means scratch: about 17 MB each at 1080p) so that the next
  planes are clustered while one is coded, and frames are staged in pinned
  host memory, a plane-sized region per denoised plane, three frames in
  flight for gray (25 MB at 1080p) and two for RGB (50 MB).
- **Speed.** On an RTX 5080 with one stream (the default; more streams add
  nothing), 2026-10-15: gray 1080p 340 fps, RGB 1080p 111, `block_step=4` 83;
  14–22x the CPU plugin at 16 threads.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- `radius > 0` is rejected: LSSC has no temporal mode.
- No `_NSS*` frame properties on this filter.
