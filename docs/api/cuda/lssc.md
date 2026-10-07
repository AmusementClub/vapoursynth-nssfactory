# nss_cuda.LSSC

Learned simultaneous sparse coding on the device. Spatial only: `radius > 0` is rejected, as on the CPU.

The parameters mean what they mean in [`nss.LSSC`](../lssc.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

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

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments).

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
