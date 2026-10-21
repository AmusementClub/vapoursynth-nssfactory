# nss_cuda.BM4D

BM4D on the device: the CPU model operation for operation, so the output is
`nss.BM4D`'s bit for bit.

Algorithm, paper, parameters and quality: [`nss.BM4D`](../bm4d.md).

```python
out = core.nss_cuda.BM4D(clip, sigma=5)
```

## Speed

1080p, RTX 5080, one instance; 64 frames of an anime episode with grain of
sigma 5 added:

| Clip | fps |
|---|---:|
| gray | 19.7 |
| YUV420 | 14.2 |

About 19 times the CPU plugin on a 16-thread Ryzen 9 5950X (gray: 1.0 fps).
Asking for one frame computes its 32-frame chunk; the chunk's other frames
come from the cache.

## Signature

```python
core.nss_cuda.BM4D(clip clip[, float[] sigma = 3.0, int memory_limit_mb,
                   int device_id = 0, int num_streams])
```

`num_streams` is accepted for the common signature; chunks run one at a time.

## Notes

- **Same bits as the CPU** on every input checked
  (`tests/cuda/test_cuda_bm4d_live.py`): distances are summed in the CPU's
  order and the closest cubes chosen by (distance, offset), the transforms
  keep the CPU's arithmetic, and the sums are 64-bit fixed point, exact in
  any order, so the device's atomics add up to the CPU's sums.
- **Memory**: the chunk's 48 input and 32 output frames of the largest plane
  on the device, plus about 1 GB of work for a 1080p tile volume.
