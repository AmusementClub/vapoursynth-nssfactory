# nss_cuda.DA3D

DA3D on the device: the CPU model operation for operation, so the output is
`nss.DA3D`'s bit for bit (and so are the `_NSSDA3DPatches` /
`_NSSDA3DGuideOnly` counts).

Algorithm, paper, parameters and quality: [`nss.DA3D`](../da3d.md).

```python
guide = core.nss_cuda.BM3D(clip, sigma=20, final=1)
out = core.nss_cuda.DA3D(clip, guide, sigma=15)
```

## Speed

1080p, RTX 5080, one instance, two streams; a frame of the Ufotable test
image with Gaussian noise added and `nss_cuda.BM3D(final=1)` at 4/3 sigma as
the guide (only DA3D timed):

| sigma | YUV420 | RGB |
|---:|---:|---:|
| 3 | 16 | 4.9 |
| 10 | 36 | 12 |

The cost follows the number of patches, which grows as the noise falls and
with color (see the CPU page). The tiles of a frame run in parallel, one
thread block each; inside a tile the patches are chosen one after another.

## Signature

```python
core.nss_cuda.DA3D(clip clip, clip ref[, float[] sigma = 3.0, int memory_limit_mb,
                   int device_id = 0, int num_streams])
```

`num_streams`: frames in flight (default 2 where the memory limit allows);
more than two measured no faster.

## Notes

- **Same bits as the CPU** on every input checked
  (`tests/cuda/test_cuda_da3d_live.py`): the 32-lane sums are warp chains,
  the 64 x 64 FFT is the CPU's radix-4 arithmetic in shared memory, and the
  kernels are built without contraction.
- **Memory**: about 7 bytes a pixel per stream for a joint RGB clip (the
  tiles' sums), plus the frames.
