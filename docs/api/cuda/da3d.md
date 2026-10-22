# nss_cuda.DA3D

DA3D on the device: the CPU model with its transform arithmetic, but the
sums over a patch (the plane fit, the weights' mass and energy, the
averages) are trees instead of the CPU's ordered chains, so the output is
within a level of `nss.DA3D`'s (1080p: 110 to 146 dB apart, PSNR to the
clean image the same) and the patch counts agree to a patch or two. Every
run gives the same bits.

Algorithm, paper, parameters and quality: [`nss.DA3D`](../da3d.md).

```python
guide = core.nss_cuda.BM3D(clip, sigma=20, final=1)
out = core.nss_cuda.DA3D(clip, guide, sigma=15)
```

## Speed

1080p, RTX 5080, one instance, default streams; a frame of the Ufotable test
image with Gaussian noise added and `nss_cuda.BM3D(final=1)` at 4/3 sigma as
the guide (only DA3D timed):

| sigma | YUV420 | RGB |
|---:|---:|---:|
| 3 | 37 | 5.6 |
| 10 | 70 | 14 |

The cost follows the number of patches, which grows as the noise falls and
with color (see the CPU page). The tiles of a frame run in parallel, one
thread block each, the tiles of all its planes in one launch; inside a tile
the patches are chosen one after another.

## Signature

```python
core.nss_cuda.DA3D(clip clip, clip ref[, float[] sigma = 3.0, int memory_limit_mb,
                   int device_id = 0, int num_streams])
```

`num_streams`: frames in flight. Default 3 for a clip filtered plane by plane
(gray, YUV, RGB with different sigmas: two tiles share an SM, and a third
frame keeps the GPU full) and 2 for joint RGB, fewer where the memory limit
requires; more measured no faster.

## Notes

- **Close to the CPU, not bit for bit** (`tests/cuda/test_cuda_da3d_live.py`:
  100 dB or more from the CPU's output, the same PSNR to the clean frame,
  the patch counts within 1%): the 64 x 64 FFT is the CPU's radix-4
  arithmetic in shared memory and the kernels are built without
  contraction; only the sums' order differs (1.15x on gray, owner's choice
  of 2026-10-22).
- **Memory**: about 7 bytes a pixel per stream for a joint RGB clip (the
  tiles' sums), plus the frames; a plane-by-plane clip holds the work of
  every filtered plane at once.
