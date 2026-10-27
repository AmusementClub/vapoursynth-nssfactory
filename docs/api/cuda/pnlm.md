# nss_cuda.PNLM

Patchwise non-local means on the device, with the same bits as
[`nss.PNLM`](../pnlm.md) (the CPU model operation for operation: the same
search order, sums and exponential). Every run gives the same bits.

Algorithm, paper, parameters and quality: [`nss.PNLM`](../pnlm.md).

```python
out = core.nss_cuda.PNLM(clip, sigma=3)               # defaults: d=1, a=2, block_size=6
fast = core.nss_cuda.PNLM(clip, sigma=3, block_size=8)
```

## Speed

1080p, RTX 5080, one instance, default streams; the method of the
[overview](README.md#speed) (nine test images with noise of sigma 3, median),
with `NLM` measured in the same session:

| Call | YUV420 | RGB |
|---|---:|---:|
| `PNLM`, defaults (`d=1`, every plane) | 470 | 236 |
| `PNLM`, `sigma=[3, 0]` (luma only) | 606 | |
| `PNLM`, `d=0` | 480 | 239 |
| `NLM`, defaults (`d=1`; YUV: luma only) | 711 | 269 |

At the defaults both filters are bound by the frames' copies and the host,
not the kernels: PNLM filters every plane by default, `NLM` only luma. With
larger windows `NLM`'s kernels take over (gray, constant source: `a=4`
PNLM 528 against `NLM` 199 fps, `a=8` 483 against 52). GPU time of the
kernels a frame (gray, `nsys`):

| | `a=2` | `a=4` | `a=8` |
|---|---:|---:|---:|
| `PNLM` `d=0`, `block_size` 4 / 6 / 8 | 0.35 / 0.33 / 0.24 ms | 0.59 / 0.45 / 0.33 | 1.29 / 0.81 / 0.56 |
| `PNLM` `d=1`, `block_size` 4 / 6 / 8 | 0.58 / 0.48 / 0.35 | 1.23 / 0.80 / 0.62 | 2.74 / 1.67 / 1.20 |
| `NLM` `s=4`, `d=0` / `d=1` | 0.47 / 1.01 | 1.27 / 5.01 | 4.69 / 18.8 |

How: the references sit every `block_size / 2` pixels, so a patch is four
quarters each shared by four references (Davy and Ehret's square matcher,
JRTIP 2021, section 4.1.4): a block of 16 x 16 threads computes each
quarter's distance once per displacement and 15 x 15 references add their
four. For windows up to `a=4` and groups up to 16 the same block then
weights its references' patches from shared memory, so the match list never
goes through device memory; otherwise the matches go out as 8-byte keys and
a second kernel weights them. A gather forms every output pixel from the
estimates that cover it (no atomics).

## Signature

```python
core.nss_cuda.PNLM(clip clip[, float[] sigma = 3.0, int d = 1, int a = 2,
                   int block_size = 6, int group_size = 16, int memory_limit_mb,
                   int device_id = 0, int num_streams])
```

`num_streams`: frames in flight. Default 3, fewer where the memory limit
requires (1080p `d=1`, one / two / three streams: gray 452 / 513 / 536 fps,
YUV420 304 / 344 / 357, RGB 157 / 174 / 182).

## Notes

- **The same bits as the CPU** (`tests/cuda/test_cuda_pnlm_live.py`: gray,
  YUV420 and RGB, odd sizes, every `block_size`, both weighting paths,
  groups of 2 to 32, `d` 0 to 3, with one stream and three and frames out of
  order). The kernels are built without contraction.
- **Frames**: the frames a request searches stay on the device in a cache
  shared by the streams, so with `d > 0` each frame is uploaded once.
- **Memory** at 1080p, defaults: about 325 MiB for gray, 440 MiB for
  YUV420, 780 MiB for RGB (a cache of eight frames, five output stages, and
  per stream the estimates of every patch, four times the filtered planes).
  Under `memory_limit_mb` the stages and the cache shrink first: the least
  that runs is 103 / 139 / 246 MiB with one stream. `a` above 4 or
  `group_size` above 16 adds the match keys: a fifth of a plane per match at
  `block_size=6` (a half at 4, an eighth at 8).
- **Device code**: one 16 x 16 thread block per 15 x 15 references; the
  matcher holds 106 to 108 registers at `block_size=8` (two blocks an SM),
  fewer at 6 and 4.
