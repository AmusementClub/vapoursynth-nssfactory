# nss_cuda.PNLM

Patchwise non-local means on the device, with the same bits as
[`nss.PNLM`](../pnlm.md) (the CPU model operation for operation: the same
search order, sums and exponential). Every run gives the same bits.

Algorithm, paper, parameters and quality: [`nss.PNLM`](../pnlm.md).

```python
out = core.nss_cuda.PNLM(clip, sigma=3)               # defaults: d=1, a=2, block_size=6
fast = core.nss_cuda.PNLM(clip, sigma=3, block_size=8)
```

## Cost

At the defaults PNLM, like `NLM`, is mostly bound by the frames' copies; it
filters every plane by default (`NLM` only luma), so `sigma=[3, 0]` is cheaper
on YUV when only luma needs it.

PNLM's cost grows far more slowly than `NLM`'s with the search window `a` and
the temporal radius `d`: `NLM` computes a distance for every pixel at every
displacement, about `(2a+1)^2 (2d+1)` of them a pixel, while PNLM computes one
quarter distance a reference position (every `block_size / 2` pixels each way:
a ninth of the pixels at `block_size=6`) and shares it between four
references. **For large windows or several frames (`a` 4 or more, `d` 1 or
more) consider PNLM instead of `NLM`.** `block_size` 8 is cheaper still (fewer
references), 4 dearer.

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

`num_streams`: frames in flight. Default 3 for `d` up to 1 and 2 beyond,
fewer where the memory limit requires.

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
