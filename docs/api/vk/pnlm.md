# nss_vk.PNLM (preview)

Patchwise non-local means on a Vulkan device: every patch on a grid is
estimated from its most similar patches in a window and in the frames around
it (`d`), and the overlapping estimates are blended. On CPU frames or, on
VapourSynth R80+, on GPU frames.

Algorithm, paper and the reasoning behind the defaults: [`nss.PNLM`](../pnlm.md).
Every argument means what it does for [`nss_cuda.PNLM`](../cuda/pnlm.md).

```python
out = core.nss_vk.PNLM(clip, sigma=3)                        # defaults: d=1, a=2, block_size=6
spatial = core.nss_vk.PNLM(clip, sigma=5, d=0, a=4)
resident = core.nss_vk.PNLM(clip.std.GPUUpload(), sigma=3)   # VapourSynth R80+: stays on the GPU
```

## Signature

```python
core.nss_vk.PNLM(clip clip[, float[] sigma = 3.0, int d = 1, int a = 2,
                 int block_size = 6, int group_size = 16,
                 int memory_limit_mb, int device_id = 0, int num_streams])
```

On VapourSynth R80+ `clip` takes CPU or GPU frames (`vnode:all`;
[GPU frames](README.md#gpu-frames-vapoursynth-r80)). A plane with `sigma=0`
is the source's (shared, not copied). `num_streams` defaults to 3 for `d` up
to 1 and to 1 beyond (more streams did not help there), fewer where the memory
limit requires.

## Cost

As on the CUDA plugin, PNLM's cost grows far more slowly than `NLM`'s with the
search window `a` and the temporal radius `d` (the
[comparison with NLM](../cuda/pnlm.md#cost)): for large windows or several
frames, consider PNLM instead of `NLM`. At the defaults it is bound by its
transfers, so GPU frames and direct transfers help most there.

## Notes

- **Accuracy.** The same bits as the CPU plugin (and `nss_cuda`) on NVIDIA:
  the kernels follow the CPU model operation for operation (every operation
  precise, quotients correctly rounded), checked frame by frame on 252 frames
  of gray, YUV420 and RGB clips (odd and tiny sizes, patches of 4, 6 and 8,
  groups of 2 to 32, `d` up to 3, in and out of order), and on GPU frames.
  Drivers that split `fma` (Mesa) differ in the last bits.
- **Matching** is the square matcher of Davy and Ehret (a workgroup of 16 x 16
  invocations, one per quarter of a patch, sharing each quarter's distance
  among four references), weighting in the same workgroup for windows up to
  `a=4` and groups up to 16, in a second kernel otherwise.
- **Frames are uploaded once** however many windows they are part of (a cache
  of the window's frames shared by the streams, as NLM's).
