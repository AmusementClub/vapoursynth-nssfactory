# nss_vk.MCWNNM (preview)

Multi-channel WNNM for RGB on a Vulkan device, spatial and temporal: the
three channels of every group of matched patches are stacked into one matrix,
weighted by their noise, and shrunk through its singular values by ADMM; the
groups are blended, over the outer rounds of `iters`. On CPU frames or, on
VapourSynth R80+, on GPU frames.

Algorithm, paper and the reasoning behind the defaults: [`nss.MCWNNM`](../mcwnnm.md).
Every argument means what it does for [`nss_cuda.MCWNNM`](../cuda/mcwnnm.md).

```python
out = core.nss_vk.MCWNNM(rgb_clip, sigma=[25, 25, 25])        # defaults
quick = core.nss_vk.MCWNNM(rgb_clip, sigma=25, iters=1)        # one round
temporal = core.nss_vk.MCWNNM(rgb_clip, sigma=25, radius=1)    # finished frames
resident = core.nss_vk.MCWNNM(rgb_clip.std.GPUUpload(), sigma=25)  # VapourSynth R80+: stays on the GPU
```

## Signature

`nss_cuda.MCWNNM`'s ([signature](../cuda/mcwnnm.md#signature)) under the
`nss_vk` name, with `num_streams = 1` by default. On VapourSynth R80+ `clip`
and `rclip` take CPU or GPU frames (`vnode:all`); both must be the same kind
([GPU frames](README.md#gpu-frames-vapoursynth-r80)). RGB only.

## Cost

As on the CUDA plugin, `iters` (the outer rounds, two by default) and
`admm_iter` multiply the work, and `block_step` sets the number of groups
(`1/step^2`); `iters=1` is about twice as fast spatially. Groups up to 8
patches take three kernels (the ADMM one invocation a group), groups of 9 to
32 one workgroup of 32 invocations a group (two of up to 16) with parallel
Jacobi sweeps. Channels of equal sigma share their ADMM state (one channel
class instead of up to three). Matching is joint over the three channels and
takes the general search.

## Notes

- **Accuracy.** 134.6-140.9 dB against the CPU plugin on the reference set
  (1080p, `sigma=25`, temporal legacy output through `nss.VAggregate`);
  identical from run to run, and the same bits on GPU frames as on CPU frames.
  The legacy output of more than one round is the last round's slices (as
  `nss_cuda`'s): reduced by `nss.VAggregate` it is the CPU's result.
- **Rounds** relax every window frame's estimate toward its noisy frame by
  `delta` and match and filter the estimates again, on the device; the first
  round matches on `rclip` where one is given.
- **Emulated warps.** On a device whose subgroups are narrower than a kernel's
  lanes, the kernels run on the emulated warp layer with the same results
  ([support levels](README.md#device-requirements)).
