# nss_vk.WNNM (preview)

Weighted nuclear norm minimization on a Vulkan device, spatial and temporal:
every group of matched patches is shrunk through the singular values of its
matrix, and the groups are blended. On CPU frames or, on VapourSynth R80+, on
GPU frames.

Algorithm, paper and the reasoning behind the defaults: [`nss.WNNM`](../wnnm.md).
Every argument means what it does for [`nss_cuda.WNNM`](../cuda/wnnm.md).

```python
out = core.nss_vk.WNNM(clip, sigma=25)                       # defaults
hiq = core.nss_vk.WNNM(clip, sigma=25, block_step=4, group_size=16)
temporal = core.nss_vk.WNNM(clip, sigma=25, radius=1)         # finished frames
resident = core.nss_vk.WNNM(clip.std.GPUUpload(), sigma=25)   # VapourSynth R80+: stays on the GPU
```

## Signature

`nss_cuda.WNNM`'s ([signature](../cuda/wnnm.md#signature)) under the
`nss_vk` name, with `num_streams = 1` by default. On VapourSynth R80+ `clip`
and `rclip` take CPU or GPU frames (`vnode:all`); both must be the same kind
([GPU frames](README.md#gpu-frames-vapoursynth-r80)).

## Cost

As on the CUDA plugin, groups above 8 patches cost much more (an
eigendecomposition per group), and `block_step` sets the number of groups
(`1/step^2`). Groups of 9 to 16 patches take parallel Jacobi sweeps, groups of
17 to 32 Householder tridiagonalization and divide and conquer, both on one
workgroup of 32 invocations per group (two groups of up to 16). Those of 17 to
32 cost relatively more here than on `nss_cuda`; settings closer to the paper
([CUDA page](../cuda/wnnm.md#closer-to-the-papers)) are slower here than there.

## Notes

- **Accuracy.** 90–128 dB against the CPU plugin on the reference set, within
  about 1 dB of `nss_cuda`; worst 80.7 dB over 190 live cases (blocks 1 to 16,
  groups 1 to 32, `residual`, `adaptive_aggregation`, `block_step`, gray,
  YUV420 and RGB, `rclip`, `memory_limit_mb`, both temporal modes). Identical
  from run to run, and the same bits on GPU frames as on CPU frames. As on
  `nss_cuda`, groups of 17 to 32 agree less closely, because the divide and
  conquer is accurate to the largest eigenvalue rather than to each one.
- **Matching** is BM3D's ([matching notes](bm3d.md#notes)): the column and
  square matchers where the shape applies (the square one also covers the
  centre frame of a temporal search), the general window search otherwise.
- **Emulated warps.** On a device whose subgroups are narrower than a kernel's
  lanes, the kernels run on the emulated warp layer with the same results
  ([support levels](README.md#device-requirements)).
