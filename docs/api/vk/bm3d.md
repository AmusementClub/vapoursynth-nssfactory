# nss_vk.BM3D (preview)

Block matching and 3D collaborative filtering on a Vulkan device: spatial,
`ref` (Wiener), temporal (`legacy` and `rolling`), `chroma` (CBM3D) and
`final` (both stages in one call), on CPU frames or, on VapourSynth R80+, on
GPU frames.

Algorithm, paper and the reasoning behind the defaults: [`nss.BM3D`](../bm3d.md).
Every argument means what it does for [`nss_cuda.BM3D`](../cuda/bm3d.md).

```python
basic = core.nss_vk.BM3D(clip, sigma=25)
both = core.nss_vk.BM3D(clip, sigma=25, final=1)               # both stages, estimate kept on the device
temporal = core.nss_vk.BM3D(clip, sigma=25, radius=1)          # finished frames (temporal_mode="rolling")
fat = core.nss_vk.BM3D(clip, sigma=25, radius=1, temporal_mode="legacy")
temporal = core.nss.VAggregate(fat, clip, radius=1)            # the CPU plugin's VAggregate
resident = core.nss_vk.BM3D(clip.std.GPUUpload(), sigma=25)    # VapourSynth R80+: stays on the GPU
```

## Signature

`nss_cuda.BM3D`'s ([signature](../cuda/bm3d.md#signature)) under the
`nss_vk` name, with `num_streams = 1` by default. On VapourSynth R80+ `clip`
and `ref` take CPU or GPU frames (`vnode:all`); both must be the same kind
([GPU frames](README.md#gpu-frames-vapoursynth-r80)).

## Cost

Spatial BM3D is bound by its transfers on CPU frames and by its kernels on GPU
frames; `final=1` and `radius=2` are bound by their kernels. Some search
shapes have faster dedicated matchers (see Matching below): when the shape is
up to you, consider one of them.

## Notes

- **Accuracy.** Worst 87.7 dB against the CPU plugin over 426 live cases
  (block, group and step shapes, `ref`, `final`, `chroma`, temporal modes);
  identical from run to run, and the same bits on GPU frames as on CPU frames.
  The sums are fixed point: rolling output gives the same bits whatever
  `rolling_chunk` and the order frames are asked for in, and agrees with the
  legacy intermediate reduced by `VAggregate` to rounding.
- **Model.** BM3D model 3 (orthonormal Haar along the group), as the CPU and
  CUDA plugins.
- **Matching.** One-channel spatial searches use, in this order:
  - the exact column matcher: blocks 4/8/12/16 at steps 1–3, groups
    8/16/32, except block 4 step 2, block 12 step 3, block 16 group 32 and
    block 16 / group 8 step 3;
  - the square matcher for `block = 2 x step` (2/1, 4/2, 8/4, 12/6, 16/8) and
    the generalized 12/3, 12/4, 16/4 and 8/2, groups 2–32;
  - on devices with 32-wide subgroups, an exact window matcher, any
    `block_step`: `group_size` up to 32 at block 8 with `bm_range` 7 or more
    (the default `block_step=8, bm_range=9` among them) and at block 4 with
    12 or more; `group_size=64` at blocks 2, 4 and 8 with every `bm_range`;
    blocks 12 and 16 with `group_size` up to 32 and `bm_range` 12 or more.

  Column and window give the matches of the matcher they replace bit for bit;
  square matches agree up to near ties. Everything else keeps the lane
  matcher (or, for `group_size=64`, the block fallback). Temporal search matches the centre frame with the square matcher
  or the window matcher where one applies (the window matcher from
  `bm_range` 9 at block 8 and 14 at block 4), then searches the other frames
  predictively from those matches (a no-dedupe variant for `ps_num=1`). The
  tensor-core matcher is not ported.
- **Memory.** Peak device memory at 1080p `radius=1` (eight threads): gray
  566 MiB, YUV420 839 MiB, RGB 1657 MiB; RGB legacy 1226 MiB.

## Pitfalls

As for [`nss_cuda.BM3D`](../cuda/bm3d.md#pitfalls): `sigma` is in 8-bit
units, and `temporal_mode="legacy"` returns a taller intermediate for
`core.nss.VAggregate`, not a viewable frame.
