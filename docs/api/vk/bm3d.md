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

## Speed

1080p, default arguments with `sigma=3`, RTX 5080 (see the
[overview](README.md#speed)):

| Frames | gray | YUV420 | RGB |
|---|---:|---:|---:|
| VapourSynth R81, CPU frames | 730 | 485 | 245 |
| vapoursynth-classic, direct transfers | 1165 | 756 | 375 |
| R81, GPU frames | 1703 | 1051 | 586 |
| `radius=1`, R81 CPU frames / GPU frames | 720 / 908 | 475 / 585 | 238 / 313 |

`final=1` and `radius=2` are bound by their kernels (gray 1080p `radius=1,
final=1` on classic: 478 fps against `nss_cuda`'s 640).

## Notes

- **Accuracy.** Worst 87.7 dB against the CPU plugin over 426 live cases
  (block, group and step shapes, `ref`, `final`, `chroma`, temporal modes);
  identical from run to run, and the same bits on GPU frames as on CPU frames.
  The sums are fixed point: rolling output gives the same bits whatever
  `rolling_chunk` and the order frames are asked for in, and agrees with the
  legacy intermediate reduced by `VAggregate` to rounding.
- **Model.** BM3D model 3 (orthonormal Haar along the group), as the CPU and
  CUDA plugins. Matching runs on the lane matcher; `nss_cuda`'s square and
  tensor-core matchers are not ported.
- **Memory.** Peak device memory at 1080p `radius=1` (eight threads): gray
  566 MiB, YUV420 839 MiB, RGB 1657 MiB; RGB legacy 1226 MiB.

## Pitfalls

As for [`nss_cuda.BM3D`](../cuda/bm3d.md#pitfalls): `sigma` is in 8-bit
units, and `temporal_mode="legacy"` returns a taller intermediate for
`core.nss.VAggregate`, not a viewable frame.
