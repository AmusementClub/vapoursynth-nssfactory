# nss_cuda.BM3D

Block matching and 3D collaborative filtering on the device: spatial, `ref`
(Wiener), temporal, `chroma` (CBM3D) and `final` (both stages in one call).

Algorithm, paper and the reasoning behind the defaults: [`nss.BM3D`](../bm3d.md).

```python
basic = core.nss_cuda.BM3D(clip, sigma=25)
final = core.nss_cuda.BM3D(clip, ref=basic, sigma=25)
both = core.nss_cuda.BM3D(clip, sigma=25, final=1)        # the two lines above, estimate kept on the device
color = core.nss_cuda.BM3D(yuv444, sigma=[25, 15, 15], chroma=1)
# Temporal: finished frames by default (temporal_mode="rolling").
temporal = core.nss_cuda.BM3D(clip, sigma=25, radius=1)
# The CPU plugin's two-step form, for mixing backends or debugging.
fat = core.nss_cuda.BM3D(clip, sigma=25, radius=1, temporal_mode="legacy")
temporal = core.nss.VAggregate(fat, clip, radius=1)       # the CPU plugin's VAggregate
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default | 510 | 285 |
| `radius=1` | 495 | 250 |

Against [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) with
the same search, at 1080p gray: about 1.25x spatial, and 2.7x / 3.3x at
`radius=1` / `2` when frames are rendered in order. With `chroma=1` at YUV444:
243 fps against its 200.

## Signature

```python
core.nss_cuda.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8,
                   int[] group_size = 8, int[] block_step, int[] bm_range = 7,
                   int radius = 0, int[] ps_num, int[] ps_range = 4, int chroma = 0,
                   int final = 0, float[] sigma_basic, int[] block_size_basic,
                   int[] group_size_basic,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

Array-typed parameters take one value per plane; a single value broadcasts.
Omitted `block_step` adapts to `min(8, block_size)` per plane; omitted
`ps_num` adapts to `min(2, group_size)`.

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise standard deviation per plane. Controls threshold/Wiener strength. Note the input is remapped through the BM3D effective-noise profile, so the response tracks the paper's expected behaviour. `sigma=0` bypasses the plane. |
| `block_size` | 8 | {1,2,4,8,12,16,32} | Patch edge. 8 is the general default; 12 exists for high-noise DCT profiles. Larger blocks cost quadratically per patch. |
| `block_step` | min(8, block) | [1, block] | Stride between reference patches. Halving it roughly quadruples positions (2D) — the main quality/speed trade. |
| `group_size` | 8 | {1,2,4,8,16,32,64} | Maximum patches per 3D stack. More matches help at high noise; marginal at low noise. |
| `bm_range` | 7 | [1, 64] | Search window radius (`window = 2*bm_range + 1`). Cost grows quadratically; motion/texture may justify a larger window. |
| `radius` | 0 | [0, 16] | Temporal radius in frames. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
| `ref` | none | clip | External reference clip guiding both stages' matching (paper-style two-stage usage). |
| `chroma` | 0 | {0, 1} | CBM3D for YUV 4:4:4 clips: groups are matched on plane 0 only (of `ref` when given) and all three planes are filtered with them. See "Color (CBM3D)". |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `ps_num` | min(2, group) | [1, group] | Predictive-search seeds kept per temporal step (only used with `radius > 0`). |
| `ps_range` | 4 | [1, 64] | Predictive-search window radius around each seed (temporal mode). |

## Both stages in one call (`final`)

`BM3D(clip, final=1, ...)` runs the basic stage and then the Wiener stage
with the basic estimate as its reference. It gives exactly what the two calls

```python
basic = core.nss_cuda.BM3D(clip, sigma=sigma_basic, block_size=block_size_basic,
                      group_size=group_size_basic, ...)   # finished frames
out = core.nss_cuda.BM3D(clip, ref=basic, sigma=sigma, block_size=block_size,
                    group_size=group_size, ...)
```

give, bit for bit.

| Parameter | Default | Meaning |
|---|---|---|
| `final` | 0 | 1 runs both stages. Cannot be combined with `ref` (which supplies a basic estimate of your own). |
| `sigma_basic` | `sigma` | Noise level of the basic stage, per plane. 0 leaves a plane's estimate as the source. |
| `block_size_basic` | `block_size` | Patch edge of the basic stage. |
| `group_size_basic` | `group_size` | Group size of the basic stage. |

- The other arguments hold for both stages. An omitted `block_step` or
  `ps_num` adapts to each stage's block and group; a given one must fit both.
- The `*_basic` arguments need `final=1`.
- With `radius > 0` the basic stage is temporal too, and its finished frames
  are the reference of the second stage in either temporal mode.
- The basic estimate stays on the device (see Notes below); on the
  CPU plugin `final=1` is the two calls chained.

## Color (CBM3D)

With `chroma=1` the filter follows the color variant of the paper (Dabov, Foi,
Katkovnik & Egiazarian, *Color Image Denoising via Sparse 3D Collaborative
Filtering with Grouping Constraint in Luminance-Chrominance Space*, ICIP 2007):
the groups are found on the luminance plane, where the structure is strongest,
and each plane is collaboratively filtered with those same groups.

- The clip must be YUV 4:4:4. The filter does not convert color: give it
  YUV or an opponent space stored as YUV, and convert back afterwards.
- Matching runs on plane 0 of `ref` when a `ref` is given, of `clip` otherwise.
  In the second stage each plane's Wiener gains come from the same plane of
  `ref`.
- `block_size`, `group_size`, `block_step`, `bm_range`, `ps_num` and
  `ps_range` take their first value for all planes. `sigma` stays per plane.
- A plane with `sigma = 0` is copied. Plane 0 still provides the groups, so
  `sigma=[0, s, s]` filters the chroma planes under luma's guidance.
- Plane 0 is filtered exactly as without `chroma`; planes 1 and 2 differ.
- `radius > 0`, `ref` and both temporal modes work as usual.

It matches once instead of three times, so it is also faster wherever the
matching is a large part of the work.

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** Within the 60 dB gate against the CPU plugin; identical from
  run to run.
- **`final=1`** keeps the basic estimate on the GPU (one upload and one
  download per plane instead of three and two) and is bit for bit the two
  calls; with `temporal_mode="legacy"` it is the two calls chained.
- **Temporal output** agrees with the legacy intermediate reduced by
  `VAggregate` to a few ulp (bit for bit for the shapes that sort their
  patches).
- **Memory.** The smallest `memory_limit_mb` one stream accepts at 4K, gray /
  YUV420 / RGB: about 350 / 370 / 420 MiB spatial, 1140 / 1300 / 1780 at
  `radius=1`, 1590 / 1750 / 2220 at `radius=2`. `ref` adds about a third,
  `final=1` about half.

## Pitfalls

- `sigma` is in 8-bit units even though the clip is float32; 25 means the
  usual "25/255" noise.
- Array parameters are per-plane: `block_size=[8,16]` is legal on YUV.
- With `radius > 0` and `temporal_mode="legacy"` the output is a **taller
  intermediate**, not a viewable frame: pass it through
  `core.nss.VAggregate(out, clip, radius=radius)`. The default mode
  returns finished frames.
