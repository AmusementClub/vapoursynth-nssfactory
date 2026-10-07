# nss.BM3D

The reference block-matching 3D filter: similar patches are grouped into 3D
stacks, collaboratively filtered in a transform domain (hard thresholding in
the Basic stage, Wiener filtering in the second stage), and aggregated back.
Fast, mature, and the quality baseline every other filter here is compared
against.

```python
out = core.nss.BM3D(clip, sigma=25)                       # spatial default
fine = core.nss.BM3D(clip, sigma=25, block_size=8, block_step=4, bm_range=10)
temporal = core.nss.BM3D(clip, sigma=25, radius=2)        # see radius note below
color = core.nss.BM3D(yuv444, sigma=[25, 15, 15], chroma=1)  # CBM3D: groups from luma
both = core.nss.BM3D(clip, sigma=25, final=1)             # basic + Wiener stage in one call
```

## Signature

```python
core.nss.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8,
              int[] group_size = 8, int[] block_step, int[] bm_range = 7,
              int radius = 0, int[] ps_num, int[] ps_range = 4, int chroma = 0,
              int final = 0, float[] sigma_basic, int[] block_size_basic,
              int[] group_size_basic, int memory_limit_mb])
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
| `radius` | 0 | [0, 16] | Temporal radius in frames. >0 switches output to the weighted intermediate for `VAggregate`. |
| `ref` | none | clip | External reference clip guiding both stages' matching (paper-style two-stage usage). |
| `chroma` | 0 | {0, 1} | CBM3D for YUV 4:4:4 clips: groups are matched on plane 0 only (of `ref` when given) and all three planes are filtered with them. See "Color (CBM3D)". |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `ps_num` | min(2, group) | [1, group] | Predictive-search seeds kept per temporal step (only used with `radius > 0`). |
| `ps_range` | 4 | [1, 64] | Predictive-search window radius around each seed (temporal mode). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Both stages in one call (`final`)

`BM3D(clip, final=1, ...)` runs the basic stage and then the Wiener stage
with the basic estimate as its reference. It gives exactly what the two calls

```python
basic = core.nss.BM3D(clip, sigma=sigma_basic, block_size=block_size_basic,
                      group_size=group_size_basic, ...)   # finished frames
out = core.nss.BM3D(clip, ref=basic, sigma=sigma, block_size=block_size,
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
- On the CPU this is the two calls chained. On `nss_cuda` the estimate stays
  on the device (see the [CUDA reference](cuda.md#bm3d-and-vaggregate)).

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

## Algorithm and paper

Dabov, Foi, Katkovnik & Egiazarian, *Image Denoising by Sparse 3D
Transform-Domain Collaborative Filtering*, IEEE TIP 2007. Two stages (Basic
hard-threshold, Wiener) with the standard `2.7 * sigma` hard-threshold
calibration, DCT/block transforms as applicable, and the plugin's deterministic
block-matching underneath. The paper's pipeline is followed; the numerics are
pinned down (fixed reduction orders, deterministic matcher tie-breaking) so the
same build is reproducible bit-for-bit on a given toolchain.

## Defaults rationale and performance

`block=8, step=8, group=8, bm_range=7, sigma=3` is the balanced point chosen
from this repository's calibration matrix. ~59 ms per 1080p GRAYS frame
on a single Emerald Rapids core (~17 fps) — the fastest filter here. The
dominant cost driver is `block_step` (positions scale as `1/step^2`), then
`bm_range` and `group_size`.

## Pitfalls

- With `radius > 0` the output is a **taller intermediate**, not a viewable
  frame — pipe it through `core.nss.VAggregate(out, clip, radius=radius)`.
- `sigma` is in 8-bit units even though the clip is float32; 25 means the
  usual "25/255" noise.
- Array parameters are per-plane: `block_size=[8,16]` is legal on YUV.

## CUDA

`core.nss_cuda.BM3D` and `core.nss_cuda.VAggregate` take the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA reference](cuda.md#bm3d-and-vaggregate).
