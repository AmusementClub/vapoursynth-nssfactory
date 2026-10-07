# nss.NCSR

Nonlocally Centralized Sparse Representation: patch groups are projected onto
a PCA basis learned per group, the coefficients are centralized (weighted
mean removed) and soft-thresholded against a per-row noise-adaptive threshold,
then reconstructed. Sparse-coding quality between WNNM and LSSC in both cost
and behaviour.

```python
out = core.nss.NCSR(clip, sigma=25)                    # defaults
hiq  = core.nss.NCSR(clip, sigma=25, block_step=4)
```

## Signature

```python
core.nss.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8,
              int block_step = 8, int group_size = 8, int bm_range = 7,
              int radius = 0, int ps_num = 2, int ps_range = 4,
              clip rclip = None, int iters = 2, float delta = 0.1,
              int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Drives the adaptive soft-threshold (`tau ~ sigma^2 / sigma_row`). `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (columns of the PCA matrix). |
| `bm_range` | 7 | [1, 64] | Search window radius. |
| `iters` | 2 | [1, 64] | Outer re-estimation rounds; the second round re-matches on the current estimate. ~2x cost per extra round. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `delta` | 0.1 | [0, 1] | Relaxation mixing previous and current estimates between rounds. |
| `ps_num` / `ps_range` | 2 / 4 | [1,group] / [1,64] | Predictive temporal search (with `radius > 0`). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Algorithm and paper

Dong, Zhang, Shi & Li, *Nonlocally Centralized Sparse Representation for Image
Restoration*, IEEE TIP 2013. Per group: PCA projection of the patch matrix,
weighted centralization of coefficients, noise-adaptive soft thresholding, and
reconstruction with the group mean added back. The batch implementation keeps
bitwise-exact reduction orders across ISA lanes. A separate higher-quality
`HQ` path exists in the tree but is not part of this public filter.

## Defaults rationale and performance

Shared geometry (8/8/8/7) + two rounds, matching the calibration matrix.
~268 ms per 1080p GRAYS frame single-core (~3.7 fps). Matching cost scales
with `1/step^2 * bm_range^2`; per-group cost scales with the PCA/SVD solve at
`block^2 x group`.

## Pitfalls

- `iters=1` skips the second matching round — sizeable speedup, visible
  quality change on textured noise; measure before shipping.
- `radius > 0` output requires `VAggregate`.
- No `_NSS*` frame properties on this filter.

## CUDA

`core.nss_cuda.NCSR` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/ncsr.md).
