# nss.LSSC

Learned Simultaneous Sparse Coding: patches are clustered (k-means), each
cluster learns a dictionary, and patches in a cluster are coded
*simultaneously* — the joint sparsity pattern is shared, which preserves grain
and stochastic texture better than independent sparse coding. Moderate cost,
distinctly "textured" output character.

```python
out = core.nss.LSSC(clip, sigma=25)                    # defaults
hiq  = core.nss.LSSC(clip, sigma=25, block_step=4)
```

## Signature

```python
core.nss.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8,
              int block_step = 8, int radius = 0, int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Drives the sparse-coding regularization. `sigma=0` bypasses the plane. |
| `block_size` | 8 | {1,2,4,8,16} | Patch edge (power-of-two family only). Dictionary dimension is `block^2`. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. The main quality/speed trade. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

The dictionary size (256 atoms) and cluster count (64) are internal constants,
capped by the actual patch count; they are intentionally not exposed.

## Algorithm and paper

Mairal, Bach, Ponce, Sapiro & Zisserman, *Non-local Sparse Models for Image
Restoration*, ICCV 2009. This implementation: k-means clustering of reference
patches, per-cluster dictionary update (K-SVD-style step, `ksvd_iters=1`), and
simultaneous OMP coding with mixed-precision screening (FP32 correlation
screen, ordered FP64 refinement and residual). The dictionary update is this
repository's own formulation, not Mairal's ODL, by design.

## Defaults rationale and performance

`block=8, step=8, sigma=3`. ~369 ms per 1080p GRAYS frame single-core
(~2.7 fps). The cost splits between the clustering/matching passes (scaling
with `1/step^2`) and the per-cluster ISTA/dictionary iterations (scaling with
patch count and atoms). An opt-in Apple Silicon SME matrix leaf exists for
eligible builds (`-DNSS_ENABLE_LSSC_SME=ON`); the default path is Highway-only
and deterministic.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- The clustering uses the patches themselves as seeds; pathological inputs
  (constant frames) exercise the empty-cluster steal logic and still complete.
- No `_NSS*` frame properties on this filter.

## CUDA (`core.nss_cuda.LSSC`)

`core.nss_cuda.LSSC` (from `libnss_cuda`, built with `-DNSS_ENABLE_CUDA=ON`)
takes the same arguments and gives the same errors as `nss.LSSC`. It adds
`device_id` (default 0) and `num_streams` (default up to 3, fitted to
`memory_limit_mb`) at the end of the argument list.

- **Device-resident.** Patch packing, k-means, the dictionary (DCT atoms,
  sampled patches, one K-SVD round), the per-cluster sparse coding, the
  reconstruction and the aggregation run on the device. The host only
  sequences the steps and keeps index bookkeeping (cluster member lists and
  the k-means round control).
- **Numerics.** The K-SVD sparse codes use FP64 correlations and solves as on
  the CPU; an atom's rank-one update takes the dominant singular triplet from
  an FP64 power iteration instead of a full SVD. Everything else is FP32.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (139–141 dB on the frozen references) and run-to-run identical.
- **Memory.** The coefficients cost `min(256, patches)` floats per grid patch,
  so a dense `block_step` on a large frame needs a matching
  `memory_limit_mb`.
- **Speed.** The dictionary update is sequential over the atoms, so the gain
  over the CPU is small: about 1–2x the CPU plugin at 16 threads for 1080p.
