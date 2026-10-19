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
| `radius` | 0 | 0 only | Kept for the common signature: LSSC has no temporal mode, and `radius > 0` is rejected. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

The dictionary size (256 atoms) and cluster count (64) are internal constants,
capped by the actual patch count; they are intentionally not exposed.

## Algorithm and paper

Mairal, Bach, Ponce, Sapiro & Zisserman, *Non-local Sparse Models for Image
Restoration*, ICCV 2009. This implementation: k-means clustering of reference
patches, one dictionary per plane (DCT atoms plus sampled patches, one
K-SVD-style update with OMP at sparsity 8, mixed-precision screening: FP32
correlation screen, ordered FP64 refinement and residual), then each cluster
coded jointly by an l1,2 group-sparse problem (below). The dictionary update
is this repository's own formulation, not Mairal's ODL, by design.

## Model version 3: the cluster coding

Each cluster's centered patches Y are coded against the dictionary D by 12
rounds of accelerated proximal gradient (FISTA) on

```text
0.5 (Y - D A)^T M (Y - D A) + sum_a lambda_a ||A_a||_2
M = c (D D^T + delta I)^-1,  delta = 3 s,  c = s + delta,  s = trace(D D^T) / m
lambda_a = (sigma / 255) sqrt(cluster size) ||M d_a||
```

M weighs a direction of eigenvalue e of `D D^T` by `(s + delta) / (e + delta)`
(1 at the mean eigenvalue). The learned dictionaries are ill-conditioned
(eigenvalues about 0.34 to 27 at the defaults), and model version 2 (16 plain
ISTA rounds on the unweighted problem) stopped far from its solution at low
sigma: strokes lost contrast and white text darkened by about 12 levels at the
default sigma. Version 3 reaches the converged quality of the old objective in
12 rounds: on the development control set (three DIV2K crops and a synthetic
stroke image, two noise seeds), against version 2, PSNR +12.3 / +4.9 / +0.75 /
-0.14 dB on gray and +9.9 / +4.4 / +0.69 / -0.33 dB on RGB at sigma 1 / 3 /
10 / 25. The loss at sigma 25 is the objective's, not the solver's: the old
objective solved to convergence loses the same 0.14 dB on gray. At sigma 1
the output is still below the noisy input on every control image, as with
version 2; the penalty is not calibrated for very low noise.

The preconditioner is built once per plane (`D D^T`, a Cholesky factor of
`D D^T + delta I`, two triangular solves per atom); the step is 0.9 / L' with
`L' = c L / (L + delta)` from the power estimate L of `||D||^2`. Constants
and the momentum table are in `include/nss/params/lssc.hpp`, shared by both
plugins. The model semantic version (`include/nss/contracts.hpp`) is 3; LSSC
stamps no frame properties, so outputs of the two versions are told apart only
by the plugin build.

## Defaults rationale and performance

`block=8, step=8, sigma=3`. About 330 ms per 1080p GRAYS frame on one Ryzen 9
5950X core (version 2: 410 ms on the same core). The cost splits between the
clustering/matching passes (scaling with `1/step^2`) and the per-cluster
coding/dictionary iterations (scaling with patch count and atoms). The coding
keeps the previous iterate: the workspace holds `min(256, patches)` more
floats per patch than version 2. An opt-in Apple Silicon SME matrix leaf exists
for eligible builds (`-DNSS_ENABLE_LSSC_SME=ON`); the default path is
Highway-only and deterministic.

## Pitfalls

- Only block sizes 1/2/4/8/16 are accepted; other values fail at creation.
- The clustering uses the patches themselves as seeds; pathological inputs
  (constant frames) exercise the empty-cluster steal logic and still complete.
- No `_NSS*` frame properties on this filter.

## CUDA

`core.nss_cuda.LSSC` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/lssc.md).
