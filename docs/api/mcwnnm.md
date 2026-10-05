# nss.MCWNNM

Multi-Channel WNNM: extends weighted nuclear norm minimization to color by
solving a joint ADMM over the RGB channels of each patch group, exploiting
cross-channel correlation. The plugin's dedicated color-image denoiser — and
its most expensive one per pixel family.

```python
out = core.nss.MCWNNM(rgb_clip, sigma=[25, 25, 25])   # defaults
quick = core.nss.MCWNNM(rgb_clip, sigma=25, iters=1)  # coarser but ~2x faster
```

## Signature

```python
core.nss.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
                int block_step = 8, int group_size = 8, int bm_range = 7,
                int radius = 0, int ps_num = 2, int ps_range = 4,
                int residual = 1, int adaptive_aggregation = 0,
                clip rclip = None, int admm_iter = 10, float rho = 3.0,
                float mu = 1.001, int iters = 2, float delta = 0.1,
                int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | [3,3,3] | >= 0 | 8-bit noise stddev per channel (list of up to 3; last value broadcasts). Unequal channel noise is supported and common for real sensors. `sigma=0` bypasses the channel. |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group. |
| `bm_range` | 7 | [1, 64] | Search window radius. |
| `iters` | 2 | [1, 64] | Outer re-estimation rounds (re-match on the current estimate). Round 2 costs ~another full pass; reduce to 1 for a ~2x speedup when quality allows. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `residual` | 1 | 0 / 1 | `1` demeans patch rows (Matlab Estimation pipeline, the factory default); `0` keeps DC in the ADMM matrix (bare `MCWNNM_ADMM.m`). |
| `admm_iter` | 10 | [1, 1000] | Inner ADMM iterations per group per round. The dominant per-group cost; see the solver contract before lowering. |
| `rho` | 3.0 | > 0 | Initial ADMM penalty. |
| `mu` | 1.001 | >= 1 | Penalty growth factor per inner iteration (near-constant by default). |
| `delta` | 0.1 | [0, 1] | Relaxation mixing the previous estimate between outer iterations. |
| `adaptive_aggregation` | 0 | 0 / 1 | Residual-weighted aggregation (off by default here, unlike WNNM). |
| `ps_num` / `ps_range` | 2 / 4 | [1,group] / [1,64] | Predictive temporal search (with `radius > 0`). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Algorithm and paper

Xu, Li, Yang & Zhang, *Multi-channel Weighted Nuclear Norm Minimization for
Real Color Image Denoising*, ICCV 2017. Patch groups stack all three channels;
the WNNM objective is solved with ADMM (X/Z/A iterates with the rho/mu penalty
schedule above). Two outer rounds re-estimate from the current reconstruction.
The default `residual=1` matches the authors' Estimation pipeline. Solver
tolerances and validation lanes are pinned so a given toolchain reproduces
output bit-for-bit.

## Defaults rationale and performance

Defaults mirror the shared geometry (block/step/group/range 8/8/8/7) plus the
authors' ADMM schedule. ~2326 ms per 1080p frame single-core (~0.43 fps) —
**note this is for all three RGB planes**, so per-channel cost is ~3x a gray
filter at the same settings. `admm_iter` and `iters` multiply the solve cost
directly; `block_step` scales positions as usual.

## Pitfalls

- Input must be RGBS (three planes); for Gray use WNNM instead.
- The `mu=1.001` default keeps rho nearly constant across inner iterations —
  deliberate, matching the authors; do not "fix" it to a large growth factor
  without quality validation.
- No `_NSS*` frame properties; diagnostics live on NLH/TWSC.

## CUDA (`core.nss_cuda.MCWNNM`)

`core.nss_cuda.MCWNNM` (from `libnss_cuda`, built with `-DNSS_ENABLE_CUDA=ON`)
takes the same arguments and gives the same errors as `nss.MCWNNM`. It adds
`device_id` (default 0) and `num_streams` (default 1) at the end of the argument list.

- **Device-resident.** The joint three-channel matching, the ADMM solve of
  every group, the aggregation and all outer rounds run on the device.
  `radius > 0` returns the same fat intermediate as the CPU, so either
  backend's `VAggregate` can reduce it.
- **Temporal output.** With `radius > 0` the device plugin returns finished,
  normal-height frames by default (`temporal_mode = "rolling"`): it
  reduces the fat intermediate with `nss_cuda.VAggregate` inside the filter.
  `temporal_mode = "legacy"` returns the fat intermediate instead, as the CPU
  plugin does. `rolling_chunk` (default 4, range 1 to 64) and
  `rolling_cache_limit` (default 1) set the chunk size and the number of
  finished chunks kept; they only matter where the accumulation stays on the
  device.
- **Numerics.** Each ADMM step shrinks through the FP32 Gram matrix with a
  cyclic Jacobi eigensolver.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (135–142 dB on the frozen references).
  - One thread handles each group in a fixed order, and aggregation is
    ordered.
- **Memory.** The ADMM state costs about 12 KiB of device memory per group at
  the defaults, so a 1080p frame runs in several batches.
