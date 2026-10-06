# nss.TWSC

Trilateral Weighted Sparse Coding: sparse coding with three
weights — the usual residual weight plus weights derived from the noise
estimate and the current reconstruction — solved with a full ADMM. Aimed at
real-world (signal-dependent) noise where plain AWGN models misestimate.
**The balanced default is deliberately heavy**; see the performance note.

```python
out = core.nss.TWSC(clip, sigma=25)                    # balanced default
fast = core.nss.TWSC(clip, sigma=25, block_step=8, group_size=8, iters=2)
blind = core.nss.TWSC(clip, estimate_sigma=1)          # per-frame sigma
```

## Signature

```python
core.nss.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0,
              int block_size = 8, int block_step = 1, int group_size = 90,
              int search_window = 60, int bm_range, int radius = 0,
              int ps_num = 2, int ps_range = 4, float lambda2 = 1.0,
              clip rclip = None, int iters = 12, float delta = 0.0,
              int admm_iter = 10, float rho = 0.5, float mu = 1.1,
              float tol = 1e-6,
              int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Controls denoising strength only — one geometry is kept for all noise levels. `sigma=0` bypasses the plane. Mutually exclusive with `estimate_sigma`. |
| `estimate_sigma` | 0 | 0 / 1 | Estimate sigma per frame/channel instead of taking it. |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 1 | [1, block] | Reference-patch stride. **The default is 1 (every position)** — overriding it is the single largest speed lever: positions scale as `1/step^2`, so `block_step=8` is ~64x fewer groups. |
| `group_size` | 90 | [1, 256] | Matched patches per group. 90 follows the paper's high-quality regime; 8–16 trades quality for large speedups. |
| `search_window` | 60 | [1, 129] | Matching window edge. `bm_range=r` means `2r+1`; passing both is an error. |
| `iters` | 12 | [1, 64] | Outer iterations (re-match and re-solve per round). Nearly linear cost; 12 is the balanced default, 2–4 for quick work. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `lambda2` | 1.0 | >= 0 | Weight on the reconstruction-fidelity side of the three-weight objective. |
| `delta` | 0.0 | [0, 1] | Relaxation mixing between iterations (0 = no mixing, the default). |
| `admm_iter` | 10 | [1, 1000] | Inner ADMM iterations per group solve. |
| `rho` | 0.5 | > 0 | Initial ADMM penalty. |
| `mu` | 1.1 | >= 1 | Penalty growth factor per inner iteration. |
| `tol` | 1e-6 | > 0 | ADMM convergence tolerance (early stop). |
| `ps_num` / `ps_range` | 2 / 4 | [1,group] / [1,64] | Predictive temporal search (`radius > 0`). `ps_num` defaults to 1 when `group_size=1` is explicit. |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Diagnostics

Each output frame stamps: `_NSSSigma` (per-plane, in 8-bit units),
`_NSSGroups`, `_NSSADMMMaxIterGroups`, `_NSSSvdDoubleGroups`,
`_NSSSylvesterResidual`, `_NSSBlockSize`, `_NSSGroupSize`, `_NSSBlockStep`,
`_NSSSearchWindow`, `_NSSIterations`, and `_NSSModelVersion=3`.

## Algorithm and paper

Xu, Zhang, Yang & Zhang, *Trilateral Weighted Sparse Coding for Real-world
Image Denoising*, ECCV 2018. The plugin implements the complete three-weight ADMM
objective with the selected author preprocessing; there is no legacy runtime
mode, and the removed `lambda1` parameter is rejected (including zero).

## Defaults rationale and performance

The plugin deliberately uses one stable geometry (`block 8, step 1, group 90, window 60,
iters 12`) across noise levels — sigma controls strength, not geometry. This is
**far** heavier than the geometry used in the plugin's benchmark harness
(`step 8, group 8, iters 2`, ~1337 ms/frame, ~0.75 fps single-core 1080p
GRAYS). The true default has ~64x more reference positions, ~11x larger groups,
and 6x the iterations; it has not been formally timed and should be expected to
take minutes per frame. Override the three levers explicitly for interactive
work.

## Pitfalls

- The balanced default is a quality-first configuration, not a realtime one —
  always check whether you actually want `block_step=1`.
- `estimate_sigma` and explicit `sigma` are mutually exclusive (creation-time
  error).
- `ps_num > group_size` is an error after resolution.

## CUDA (`core.nss_cuda.TWSC`)

`core.nss_cuda.TWSC` (from `libnss_cuda`, built with `-DNSS_ENABLE_CUDA=ON`)
takes the same arguments and gives the same errors as `nss.TWSC`. It adds
`device_id` (default 0) and `num_streams` (default 1) at the end of the argument list.

- **Device-resident.** The blind noise estimate, the joint matching, the
  per-group dictionary and ADMM solve, the aggregation and every round run on
  the device. `radius > 0` returns the same fat intermediate as the CPU, so
  either backend's `VAggregate` can reduce it.
- **Temporal output.** With `radius > 0` the device plugin returns finished,
  normal-height frames by default (`temporal_mode = "rolling"`): it
  reduces the fat intermediate with `nss_cuda.VAggregate` inside the filter.
  `temporal_mode = "legacy"` returns the fat intermediate instead, as the CPU
  plugin does. `rolling_chunk` (default 4, range 1 to 64) and
  `rolling_cache_chunks` / `rolling_cache_limit` (defaults 1 and 16) set the
  chunk size and the number of finished chunks kept at first and at most; they
  only matter where the accumulation stays on the device.
- **Numerics.** The dictionary comes from an FP32 Jacobi eigendecomposition of
  the Gram matrix of the smaller group side, and the solver runs in FP32
  (FP64 was measured and gave the same agreement with the CPU).
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (92–143 dB on the frozen references) and run-to-run identical.
- **Frame properties.** `_NSSSigma`, `_NSSGroups` and the shape properties
  match the CPU. `_NSSADMMMaxIterGroups` counts this backend's own
  non-converged groups; `_NSSSvdDoubleGroups` and `_NSSSylvesterResidual` are
  CPU-solver diagnostics and are reported as 0.
- **Speed.** The default settings remain very heavy (one 64x64
  eigendecomposition per pixel position per round): about 8 s for a 128x128
  frame on an RTX 5080, against about 190 s on 16 CPU threads.
