# nss_cuda.TWSC

Trilateral weighted sparse coding on the device: given or blind sigma, Gray /
YUV / RGB, spatial and temporal.

Algorithm, paper and the reasoning behind the defaults: [`nss.TWSC`](../twsc.md).

```python
out = core.nss_cuda.TWSC(clip, sigma=25)                    # fast default
best = core.nss_cuda.TWSC(clip, sigma=25, block_step=4, group_size=90, iters=12)   # quality
blind = core.nss_cuda.TWSC(clip, estimate_sigma=1)          # per-frame sigma
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default | 66 | 34 |

The default is a speed setting. The quality geometry `block_step=1,
group_size=90, iters=12` (4 on the fast color path) denoises 8 to 10 dB better
at sigma 5 to 100 and takes about 80 s per YUV420 1080p frame and 45 s per RGB
frame; `block_step=4` keeps its quality at a sixteenth of the groups. The
[CPU page](../twsc.md#defaults-rationale-and-performance) has the
measurements.

## Signature

```python
core.nss_cuda.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0,
                   int block_size = 8, int block_step = 8, int group_size = 8,
                   int search_window = 60, int bm_range, int radius = 0,
                   int ps_num = 2, int ps_range = 4, float lambda2 = 1.0,
                   clip rclip = None, int iters = 2, float delta = 0.0,
                   int admm_iter = 10, float rho = 0.5, float mu = 1.1,
                   float tol = 1e-6, int cfast = 1,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Controls denoising strength only — one geometry is kept for all noise levels. `sigma=0` bypasses the plane. Mutually exclusive with `estimate_sigma`. |
| `estimate_sigma` | 0 | 0 / 1 | Estimate sigma per frame/channel instead of taking it (the NLH estimator: 8x8 blocks on every fourth position per axis). |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | min(8, block) | [1, block] | Reference-patch stride: positions scale as `1/step^2`. 4 keeps the quality of 1 (within 0.1 dB) at a sixteenth of the groups. |
| `group_size` | 8 | [1, 256] | Matched patches per group. 90 follows the paper's high-quality regime (see [Speed](#speed)). |
| `search_window` | 60 | [1, 129] | Matching window edge. `bm_range=r` means `2r+1`; passing both is an error. |
| `iters` | 2 | [1, 64] | Outer iterations (re-match and re-solve per round), nearly linear cost. With `group_size=90`: 12 for the joint filter, 4 on the fast color path. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
| `rclip` | none | clip | Reference clip guiding matching. |
| `cfast` | 1 | 0 / 1 | Fast color path: an RGB frame whose three channels share one given `sigma` is filtered as three opponent-color planes, each alone, instead of as stacked three-channel patches. See [Fast color path](#fast-color-path-cfast). `cfast=0` keeps the joint filter. |

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

## Diagnostics

Each output frame stamps: `_NSSSigma` (per-plane, in 8-bit units),
`_NSSGroups`, `_NSSADMMMaxIterGroups`, `_NSSSvdDoubleGroups`,
`_NSSSylvesterResidual`, `_NSSBlockSize`, `_NSSGroupSize`, `_NSSBlockStep`,
`_NSSSearchWindow`, `_NSSIterations`, `_NSSCFast` (1 when the fast color
path ran) and `_NSSModelVersion=6`.

## Fast color path (`cfast`)

With `cfast=1` (the default) an RGB clip whose three channels have one given
`sigma` (not `estimate_sigma`, and only with `radius=0`) is filtered as three
opponent-color planes, each on its own, and, with the quality geometry, 4 rounds are best (12 joint). The
conditions, the quality it was measured at and when it pays are on the
[CPU page](../twsc.md#fast-color-path-cfast). `_NSSCFast` is 1 on frames where
it ran.

- It pays when `group_size` is larger than `block_size^2` (the default 90
  against 64): RGB 1080p, `sigma=3, block_size=4, block_step=4,
  group_size=32, search_window=20` runs at 13.1 fps with it at `iters=2` and
  9.0 without.
- With a small group (the default `group_size=8`) the joint dictionary is no
  larger and three planes are three times the groups: `cfast=0` is the faster
  call there.

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 93–143 dB against the CPU plugin on the reference set (142 dB
  on the fast color path); identical from run to run. The solver runs in FP32
  (FP64 gave the same agreement).
- **Frame properties.** `_NSSSigma`, `_NSSGroups` and the shape properties
  match the CPU. `_NSSADMMMaxIterGroups` counts this plugin's own
  non-converged groups; `_NSSSvdDoubleGroups` and `_NSSSylvesterResidual`
  describe the CPU solver and are 0 here.
- **Temporal output** is bit for bit the legacy intermediate reduced by
  `VAggregate`, and takes about 80 MiB more than legacy output at gray 1080p
  `radius=1`.

## Pitfalls

- The default is the fast geometry; for quality pass the larger group and
  more rounds (see [Speed](#speed)).
- `estimate_sigma` and explicit `sigma` are mutually exclusive (creation-time
  error).
- `ps_num > group_size` is an error after resolution.
- `temporal_mode="legacy"` output needs an `nss.VAggregate` call; the default mode
  returns finished frames.
