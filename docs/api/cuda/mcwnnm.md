# nss_cuda.MCWNNM

Multi-channel WNNM for RGB on the device, spatial and temporal.

Algorithm, paper and the reasoning behind the defaults: [`nss.MCWNNM`](../mcwnnm.md).

```python
out = core.nss_cuda.MCWNNM(rgb_clip, sigma=[25, 25, 25])   # defaults
quick = core.nss_cuda.MCWNNM(rgb_clip, sigma=25, iters=1)
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default | — | 200 |
| `radius=1` | — | 135 |

`iters=1` (one round instead of two) is about 1.5x faster at `radius=1`.

## Closer to the papers

The defaults are fast settings. Settings closer to the paper, with their quality and cost. Measured on one 1080p RGB frame (an anime credits frame with its own film grain) with Gaussian noise added, against the frame without it; the noisy input is 24.97 dB at sigma 15 and 20.98 dB at sigma 25. One image: take these as starting points, not tuned presets. σ is the standard deviation of the noise in 8-bit units; speed at 1080p on an RTX 5080 (the second row's on the test frame).

| Arguments | PSNR, sigma 15 | PSNR, sigma 25 | fps YUV420 / RGB |
|---|---:|---:|---:|
| defaults, `sigma=σ` | 25.62 dB | 21.63 dB | — / 200 |
| `block_step=4, group_size=32, iters=4, sigma=σ` | 33.68 dB | 30.93 dB | — / 0.38 |

- The outer rounds matter: with `iters=1` the same geometry gives 25.99 dB at sigma 15.

## Signature

```python
core.nss_cuda.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
                     int block_step = 8, int group_size = 8, int bm_range = 7,
                     int radius = 0, int ps_num = 2, int ps_range = 4,
                     int residual = 1, int adaptive_aggregation = 0,
                     clip rclip = None, int admm_iter = 10, float rho = 3.0,
                     float mu = 1.001, int iters = 2, float delta = 0.1,
                     int memory_limit_mb, string temporal_mode = "rolling",
                     int rolling_chunk = 4, int rolling_cache_chunks = 1,
                     int rolling_cache_limit = 16, int device_id = 0,
                     int num_streams = 1])
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
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
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

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 135–142 dB against the CPU plugin on the reference set;
  identical from run to run. Channels with the same `sigma` iterate as one, so
  different values per channel cost up to a quarter of the speed.
- **Temporal output** runs every round on the GPU and agrees with the legacy
  intermediate reduced by `VAggregate` to 148 dB.
- **Memory.** At RGB 1080p `radius=1` an instance takes about 920 MiB of GPU
  memory; the smallest accepted `memory_limit_mb` is 1183 MiB (827 with
  `iters=1`, 1752 at `radius=2`).

## Speed

RGB 1080p, RTX 5080 with a Ryzen 5950X, one stream, 2026-10-16: default 178 fps,
`radius=1` 113 (`iters=1`: 169), `radius=2` 90. Before the rolling mode ran on the
device (legacy + reduction on the host) `radius=1` took 39 fps (`iters=1`: 40).

## Pitfalls

- Input must be RGBS (three planes); for Gray use WNNM instead.
- The `mu=1.001` default keeps rho nearly constant across inner iterations,
  matching the authors; do not raise it without checking quality.
- `temporal_mode="legacy"` output needs an `nss.VAggregate` call; the default mode
  returns finished frames.
- No `_NSS*` frame properties.
