# nss_cuda.MCWNNM

Multi-channel WNNM for RGB on the device, spatial and temporal.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.MCWNNM`](../mcwnnm.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.MCWNNM(rgb_clip, sigma=[25, 25, 25])   # defaults
quick = core.nss_cuda.MCWNNM(rgb_clip, sigma=25, iters=1)
```

## Signature

```python
core.nss_cuda.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
                     int block_step = 8, int group_size = 8, int bm_range = 7,
                     int radius = 0, int ps_num = 2, int ps_range = 4,
                     int residual = 1, int adaptive_aggregation = 0,
                     clip rclip = None, int admm_iter = 10, float rho = 3.0,
                     float mu = 1.001, int iters = 2, float delta = 0.1,
                     string temporal_mode = "rolling", int rolling_chunk = 4,
                     int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                     int memory_limit_mb, int device_id = 0, int num_streams = 1])
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

## Device parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `device_id` | 0 | device index | CUDA device the instance runs on. One instance uses one device; `core.nss_cuda.Backend(device_id)` reports whether it is supported. |
| `num_streams` | 1 | [1, 16] | Frames (or temporal chunks) the instance has in flight on the device at once. Each stream owns its device buffers, so memory grows with it. |
| `memory_limit_mb` | none | > 0 | Caps the instance's device and pinned host memory. The internal batches are fitted to it and the output does not change; a limit that cannot hold the streams is a creation error. Without it the filter takes what its plan needs. |
| `temporal_mode` | `"rolling"` | rolling / legacy | With `radius > 0`: `"rolling"` returns finished, normal-height frames; `"legacy"` returns the fat intermediate for `VAggregate`, as the CPU plugin does. |
| `rolling_chunk` | 4 | [1, 64] | Frames accumulated per rolling chunk. |
| `rolling_cache_chunks` | 1 | [1, 64] | Finished chunks kept for later frame requests at first. Given alone, the cache stays at this size. |
| `rolling_cache_limit` | 16 | [1, 64], >= `rolling_cache_chunks` | What the chunk cache may grow to: it keeps one more chunk each time two requests miss on chunks it dropped recently. Under `memory_limit_mb` it grows only into what the limit leaves. |

How the streams, the memory limit and the rolling output behave across filters is in the [shared notes](README.md#shared-by-every-filter).

## On the device

- **Device-resident.** The joint three-channel matching, the ADMM solve of
  every group, the aggregation and all outer rounds run on the device.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Numerics.** Each ADMM step shrinks through the FP32 Gram matrix with a
  cyclic Jacobi eigensolver that starts from the eigenvectors of the step
  before. The iteration runs on one small matrix pair per channel (the
  iterates are the centered input times these), which is the same
  mathematics as the CPU's iterates with other rounding.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (135–142 dB on the frozen references).
  - The output is run-to-run identical: every group is solved in a fixed
    order. Spatial output aggregates from inside the filter kernel with
    integer atomics on fixed-point sums (exact, so independent of the
    order); temporal output sorts its patches and sums them in a fixed
    order.
  - Groups of up to 8 patches run as three kernels: the channel Gram
    matrices and the output read the patches with 8 threads per group, and
    the ADMM runs one thread per group. Larger groups run 16 or 32 threads
    per group with a round-robin parallel Jacobi.
  - Channels with the same `sigma` have the same weight and iterate as
    one: with one `sigma` for all three (the default) the ADMM keeps and
    updates a third of the state. Different values per channel cost up to
    a quarter of the speed.
- **Memory.** The ADMM state costs about 2.5 KiB of device memory per group
  at the defaults.

## Pitfalls

- Input must be RGBS (three planes); for Gray use WNNM instead.
- The `mu=1.001` default keeps rho nearly constant across inner iterations,
  matching the authors; do not raise it without checking quality.
- `temporal_mode="legacy"` output needs a `VAggregate` call; the default mode
  returns finished frames.
- No `_NSS*` frame properties.
