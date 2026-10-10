# nss_cuda.NCSR

Nonlocally centralized sparse representation on the device, spatial and temporal.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.NCSR`](../ncsr.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.NCSR(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.NCSR(clip, sigma=25, block_step=4)
temporal = core.nss_cuda.NCSR(clip, sigma=25, radius=1, iters=1)  # accumulation stays on the device
```

## Signature

```python
core.nss_cuda.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int group_size = 8, int bm_range = 7,
                   int radius = 0, int ps_num = 2, int ps_range = 4,
                   clip rclip = None, int iters = 2, float delta = 0.1,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
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
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `delta` | 0.1 | [0, 1] | Relaxation mixing previous and current estimates between rounds. |
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

- **Device-resident.** Matching, the per-group PCA and centralized shrinkage,
  the aggregation and all outer rounds run on the device.
- **Temporal output.** The temporal accumulation stays on the device when
  `iters = 1`; with more rounds the fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Numerics.** The PCA is taken through the FP32 Gram matrix with a cyclic
  Jacobi eigensolver, and the column weights use `expf` where the CPU uses
  its fast exponential.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (105–139 dB on the frozen references).
  - The output is run-to-run identical: every group is solved in a fixed
    order. Spatial and rolling output aggregates from inside the filter
    kernel with integer atomics on fixed-point sums (exact, so independent
    of the order); legacy temporal output sorts its patches and sums them in
    a fixed order.
  - Groups of up to 8 patches run as three kernels: the Gram matrices and
    the reconstruction read the patches with 8 threads per group, and the
    eigen step runs one thread per group in registers. Larger groups run 16
    or 32 threads per group with a round-robin parallel Jacobi.

## Pitfalls

- `iters=1` skips the second matching round: a sizeable speedup and a visible
  quality change on textured noise. It is also the setting whose temporal
  accumulation stays on the device.
- `temporal_mode="legacy"` output needs a `VAggregate` call; the default mode
  returns finished frames.
- No `_NSS*` frame properties on this filter.
