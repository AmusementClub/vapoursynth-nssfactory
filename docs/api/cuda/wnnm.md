# nss_cuda.WNNM

Weighted nuclear norm minimization on the device, spatial and temporal.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.WNNM`](../wnnm.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.WNNM(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.WNNM(clip, sigma=25, block_step=4, group_size=16)
temporal = core.nss_cuda.WNNM(clip, sigma=25, radius=1)      # finished frames
```

## Signature

```python
core.nss_cuda.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int group_size = 8, int bm_range = 7,
                   int radius = 0, int ps_num = 2, int ps_range = 4,
                   int residual = 0, int adaptive_aggregation = 1,
                   clip rclip = None,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane (broadcasts). Sets the shrinkage strength. `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. 8 is the balanced default; larger blocks raise SVD cost steeply. |
| `block_step` | 8 | [1, block] | Reference-patch stride. The main quality/speed trade: positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (SVD matrix columns). |
| `bm_range` | 7 | [1, 64] | Search window radius (`2r+1`). |
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `residual` | 0 | 0 / 1 | `1` demeans patch rows before shrinkage (Matlab Estimation-pipeline flavour); `0` keeps the DC component in the matrix (bare `MCWNNM_ADMM.m` flavour). The factory default keeps DC. |
| `adaptive_aggregation` | 1 | 0 / 1 | Weight group contributions by their residual confidence when aggregating. Usually leave on. |
| `ps_num` | 2 | [1, group] | Predictive-search seeds per temporal step (with `radius > 0`). |
| `ps_range` | 4 | [1, 64] | Predictive-search window radius (temporal mode). |

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

- **Device-resident.** Matching, the per-group SVD shrinkage and the
  aggregation all run on the device.
- **Temporal output.** The temporal accumulation stays on the device.
- **Numerics.** The SVD is taken through the FP32 Gram matrix with a cyclic
  Jacobi eigensolver.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (86–143 dB on the frozen references).
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

- `group_size=1` degenerates the SVD to per-patch shrinkage; legal but rarely
  useful.
- `temporal_mode="legacy"` output is the taller weighted intermediate and
  needs a `VAggregate` call; the default mode returns finished frames.
- This filter emits no `_NSS*` frame properties.
