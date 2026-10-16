# nss_cuda.NCSR

Nonlocally centralized sparse representation on the device, spatial and temporal.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.NCSR`](../ncsr.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.NCSR(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.NCSR(clip, sigma=25, block_step=4)
temporal = core.nss_cuda.NCSR(clip, sigma=25, radius=1)  # finished frames, every round on the device
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
| `temporal_mode` | `"rolling"` | rolling / legacy | With `radius > 0`: `"rolling"` returns finished, normal-height frames; `"legacy"` returns the fat intermediate, as the CPU plugin does; reduce it with `nss.VAggregate`. |
| `rolling_chunk` | 4 | [1, 64] | Frames accumulated per rolling chunk. |
| `rolling_cache_chunks` | 1 | [1, 64] | Finished chunks kept for later frame requests at first. Given alone, the cache stays at this size. |
| `rolling_cache_limit` | 16 | [1, 64], >= `rolling_cache_chunks` | What the chunk cache may grow to: it keeps one more chunk each time two requests miss on chunks it dropped recently. Under `memory_limit_mb` it grows only into what the limit leaves. |

How the streams, the memory limit and the rolling output behave across filters is in the [shared notes](README.md#shared-by-every-filter).

## On the device

- **Device-resident.** Matching, the per-group PCA and centralized shrinkage,
  the aggregation and all outer rounds run on the device.
- **Temporal output.** The rolling mode (the default) runs every round on the
  device for any `iters`. As in the CPU's fat output, each center's rounds
  before the last work on estimates of that center's own window (finished
  from its groups alone, then relaxed toward the noisy frames by `delta`);
  the last round adds to the rolling sums. It agrees with
  `VAggregate(NCSR(..., temporal_mode="legacy"))` to 142–152 dB on the test
  clips.
  - With `iters > 1` a stream also holds the 2R+1 estimates of its center and
    their fixed-point sums: 1080p `radius=1` takes 308 MiB of device memory
    above a bare context (204 with the earlier legacy-inside path), and the
    smallest accepted `memory_limit_mb` is 405 MiB (228 before).
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
    or 32 threads per group: up to 16 patches with a round-robin parallel
    Jacobi, 17 to 32 with Householder tridiagonalization and divide and
    conquer (closer to the CPU than the Jacobi there, and 2.3x faster; it
    takes 14.5 KiB of device scratch per group of a batch).

## Speed

1080p, RTX 5080 with a Ryzen 5950X, one stream, 2026-10-16: default (gray) 516 fps,
`block_step=4` 266, `group_size=16` 177, `group_size=32, block_step=4` 13.3, RGB 195,
`radius=1` 310 (`iters=1`: 456); `group_size=32` with the default step 50. The
16-thread CPU plugin on the same host: 35.8, 6.8, 3.0, 0.084, 8.4 and 11.7 (21.6).
The spatial default is bound by the host's frame copies (about 600 fps with
`iters=1`); more streams do not raise it.

## Pitfalls

- `iters=1` skips the second matching round: a sizeable speedup and a visible
  quality change on textured noise.
- `temporal_mode="legacy"` output needs an `nss.VAggregate` call; the default mode
  returns finished frames.
- No `_NSS*` frame properties on this filter.
