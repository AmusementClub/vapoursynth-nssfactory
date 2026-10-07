# nss_cuda.NCSR

Nonlocally centralized sparse representation on the device, spatial and temporal.

The parameters mean what they mean in [`nss.NCSR`](../ncsr.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

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

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments). `temporal_mode` and the `rolling_*`
arguments are described under [temporal output](README.md#temporal-output).

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
    order, and aggregation is ordered.
  - Groups of up to 8 patches run one thread per group. Larger groups run one
    thread block per group with a round-robin parallel Jacobi.
