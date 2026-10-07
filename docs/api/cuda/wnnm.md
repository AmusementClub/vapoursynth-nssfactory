# nss_cuda.WNNM

Weighted nuclear norm minimization on the device, spatial and temporal.

The parameters mean what they mean in [`nss.WNNM`](../wnnm.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

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

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments). `temporal_mode` and the `rolling_*`
arguments are described under [temporal output](README.md#temporal-output).

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
  - Groups of up to 8 patches run one thread per group. Larger groups run 16
    or 32 threads per group with a round-robin parallel Jacobi.
