# nss_cuda.MCWNNM

Multi-channel WNNM for RGB on the device, spatial and temporal.

The parameters mean what they mean in [`nss.MCWNNM`](../mcwnnm.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

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

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments). `temporal_mode` and the `rolling_*`
arguments are described under [temporal output](README.md#temporal-output).

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
  - Groups of up to 8 patches run one thread per group. Larger groups run 16
    or 32 threads per group with a round-robin parallel Jacobi.
- **Memory.** The ADMM state costs about 2.5 KiB of device memory per group
  at the defaults.
