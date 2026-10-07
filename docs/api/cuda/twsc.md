# nss_cuda.TWSC

Trilateral weighted sparse coding on the device: given or blind sigma, Gray / YUV / RGB, spatial and temporal.

The parameters mean what they mean in [`nss.TWSC`](../twsc.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

```python
out = core.nss_cuda.TWSC(clip, sigma=25)                    # balanced default
fast = core.nss_cuda.TWSC(clip, sigma=25, block_step=8, group_size=8, iters=2)
blind = core.nss_cuda.TWSC(clip, estimate_sigma=1)          # per-frame sigma
```

## Signature

```python
core.nss_cuda.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0,
                   int block_size = 8, int block_step = 1, int group_size = 90,
                   int search_window = 60, int bm_range, int radius = 0,
                   int ps_num = 2, int ps_range = 4, float lambda2 = 1.0,
                   clip rclip = None, int iters = 12, float delta = 0.0,
                   int admm_iter = 10, float rho = 0.5, float mu = 1.1,
                   float tol = 1e-6,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments). `temporal_mode` and the `rolling_*`
arguments are described under [temporal output](README.md#temporal-output).

## On the device

- **Device-resident.** The blind noise estimate, the joint matching, the
  per-group dictionary and ADMM solve, the aggregation and every round run on
  the device.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
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
