# nss_cuda.NLH

Non-local Haar transform denoising on the device: given or blind sigma, Gray / YUV / RGB, spatial and temporal.

The parameters mean what they mean in [`nss.NLH`](../nlh.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

```python
out = core.nss_cuda.NLH(clip)                          # blind estimation
awgn = core.nss_cuda.NLH(clip, sigma=25)               # explicit sigma
custom = core.nss_cuda.NLH(clip, sigma=25, block_size=[6, 9], block_step=[3, 6])
```

## Signature

```python
core.nss_cuda.NLH(clip clip[, float[] sigma, string noise_model = "auto",
                  int[] block_size, int[] block_step, int[] group_size,
                  int[] search_window, int bm_range, int radius = 0,
                  int ps_num = 2, int ps_range = 4, int[] q, clip rclip = None,
                  int basic_iters, float lambda_basic, float hard_strength,
                  int wiener_iters, float wiener_sigma_scale,
                  string temporal_mode = "rolling", int rolling_chunk = 4,
                  int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                  int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments). `temporal_mode` and the `rolling_*`
arguments are described under [temporal output](README.md#temporal-output).

## On the device

- **Device-resident.** The whole frame pipeline runs on the device: the
  RGB/YUV conversion, the blind noise estimate, every Basic round over the
  request window, the Wiener round and the per-pixel aggregation.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Frame properties.** The `_NSS*` diagnostics match the CPU; `_NSSSigma` of
  a blind estimate agrees to float precision.
- **Numerics.** Matching, pixel selection, the Haar transform and the
  thresholds follow the CPU model; the per-pixel sums are accumulated in FP32
  where the CPU uses FP64.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (77–107 dB on the frozen references; the differences are coefficients
    that fall on the other side of the hard threshold).
  - The output is run-to-run identical.
