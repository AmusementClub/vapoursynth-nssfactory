# nss.WNNM

Weighted Nuclear Norm Minimization: patch groups are shrunk through a weighted
singular-value soft-threshold instead of a fixed transform. Better texture and
edge retention than BM3D on natural noise, at about 2x BM3D's cost.

```python
out = core.nss.WNNM(clip, sigma=25)                    # defaults
hiq  = core.nss.WNNM(clip, sigma=25, block_step=4, group_size=16)
```

## Signature

```python
core.nss.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
              int block_step = 8, int group_size = 8, int bm_range = 7,
              int radius = 0, int ps_num = 2, int ps_range = 4,
              int residual = 0, int adaptive_aggregation = 1,
              clip rclip = None,
              int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane (broadcasts). Sets the shrinkage strength. `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. 8 is the balanced default; larger blocks raise SVD cost steeply. |
| `block_step` | 8 | [1, block] | Reference-patch stride. The main quality/speed trade: positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (SVD matrix columns). |
| `bm_range` | 7 | [1, 64] | Search window radius (`2r+1`). |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `residual` | 0 | 0 / 1 | `1` demeans patch rows before shrinkage (Matlab Estimation-pipeline flavour); `0` keeps the DC component in the matrix (bare `MCWNNM_ADMM.m` flavour). The factory default keeps DC. |
| `adaptive_aggregation` | 1 | 0 / 1 | Weight group contributions by their residual confidence when aggregating. Usually leave on. |
| `ps_num` | 2 | [1, group] | Predictive-search seeds per temporal step (with `radius > 0`). |
| `ps_range` | 4 | [1, 64] | Predictive-search window radius (temporal mode). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Algorithm and paper

Gu, Zhang, Zuo & Feng, *Weighted Nuclear Norm Minimization with Application to
Image Denoising*, CVPR 2014. Groups of matched patches are decomposed by SVD
and their singular values shrunk by the closed-form weighted soft-threshold
operator `(s + sqrt(s^2 - c)) / 2` walking the ordered spectrum. The group mean
is subtracted/added around the solve. Reduction orders and validation lanes are
pinned so a given toolchain reproduces output bit-for-bit.

## Defaults rationale and performance

`block=8, step=8, group=8, bm_range=7, sigma=3` mirrors the shared search
geometry used across this plugin's denoisers so algorithms can be compared at
matched settings. ~131 ms per 1080p GRAYS frame single-core (~7.6 fps).
Cost scales with `1/step^2` for matching plus per-group SVD work growing with
`group_size` and `block_size^2`.

## Pitfalls

- `radius > 0` output is the taller weighted intermediate and needs a
  `VAggregate` call (`core.nss.VAggregate(out, clip, radius=radius)`) to get
  the viewable frame.
- `group_size=1` degenerates the SVD to per-patch shrinkage; legal but rarely
  useful.
- This filter emits no `_NSS*` frame properties; diagnostics live on NLH/TWSC.

## CUDA (`core.nss_cuda.WNNM`)

`core.nss_cuda.WNNM` (from `libnss_cuda`, built with `-DNSS_ENABLE_CUDA=ON`)
takes the same arguments and gives the same errors as `nss.WNNM`. It adds
`device_id` (default 0) and `num_streams` (default 1) at the end of the argument list.

- **Device-resident.** Matching, the per-group SVD shrinkage and the
  aggregation all run on the device. With `temporal_mode = "legacy"`, `radius > 0` returns the same fat
  intermediate as the CPU, so either backend's `VAggregate` can reduce it.
- **Temporal output.** With `radius > 0` the device plugin returns finished,
  normal-height frames by default (`temporal_mode = "rolling"`): it
  keeps the temporal accumulation on the device and copies back only final frames.
  `temporal_mode = "legacy"` returns the fat intermediate instead, as the CPU
  plugin does. `rolling_chunk` (default 4, range 1 to 64) and
  `rolling_cache_limit` (default 1) set the chunk size and the number of
  finished chunks kept; they only matter where the accumulation stays on the
  device.
- **Numerics.** The SVD is taken through the FP32 Gram matrix with a cyclic
  Jacobi eigensolver.
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (86–143 dB on the frozen references).
  - The output is run-to-run identical: every group is solved in a fixed
    order, and aggregation is ordered.
  - Groups of up to 8 patches run one thread per group. Larger groups run one
    thread block per group with a round-robin parallel Jacobi.
