# nss_cuda.WNNM

Weighted nuclear norm minimization on the device, spatial and temporal.

Algorithm, paper and the reasoning behind the defaults: [`nss.WNNM`](../wnnm.md).

```python
out = core.nss_cuda.WNNM(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.WNNM(clip, sigma=25, block_step=4, group_size=16)
temporal = core.nss_cuda.WNNM(clip, sigma=25, radius=1)      # finished frames
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default | 500 | 285 |
| `radius=1` | 455 | 230 |

Groups above 8 patches cost much more (an eigendecomposition per group): at
1080p gray, `group_size=16` runs at about 330 fps and `group_size=32,
block_step=4` at 14 fps.

## Signature

```python
core.nss_cuda.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int group_size = 8, int bm_range = 7,
                   int radius = 0, int ps_num = 2, int ps_range = 4,
                   int residual = 0, int adaptive_aggregation = 1,
                   clip rclip = None, int memory_limit_mb,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int device_id = 0, int num_streams = 1])
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

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 86–143 dB against the CPU plugin on the reference set (the SVD
  is taken through an FP32 Gram matrix); identical from run to run.

## Pitfalls

- `group_size=1` degenerates the SVD to per-patch shrinkage; legal but rarely
  useful.
- `temporal_mode="legacy"` output is the taller weighted intermediate and
  needs an `nss.VAggregate` call; the default mode returns finished frames.
- This filter emits no `_NSS*` frame properties.
