# nss_cuda.WNNM

Weighted nuclear norm minimization on the device, spatial and temporal.

Algorithm, paper and the reasoning behind the defaults: [`nss.WNNM`](../wnnm.md).

```python
out = core.nss_cuda.WNNM(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.WNNM(clip, sigma=25, block_step=4, group_size=16)
temporal = core.nss_cuda.WNNM(clip, sigma=25, radius=1)      # finished frames
```

## Cost

Groups above 8 patches cost much more (an eigendecomposition per group), and
`block_step` sets the number of groups (`1/step^2`). Spatial searches of many
shapes take faster dedicated matchers with the same matches (one match
channel): the window search for `bm_range` 7 and more at block 8 (the default
7 included), groups 34 to 64 at any `bm_range`, and other blocks from smaller
or larger windows; and the column search at `block_step` 1 to 3. Where
`block_size` is twice `block_step` (8 / 4, 16 / 8, ...) the square search
takes it, with the same matches up to near ties (the output differs in the
last bits); see [BM3D's optimized search shapes](bm3d.md#optimized-search-shapes).
With `radius > 0` the square search, or else the window search from `bm_range`
9 at block 8, also covers the centre frame.

## Closer to the papers

The defaults are fast settings. Settings closer to the paper, with their quality; they are much slower than the defaults. Measured on one 1080p RGB frame (an anime credits frame with its own film grain) with Gaussian noise added, against the frame without it; the noisy input is 24.97 dB at sigma 15 and 20.98 dB at sigma 25. One image: take these as starting points, not tuned presets. σ is the standard deviation of the noise in 8-bit units.

| Arguments | PSNR, sigma 15 | PSNR, sigma 25 |
|---|---:|---:|
| defaults, `sigma=σ` | 26.29 dB | 22.33 dB |
| defaults, `sigma=2.2 σ` | 32.09 dB | 28.27 dB |
| `block_step=4, group_size=32, bm_range=15, sigma=1.41 σ` | 33.69 dB | 30.42 dB |

- This WNNM runs one pass; the paper iterates 8 or more times with the noise re-injected. One pass needs a larger sigma for the paper's strength, and the factor falls with the group: about 2.2 σ at the defaults (group 8), 2.0 at group 16, 1.8 at group 32 (9 anime frames, σ 1 to 20, within 0.1 dB of the best at every σ). On this frame 1.41 σ kept more text detail at group 32 (2 σ gained 0.3 dB at sigma 15).
- Iterating instead does not pay here: eight rounds of the paper's scheme (11 images) matched one pass at 1.41 σ to within 0.3 dB, at 8 times the cost.
- The group is capped at 32 patches (the paper takes about 70).

## Signature

```python
core.nss_cuda.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int group_size = 8, int bm_range = 7,
                   int radius = 0, int ps_num = 2, int ps_range = 4,
                   int residual = 0, int adaptive_aggregation = 1,
                   clip rclip = None, int memory_limit_mb,
                   string temporal_mode = "rolling", int rolling_chunk = 1,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int device_id = 0, int num_streams])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane (broadcasts). Sets the shrinkage strength. `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. 8 is the balanced default; larger blocks raise SVD cost steeply. |
| `block_step` | 8 | [1, block] | Reference-patch stride. The main quality/speed trade: positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (SVD matrix columns). |
| `bm_range` | 7 | [1, 64] | Search window radius (`2r+1`). With `bm_range` 25 or more, spatial matching of one channel runs on tensor cores on GPUs of compute capability 8.0+ (block 8, group at most 16): the same matches, a workspace of one plane per stream (about 24 MiB at 1080p). |
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
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1 or 2, none | The GPU, the frames in flight ([default streams](README.md#default-streams)), and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 1, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 86–143 dB against the CPU plugin on the reference set (the SVD
  is taken through an FP32 Gram matrix); identical from run to run.
- **Matching.** Spatial search reuses [BM3D's exact window and column matchers](bm3d.md#optimized-search-shapes)
  with the same block, step and group limits. Other spatial shapes and temporal
  search keep their existing matchers.

## Pitfalls

- `group_size=1` degenerates the SVD to per-patch shrinkage; legal but rarely
  useful.
- `temporal_mode="legacy"` output is the taller weighted intermediate and
  needs an `nss.VAggregate` call; the default mode returns finished frames.
- This filter emits no `_NSS*` frame properties.
