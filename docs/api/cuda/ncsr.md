# nss_cuda.NCSR

Nonlocally centralized sparse representation on the device, spatial and temporal.

Algorithm, paper and the reasoning behind the defaults: [`nss.NCSR`](../ncsr.md).

更新後的 `r1` 附件採用與 CPU 相同的 NCSR 模型版本 3 方差修正：保留加權中心，
以實際匹配欄位計算未加權二階矩，排除 CUDA 補齊欄位。
遷移方式與已知畫質限制見 [CPU／CUDA 共用模型說明](../ncsr.md)。
下列既有速度與精度數據屬修正前的量測，不能直接視為 v3 的驗證結果。
原始 `r1` 附件為 v2；發行頁列出修正版來源 commit 與 SHA-256。

```python
out = core.nss_cuda.NCSR(clip, sigma=25)                    # defaults
hiq = core.nss_cuda.NCSR(clip, sigma=25, block_step=4)
temporal = core.nss_cuda.NCSR(clip, sigma=25, radius=1)  # finished frames, every round on the device
```

## Cost

Larger groups and a smaller `block_step` cost much more, and every extra round
(`iters`) adds about a full pass. Spatial searches of some shapes take faster
dedicated matchers with the same matches (one match channel, `group_size` up
to 32): the window search for `bm_range` up to 16 and at least 7 at block 8
(the default 7 included), 4 at blocks 3 to 6, 5 at block 7 and 6 at block 2,
any `block_step`; and the column search at `block_step` 1 to 3; see [BM3D's
optimized search shapes](bm3d.md#optimized-search-shapes). With `radius > 0`
the window search also covers the centre frame from `bm_range` 9 at block 8.

## Closer to the papers

The defaults are fast settings. Settings closer to the paper, with their quality; they are much slower than the defaults. Measured on one 1080p RGB frame (an anime credits frame with its own film grain) with Gaussian noise added, against the frame without it; the noisy input is 24.97 dB at sigma 15 and 20.98 dB at sigma 25. One image: take these as starting points, not tuned presets. σ is the standard deviation of the noise in 8-bit units.

| Arguments | PSNR, sigma 15 | PSNR, sigma 25 |
|---|---:|---:|
| defaults, `sigma=σ` | 29.42 dB | 25.59 dB |
| defaults, `sigma=1.8 σ` | 33.04 dB | 29.67 dB |
| `block_step=4, group_size=16, sigma=σ` | 33.41 dB | 30.37 dB |

- At the default geometry (group 8, 2 rounds) about 1.8 σ is best (9 anime frames, σ 1 to 20, within 0.1 dB of the best at every σ). The factor falls as the group and the rounds grow: about 1.3 σ at group 16 and σ as is at group 32.

## Signature

```python
core.nss_cuda.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int group_size = 8, int bm_range = 7,
                   int radius = 0, int ps_num = 2, int ps_range = 4,
                   clip rclip = None, int iters = 2, float delta = 0.1,
                   int memory_limit_mb, string temporal_mode = "rolling",
                   int rolling_chunk = 4, int rolling_cache_chunks = 1,
                   int rolling_cache_limit = 16, int device_id = 0,
                   int num_streams])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Drives the adaptive soft-threshold (`tau ~ sigma^2 / sigma_row`). `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (columns of the PCA matrix). |
| `bm_range` | 7 | [1, 64] | Search window radius. With `bm_range` 20 or more, spatial matching of one channel runs on tensor cores on GPUs of compute capability 8.0+ (block 8, group at most 16): the same matches, a workspace of one plane per stream (about 24 MiB at 1080p). |
| `iters` | 2 | [1, 64] | Outer re-estimation rounds; the second round re-matches on the current estimate. ~2x cost per extra round. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `delta` | 0.1 | [0, 1] | Relaxation mixing previous and current estimates between rounds. |
| `ps_num` / `ps_range` | 2 / 4 | [1,group] / [1,64] | Predictive temporal search (with `radius > 0`). |

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 2 or 3 (1 on GPU frames), none | The GPU, the frames in flight ([default streams](README.md#default-streams)), and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 105–139 dB against the CPU plugin on the reference set;
  identical from run to run.
- **Matching.** Spatial search reuses [BM3D's exact window and column matchers](bm3d.md#optimized-search-shapes)
  with the same block, step and group limits. Other spatial shapes and temporal
  search keep their existing matchers.
- **Temporal output** runs every round on the GPU and agrees with the legacy
  intermediate reduced by `VAggregate` to 142–152 dB.
- **Memory.** At gray 1080p `radius=1` an instance takes about 310 MiB of GPU
  memory; the smallest accepted `memory_limit_mb` is 405 MiB. Groups of 17 to
  32 patches add 14.5 KiB of scratch per group of a batch.

## Pitfalls

- `iters=1` skips the second matching round: a sizeable speedup and a visible
  quality change on textured noise.
- `temporal_mode="legacy"` output needs an `nss.VAggregate` call; the default mode
  returns finished frames.
- `temporal_mode="legacy"` 中間結果帶有 `_NSSFatVersion=2` 與
  `_NSSModelVersion=3`；必須搭配支援 NCSR v3 的 `nss.VAggregate`。
