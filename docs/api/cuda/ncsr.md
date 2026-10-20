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

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default | 390 | 215 |
| `radius=1` | 265 | 135 |

Larger groups cost more: at 1080p gray, `group_size=16` runs at about 180 fps,
`group_size=32` at 50 (13 with `block_step=4`).

## Signature

```python
core.nss_cuda.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8,
                   int block_step = 8, int group_size = 8, int bm_range = 7,
                   int radius = 0, int ps_num = 2, int ps_range = 4,
                   clip rclip = None, int iters = 2, float delta = 0.1,
                   int memory_limit_mb, string temporal_mode = "rolling",
                   int rolling_chunk = 4, int rolling_cache_chunks = 1,
                   int rolling_cache_limit = 16, int device_id = 0,
                   int num_streams = 1])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Drives the adaptive soft-threshold (`tau ~ sigma^2 / sigma_row`). `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (columns of the PCA matrix). |
| `bm_range` | 7 | [1, 64] | Search window radius. With `bm_range` 20 or more, spatial matching of one channel runs on tensor cores on GPUs of compute capability 8.0+ (block 8, group at most 16): the same matches, a workspace of one plane per stream (about 24 MiB at 1080p); BM3D gray 1080p `bm_range=30` 1.6x. |
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
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 105–139 dB against the CPU plugin on the reference set;
  identical from run to run.
- **Temporal output** runs every round on the GPU and agrees with the legacy
  intermediate reduced by `VAggregate` to 142–152 dB.
- **Memory.** At gray 1080p `radius=1` an instance takes about 310 MiB of GPU
  memory; the smallest accepted `memory_limit_mb` is 405 MiB. Groups of 17 to
  32 patches add 14.5 KiB of scratch per group of a batch.

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
- `temporal_mode="legacy"` 中間結果帶有 `_NSSFatVersion=2` 與
  `_NSSModelVersion=3`；必須搭配支援 NCSR v3 的 `nss.VAggregate`。
