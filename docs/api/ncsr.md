# nss.NCSR

Nonlocally Centralized Sparse Representation: patch groups are projected onto
a PCA basis learned per group, the coefficients are centralized (weighted
mean removed) and soft-thresholded against a per-row noise-adaptive threshold,
then reconstructed. Sparse-coding quality between WNNM and LSSC in both cost
and behaviour.

```python
out = core.nss.NCSR(clip, sigma=25)                    # defaults
hiq  = core.nss.NCSR(clip, sigma=25, block_step=4)
```

## Signature

```python
core.nss.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8,
              int block_step = 8, int group_size = 8, int bm_range = 7,
              int radius = 0, int ps_num = 2, int ps_range = 4,
              clip rclip = None, int iters = 2, float delta = 0.1,
              int memory_limit_mb, string temporal_mode = "legacy",
              int rolling_chunk = 4, int rolling_cache_chunks,
              int rolling_cache_limit])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Drives the adaptive soft-threshold (`tau ~ sigma^2 / sigma_row`). `sigma=0` bypasses the plane. |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 8 | [1, block] | Reference-patch stride; positions scale as `1/step^2`. |
| `group_size` | 8 | [1, 32] | Matched patches per group (columns of the PCA matrix). |
| `bm_range` | 7 | [1, 64] | Search window radius. |
| `iters` | 2 | [1, 64] | Outer re-estimation rounds; the second round re-matches on the current estimate. ~2x cost per extra round. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`, or finished frames with `temporal_mode="rolling"` ([Temporal output](README.md#temporal-output)). |
| `rclip` | none | clip | Reference clip guiding matching. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `delta` | 0.1 | [0, 1] | Relaxation mixing previous and current estimates between rounds. |
| `ps_num` / `ps_range` | 2 / 4 | [1,group] / [1,64] | Predictive temporal search (with `radius > 0`). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"legacy"`, 4, 1, 1 | — | What `radius > 0` returns: see [Temporal output](README.md#temporal-output). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Algorithm and paper

Dong, Zhang, Shi & Li, *Nonlocally Centralized Sparse Representation for Image
Restoration*, IEEE TIP 2013. Per group: PCA projection of the patch matrix,
weighted centralization of coefficients, noise-adaptive soft thresholding, and
reconstruction with the group mean added back. The batch implementation keeps
bitwise-exact reduction orders across ISA lanes. A separate higher-quality
`HQ` path exists in the tree but is not part of this public filter.

## 模型語義版本 3（r1 附件修正）

CPU 與 CUDA 使用相同的方差修正。對每個 PCA 係數列，令實際匹配
區塊數為 `n`、區塊面積為 `m`、內部雜訊標準差為 `s = sigma / 255`：

```text
w[j] = exp(-distance[j] / max(2*m*s*s, 1e-12))
beta = sum(w[j]*code[j]) / sum(w[j])
v = sum((code[j]-beta)^2) / n
tau = 2*sqrt(2)*s*s / (sqrt(v) + 1e-12)
output[j] = beta + soft(code[j]-beta, tau)
```

`beta` 仍是距離加權平均；`v` 改為實際係數對 `beta` 的未加權二階矩。
CUDA 的補齊欄位不計入總和或分母。不扣除雜訊方差，也不調整預設參數、
門檻常數、PCA、迭代或彙整方式。

舊版在低 sigma 或獨特邊緣附近可能只剩參考區塊的權重，讓加權方差趨近
零、收縮門檻過大，進而把其他區塊推向參考區塊。此修正保留整組係數的
差異，改善文字筆畫被複製或破壞的情況；不保證所有雜訊強度都提升畫質。
在三張 DIV2K 裁切圖、RGB／灰階、兩個亂數種子的既有測試中，CUDA 於
sigma 10／25 的平均 PSNR 相對舊版分別下降約 0.366／0.167 dB。

時間域中間結果標記 `_NSSModelVersion=3`，資料布局仍為
`_NSSFatVersion=2`。新版本 `VAggregate` 拒絕 NCSR v2 中間結果；升級時
請由來源影格重新產生，不可只改標記，也不可混用舊版 CPU／CUDA 外掛。
公開函式簽名、sigma 單位及 sigma=0 的原樣輸出規則不變。
此模型版本與發行版本名稱分開管理。更新後的 `r1` 附件採用 v3；
原始 `r1` 附件為 v2，請依發行頁的更新日期、來源 commit 與 SHA-256 區分。

回歸測試：`test_ncsr` 以獨立 FP64 公式驗證兩種 CPU 係數布局；
`test_ncsr_variance_plugin` 與 `test_cuda_ncsr_variance_plugin` 驗證低 sigma
文字邊緣、補齊欄位、重複執行一致性與時間域版本拒絕規則。

## Defaults rationale and performance

Shared geometry (8/8/8/7) + two rounds, matching the calibration matrix.
~268 ms per 1080p GRAYS frame single-core (~3.7 fps). Matching cost scales
with `1/step^2 * bm_range^2`; per-group cost scales with the PCA/SVD solve at
`block^2 x group`.

## Pitfalls

- `iters=1` skips the second matching round — sizeable speedup, visible
  quality change on textured noise; measure before shipping.
- `radius > 0` output needs `VAggregate` (or `temporal_mode="rolling"`).
- Temporal intermediates carry `_NSSFat*` and `_NSSModel*` properties; see the model-version notes above.

## CUDA

`core.nss_cuda.NCSR` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/ncsr.md).
