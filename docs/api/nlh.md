# nss.NLH

Non-Local Haar: pixel-level non-local matching with Haar
thresholding, iterated Basic rounds plus a Wiener stage with repeated gains,
and full pixel aggregation. The plugin's blind real-world denoiser — omit
`sigma` and each frame/channel's noise is estimated automatically. The heaviest
filter here at defaults.

```python
out = core.nss.NLH(clip)                          # blind estimation
awgn = core.nss.NLH(clip, sigma=25)               # explicit sigma (Gray AWGN lane)
custom = core.nss.NLH(clip, sigma=25, block_size=[6, 9], block_step=[3, 6])
```

Two-stage array parameters take `[Basic, Wiener]`; a single value broadcasts
to both stages. Omitted fields are resolved per lane below.

## Signature

```python
core.nss.NLH(clip clip[, float[] sigma, string noise_model = "auto",
             int[] block_size, int[] block_step, int[] group_size,
             int[] search_window, int bm_range, int radius = 0,
             int ps_num = 2, int ps_range = 4, int[] q, clip rclip = None,
             int basic_iters, float lambda_basic, float hard_strength,
             int wiener_iters, float wiener_sigma_scale,
             int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | omitted = blind | >= 0 | 8-bit noise stddev per plane. When omitted, noise is estimated per frame/channel and the lane is chosen from the estimate. Explicit `sigma=0` preserves the plane exactly. |
| `noise_model` | `"auto"` | auto / awgn / real | `auto` = AWGN for Gray, real for RGB/YUV. AWGN picks the low/high lane by the center frame's maximum working-domain sigma (<= 50 / > 50). |
| `block_size` | lane table | [2, 16] | `[Basic, Wiener]` patch edge. Smaller Basic blocks track texture; the larger Wiener block stabilizes the second stage. |
| `block_step` | lane table | [1, block] | Per-stage stride. Cost scales as `1/step^2` per stage; the Basic stage's step is the dominant cost driver. |
| `group_size` | [16, 16] | {2,4,8,16,32,64} | Pixel-matrix size per group (power of two). |
| `search_window` | [40, 40] | [1, 129] | Matching window per stage. `bm_range=r` means `2r+1` on both stages; passing both is an error. |
| `q` | lane table | {2,4,8,16}, q <= block^2 | Haar coefficient rows kept per group (power of two). |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |
| `rclip` | none | clip | Reference clip guiding matching. |

### Lane defaults (resolved when fields are omitted)

`[Basic, Wiener]` per field. Shared: `group_size=[16,16]`,
`search_window=[40,40]`, `lambda_basic=0.6`, `wiener_iters=2`.

| Lane | block_size | block_step | q | basic_iters | hard_strength | wiener_sigma_scale |
|---|---|---|---|---:|---:|---:|
| Gray AWGN, sigma <= 50 | [8, 16] | [6, 15] | [4, 4] | 4 | 1.0 | 0.32 |
| Gray AWGN, sigma > 50 | [8, 15] | [6, 7] | [4, 4] | 5 | 0.70710678 | 0.64 |
| Real (RGB/YUV auto) | [7, 16] | [4, 10] | [2, 4] | 2 | 0.125 | 0.64 |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `basic_iters` | lane table | [1, 64] | Basic rounds (iterative hard-threshold refinement). Linear cost driver. |
| `wiener_iters` | 2 | [1, 64] | Times the fixed Wiener gain is re-applied (not full extra Wiener passes). |
| `lambda_basic` | 0.6 | [0, 1] | Mix between the iterated Basic estimate and the input each round. |
| `hard_strength` | lane table | >= 0 | Scales the linear Basic threshold `2.025 * hard_strength * sigma_channel`. |
| `wiener_sigma_scale` | lane table | >= 0 | Noise scale inside the Wiener gain `r^2 / (r^2 + noise)`. |
| `ps_num` / `ps_range` | 2 / 4 | [1,min(group)] / [1,64] | Predictive temporal search (`radius > 0`). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Diagnostics

Each output frame stamps `_NSSSigma`, `_NSSBlockSize`, `_NSSBlockStep`,
`_NSSGroupSize`, `_NSSSearchWindow`, `_NSSQ`, `_NSSIterations`,
`_NSSLambdaBasic`, `_NSSHardStrength`, `_NSSHardCoefficient`,
`_NSSWienerSigmaScale`, and `_NSSModelVersion=5` — the resolved values actually
executed, so blind runs can be audited per frame.

## Algorithm and paper

Hou et al., *NLH: A Blind Pixel-level Non-local Method for Real-world Image
Denoising* (arXiv:1906.06834), author's reference code at commit `e36d833b`.
This implementation keeps the paper's structure (iterative Basic + repeated Wiener gains + full
pixel aggregation) with a calibrated **linear** Basic threshold
`k_h * hard_strength * sigma_channel` (`k_h = 2.025`). Deliberate departures:
fixed per-stage shapes instead of Normal/Fast shape switching, one Wiener image
pass with repeated gains instead of two MEX calls, and BT.601 working domain
with variance propagation instead of the author's opponent-transform sigma
multipliers.

## Defaults rationale and performance

The preset lanes come from a joint quality/speed search over DIV2K and CC
real-noise pairs. Low-noise blind estimation knowingly trades detail for the
large high-noise/real-noise gains — on nearly-clean sources (sigma around 5),
expect visible detail loss with the real-noise lane in particular. Benchmark-geometry
(block 8/8, step 8/8) runs at ~6621 ms/frame single-core 1080p GRAYS
(~0.15 fps); the true defaults with blind estimation are heavier. The Basic
stage dominates; blind estimation adds a full step-1 matching pass per frame.

## Pitfalls

- Temporal contribution buffers from older plugin builds are not interchangeable; regenerate intermediates with the current build.
- `sigma` omitted means blind estimation with an 8x8 minimum plane bootstrap;
  planes smaller than 8x8 are creation-time errors.
- Explicit invalid stage combinations (e.g., `q > block^2`) are errors, not
  silent clamps.
- The Wiener stage's `wiener_iters` re-applies the fixed gain; it is not a
  second Wiener image pass.

## CUDA

`core.nss_cuda.NLH` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/nlh.md).
