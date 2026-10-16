# nss_cuda.NLH

Non-local Haar transform denoising on the device: given or blind sigma, Gray /
YUV / RGB, spatial and temporal.

Algorithm, paper and the reasoning behind the defaults: [`nss.NLH`](../nlh.md).

```python
out = core.nss_cuda.NLH(clip)                          # blind estimation
awgn = core.nss_cuda.NLH(clip, sigma=25)               # explicit sigma
custom = core.nss_cuda.NLH(clip, sigma=25, block_size=[6, 9], block_step=[3, 6])
```

## Speed

1080p, default arguments, RTX 5080, one instance; median over nine test images
(see the [overview](README.md#speed)).

| Call | YUV420 | RGB |
|---|---:|---:|
| default (blind sigma) | 37 | 31 |
| `radius=1` | 13 | 13 |

The blind estimate is a matching pass per frame; give `sigma` when it is known.

## Signature

```python
core.nss_cuda.NLH(clip clip[, float[] sigma, int[] block_size, int[] block_step,
                  int[] group_size, int bm_range, int radius = 0,
                  int ps_num = 2, int ps_range = 4, int[] q, clip rclip = None,
                  string noise_model = "auto", int[] search_window,
                  int basic_iters, float lambda_basic, float hard_strength,
                  int wiener_iters, float wiener_sigma_scale,
                  int memory_limit_mb, string temporal_mode = "rolling",
                  int rolling_chunk = 4, int rolling_cache_chunks = 1,
                  int rolling_cache_limit = 16, int device_id = 0,
                  int num_streams = 1])
```

Two-stage array parameters take `[Basic, Wiener]`; a single value broadcasts
to both stages. Omitted fields come from the preset (table below).

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | omitted = blind | >= 0 | 8-bit noise stddev per plane. When omitted, noise is estimated per frame/channel. Explicit `sigma=0` preserves the plane exactly. |
| `noise_model` | `"auto"` | auto / awgn / real | Kept for compatibility: since model version 6 there is one preset and the value selects nothing. Version 5 chose a preset by it (and by sigma). |
| `block_size` | preset | [2, 16] | `[Basic, Wiener]` patch edge. Smaller Basic blocks track texture; the larger Wiener block stabilizes the second stage. |
| `block_step` | preset | [1, block] | Per-stage stride. Cost scales as `1/step^2` per stage; the Basic stage's step is the dominant cost driver. |
| `group_size` | [16, 16] | {2,4,8,16,32,64} | Pixel-matrix size per group (power of two). |
| `search_window` | preset | [1, 129] | Matching window per stage. `bm_range=r` means `2r+1` on both stages; passing both is an error. |
| `q` | preset | {2,4,8,16}, q <= block^2 | Haar coefficient rows kept per group (power of two). |
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
| `rclip` | none | clip | Reference clip guiding matching. |

### Preset (resolved when fields are omitted)

`[Basic, Wiener]` per field. One preset serves every format, noise model and
sigma:

| block_size | block_step | group_size | search_window | q | basic_iters | wiener_iters | lambda_basic | hard_strength | wiener_sigma_scale |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| [8, 16] | [6, 15] | [16, 16] | [24, 16] | [4, 4] | 3 | 2 | 0.6 | 1.0 | 0.32 |

An omitted block is capped at the smallest processed plane, an omitted step
at its block, and an omitted `q` is halved until it fits `block^2`.

This is the model version 6 preset. Version 5 had three, chosen by noise
model and sigma; to reproduce them pass the fields explicitly:

| Version 5 preset | block_size | block_step | search_window | q | basic_iters | hard_strength | wiener_sigma_scale |
|---|---|---|---|---|---:|---:|---:|
| Gray AWGN, sigma <= 50 | [8, 16] | [6, 15] | [40, 40] | [4, 4] | 4 | 1.0 | 0.32 |
| Gray AWGN, sigma > 50 | [8, 15] | [6, 7] | [40, 40] | [4, 4] | 5 | 0.70710678 | 0.64 |
| Real (RGB/YUV auto) | [7, 16] | [4, 10] | [40, 40] | [2, 4] | 2 | 0.125 | 0.64 |

Against version 5 on the study's test images (PSNR): gray up to sigma 50
+0.05 dB; gray at sigma 75 to 100 -0.5 dB; RGB +3.9 dB at sigma 5, +1.6 at
15, +0.6 at 25, -0.7 at 50, -1.1 to -1.4 at 75 to 100, and -1.3 dB on real
camera noise with blind estimation. For strong or real noise the version 5
values above remain the better choice.

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `basic_iters` | preset | [1, 64] | Basic rounds (iterative hard-threshold refinement). Linear cost driver. |
| `wiener_iters` | 2 | [1, 64] | Times the fixed Wiener gain is re-applied (not full extra Wiener passes). |
| `lambda_basic` | 0.6 | [0, 1] | Mix between the iterated Basic estimate and the input each round. |
| `hard_strength` | preset | >= 0 | Scales the linear Basic threshold `2.025 * hard_strength * sigma_channel`. |
| `wiener_sigma_scale` | preset | >= 0 | Noise scale inside the Wiener gain `r^2 / (r^2 + noise)`. |
| `ps_num` / `ps_range` | 2 / 4 | [1,min(group)] / [1,64] | Predictive temporal search (`radius > 0`). |

## Diagnostics

Each output frame stamps `_NSSSigma`, `_NSSBlockSize`, `_NSSBlockStep`,
`_NSSGroupSize`, `_NSSSearchWindow`, `_NSSQ`, `_NSSIterations`,
`_NSSLambdaBasic`, `_NSSHardStrength`, `_NSSHardCoefficient`,
`_NSSWienerSigmaScale`, and `_NSSModelVersion=6` — the resolved values actually
executed, so blind runs can be audited per frame.

## Device and temporal parameters

| Parameter | Default | Meaning |
|---|---|---|
| `device_id`, `num_streams`, `memory_limit_mb` | 0, 1, none | The GPU, the frames in flight, and a memory cap; see [shared arguments](README.md#shared-by-every-filter). |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"rolling"`, 4, 1, 16 | What `radius > 0` returns (finished frames by default); see [temporal output](README.md#temporal-output). |

## Notes

- **Accuracy.** 82–100 dB against the CPU plugin on the reference set (the
  differences are coefficients on the other side of the hard threshold);
  identical from run to run. The `_NSS*` properties match the CPU, and a
  blind `_NSSSigma` agrees to float precision.
- **Temporal output** is bit for bit the legacy intermediate reduced by
  `VAggregate`. Its frames carry the source's properties, not the `_NSS*`
  diagnostics (as `VAggregate`'s output does).
- **Memory.** Rolling output takes about 80 MiB more than legacy output at
  gray 1080p `radius=1`.

## Pitfalls

- `sigma` omitted means blind estimation with an 8x8 minimum plane bootstrap;
  planes smaller than 8x8 are creation-time errors. The estimate costs a
  matching pass per frame: give `sigma` when it is known.
- Explicit invalid stage combinations (e.g. `q > block^2`) are errors, not
  silent clamps.
- `wiener_iters` re-applies the fixed Wiener gain; it is not a second Wiener
  image pass.
- Legacy temporal intermediates of another model version are rejected by
  `nss.VAggregate`; regenerate them with the current build.
