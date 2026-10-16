# nss.TWSC

Trilateral Weighted Sparse Coding: sparse coding with three
weights — the usual residual weight plus weights derived from the noise
estimate and the current reconstruction — solved with a full ADMM. Aimed at
real-world (signal-dependent) noise where plain AWGN models misestimate.
The default is a fast geometry; the paper's quality geometry costs thousands
of times more (see [Defaults](#defaults-rationale-and-performance)).

```python
out = core.nss.TWSC(clip, sigma=25)                    # fast default
best = core.nss.TWSC(clip, sigma=25, block_step=4, group_size=90, iters=12)   # quality
blind = core.nss.TWSC(clip, estimate_sigma=1)          # per-frame sigma
```

## Signature

```python
core.nss.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0,
              int block_size = 8, int block_step = 8, int group_size = 8,
              int search_window = 60, int bm_range, int radius = 0,
              int ps_num = 2, int ps_range = 4, float lambda2 = 1.0,
              clip rclip = None, int iters = 2, float delta = 0.0,
              int admm_iter = 10, float rho = 0.5, float mu = 1.1,
              float tol = 1e-6, int cfast = 1,
              string temporal_mode = "legacy", int rolling_chunk = 4,
              int rolling_cache_chunks, int rolling_cache_limit, int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Controls denoising strength only — one geometry is kept for all noise levels. `sigma=0` bypasses the plane. Mutually exclusive with `estimate_sigma`. |
| `estimate_sigma` | 0 | 0 / 1 | Estimate sigma per frame/channel instead of taking it (the NLH estimator: 8x8 blocks on every fourth position per axis). |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | min(8, block) | [1, block] | Reference-patch stride: positions scale as `1/step^2`. 4 keeps the quality of 1 (within 0.1 dB) at a sixteenth of the groups. |
| `group_size` | 8 | [1, 256] | Matched patches per group. 90 follows the paper's high-quality regime (see [Defaults](#defaults-rationale-and-performance)). |
| `search_window` | 60 | [1, 129] | Matching window edge. `bm_range=r` means `2r+1`; passing both is an error. |
| `iters` | 2 | [1, 64] | Outer iterations (re-match and re-solve per round), nearly linear cost. With `group_size=90`: 12 for the joint filter, 4 on the fast color path (see [Rounds](#fast-color-path-cfast)). |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`, or finished frames with `temporal_mode="rolling"` ([Temporal output](README.md#temporal-output)). |
| `rclip` | none | clip | Reference clip guiding matching. |
| `cfast` | 1 | 0 / 1 | Fast color path: an RGB frame whose three channels share one given `sigma` is filtered as three opponent-color planes, each alone, instead of as stacked three-channel patches. See [Fast color path](#fast-color-path-cfast). `cfast=0` keeps the joint filter. |

## Secondary parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `lambda2` | 1.0 | >= 0 | Weight on the reconstruction-fidelity side of the three-weight objective. |
| `delta` | 0.0 | [0, 1] | Relaxation mixing between iterations (0 = no mixing, the default). |
| `admm_iter` | 10 | [1, 1000] | Inner ADMM iterations per group solve. |
| `rho` | 0.5 | > 0 | Initial ADMM penalty. |
| `mu` | 1.1 | >= 1 | Penalty growth factor per inner iteration. |
| `tol` | 1e-6 | > 0 | ADMM convergence tolerance (early stop). |
| `ps_num` / `ps_range` | 2 / 4 | [1,group] / [1,64] | Predictive temporal search (`radius > 0`). `ps_num` defaults to 1 when `group_size=1` is explicit. |
| `temporal_mode`, `rolling_chunk`, `rolling_cache_chunks`, `rolling_cache_limit` | `"legacy"`, 4, 1, 1 | — | What `radius > 0` returns: see [Temporal output](README.md#temporal-output). |
| `memory_limit_mb` | none | — | Workspace cap; fails instead of degrading. |

## Diagnostics

Each output frame stamps: `_NSSSigma` (per-plane, in 8-bit units),
`_NSSGroups`, `_NSSADMMMaxIterGroups`, `_NSSSvdDoubleGroups`,
`_NSSSylvesterResidual`, `_NSSBlockSize`, `_NSSGroupSize`, `_NSSBlockStep`,
`_NSSSearchWindow`, `_NSSIterations`, `_NSSCFast` (1 when the fast color
path ran) and `_NSSModelVersion=6`.

## Fast color path (`cfast`)

The joint filter stacks the three channels of a patch, so an RGB group has
three times the rows of a gray one (192 against 64 at `block_size=8`) and its
dictionary is that much larger. With `cfast=1` (the default) an RGB frame is
instead:

1. taken through an orthonormal opponent transform, `(R+G+B)/sqrt 3`,
   `(R-B)/sqrt 2`, `(R-2G+B)/sqrt 6`. White noise of one sigma on the three
   channels is white noise of the same sigma on the three planes;
2. filtered plane by plane, each with its own matching, groups and rounds
   (as a Gray clip of that plane would be, with the same arguments);
3. transformed back after each plane's estimate is finished.

It applies when all of these hold, and the joint filter runs otherwise:

- the clip is RGB,
- `sigma` is given (not `estimate_sigma=1`), positive, and the same for the
  three channels,
- `radius=0` (the weighted intermediate of `radius > 0` has one denominator
  per plane and cannot be transformed back).

`rclip` goes through the same transform. `_NSSCFast` on the frame is 1 when
the path ran. `_NSSGroups` counts the groups of the three planes.

**Rounds.** A plane of the fast path behaves like a Gray clip and needs fewer
rounds than the joint filter. With the quality geometry (`block_step=1,
group_size=90`) 4 rounds are best on the fast path against 12 for the joint
filter; the default 2 rounds apply to both. Measured on the device plugin
with that geometry (DIV2K, 18 images, 256x256 crops, PSNR to the clean
image), against the joint filter at its 12 rounds:

| `cfast=1`, rounds | sigma 5 / 15 / 25 / 50 / 75 / 100 | Time |
|---|---|---|
| 3 | -0.01 / -0.05 / +0.02 / +0.21 / +0.37 / +0.44 dB | 0.17 |
| 4 | +0.14 / +0.15 / +0.20 / +0.29 / +0.40 / +0.46 dB | 0.22 |
| 5 | +0.12 / +0.10 / +0.11 / +0.18 / +0.27 / +0.33 dB | 0.27 |
| 8 | -0.15 / -0.26 / -0.27 / -0.28 / -0.19 / -0.10 dB | 0.43 |
| 12 | -0.56 / -0.76 / -0.83 / -0.97 / -0.78 / -0.60 dB | 0.64 |

At 4 rounds 15 of the 108 cases are below the joint filter, by up to 1.2 dB,
most of them on one image (0.7 dB on average over its sigma levels);
`cfast=0` keeps the joint result. With a smaller geometry the gap to the joint
filter at the same arguments is larger: `block_size=4, block_step=4,
group_size=32, search_window=20, iters=3` measured +1.79 / +2.97 / +3.52 /
+4.27 dB at sigma 5 / 15 / 25 / 50. Filtering R, G and B alone, without the
transform, measured 2 dB below the joint filter.

**When it pays.** A group's dictionary has `min(rows, group_size)` atoms, with
`rows = block_size^2` per channel. The fast path shrinks it only when
`group_size` is larger than `block_size^2`; with a small group (such as the
default `group_size=8`) the joint dictionary is no larger, and three planes
are three times the groups, so `cfast=0` is the faster call there.

## Algorithm and paper

Xu, Zhang, Yang & Zhang, *Trilateral Weighted Sparse Coding for Real-world
Image Denoising*, ECCV 2018. The plugin implements the complete three-weight ADMM
objective with the selected author preprocessing; there is no legacy runtime
mode, and the removed `lambda1` parameter is rejected (including zero).

## Defaults rationale and performance

The default is the fast geometry `block_size=8, block_step=8, group_size=8,
iters=2`, the same for every noise level (sigma sets the strength): ~1337 ms
per 1080p gray frame on one CPU core (~0.75 fps).

It is a speed setting. The paper's high-quality regime, `block_step=1,
group_size=90, iters=12` (4 on the fast color path), denoises 8 to 10 dB
better at sigma 5 to 100 on DIV2K crops, at thousands of times the cost
(tens of minutes per 1080p frame on one CPU core). In between:

| Call | Against the quality call (gray / RGB) |
|---|---|
| `block_step=4, group_size=90, iters=12` | +0.01 / -0.08 dB, 16x fewer groups |
| `block_step=4, group_size=90, iters=5` (gray) | +0.9 dB gray, 2.3x fewer rounds |
| `block_step=8, group_size=90, iters=5` (gray) | +0.8 dB gray |

RGB needs its 12 rounds at `group_size=90` (5 rounds lose 2.6 dB there).

## Pitfalls

- The default is the fast geometry; for quality pass the larger group and
  more rounds (see [Defaults](#defaults-rationale-and-performance)).
- `estimate_sigma` and explicit `sigma` are mutually exclusive (creation-time
  error).
- `ps_num > group_size` is an error after resolution.

## CUDA

`core.nss_cuda.TWSC` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/twsc.md).
