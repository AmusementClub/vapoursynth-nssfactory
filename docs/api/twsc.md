# nss.TWSC

Trilateral Weighted Sparse Coding: sparse coding with three
weights — the usual residual weight plus weights derived from the noise
estimate and the current reconstruction — solved with a full ADMM. Aimed at
real-world (signal-dependent) noise where plain AWGN models misestimate.
**The balanced default is deliberately heavy**; see the performance note.

```python
out = core.nss.TWSC(clip, sigma=25)                    # balanced default
fast = core.nss.TWSC(clip, sigma=25, block_step=8, group_size=8, iters=2)
blind = core.nss.TWSC(clip, estimate_sigma=1)          # per-frame sigma
```

## Signature

```python
core.nss.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0,
              int block_size = 8, int block_step = 1, int group_size = 90,
              int search_window = 60, int bm_range, int radius = 0,
              int ps_num = 2, int ps_range = 4, float lambda2 = 1.0,
              clip rclip = None, int iters = 12 (4 with cfast), float delta = 0.0,
              int admm_iter = 10, float rho = 0.5, float mu = 1.1,
              float tol = 1e-6, int cfast = 1,
              int memory_limit_mb])
```

## Primary parameters

| Parameter | Default | Range | Meaning and impact |
|---|---|---|---|
| `sigma` | 3.0 | >= 0 | 8-bit noise stddev per plane. Controls denoising strength only — one geometry is kept for all noise levels. `sigma=0` bypasses the plane. Mutually exclusive with `estimate_sigma`. |
| `estimate_sigma` | 0 | 0 / 1 | Estimate sigma per frame/channel instead of taking it (the NLH estimator: 8x8 blocks on every fourth position per axis). |
| `block_size` | 8 | [1, 16] | Patch edge. |
| `block_step` | 1 | [1, block] | Reference-patch stride. **The default is 1 (every position)** — overriding it is the single largest speed lever: positions scale as `1/step^2`, so `block_step=8` is ~64x fewer groups. |
| `group_size` | 90 | [1, 256] | Matched patches per group. 90 follows the paper's high-quality regime; 8–16 trades quality for large speedups. |
| `search_window` | 60 | [1, 129] | Matching window edge. `bm_range=r` means `2r+1`; passing both is an error. |
| `iters` | 12; 4 with `cfast` | [1, 64] | Outer iterations (re-match and re-solve per round). Nearly linear cost; 12 is the balanced default, 2–4 for quick work. When the fast color path applies the default is 4 (see [Rounds](#fast-color-path-cfast)). |
| `radius` | 0 | [0, 16] | Temporal radius. >0 returns the weighted intermediate for `VAggregate`. |
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
rounds than the joint filter, so **`iters` defaults to 4 when the fast path
applies** (12 otherwise; an explicit `iters` is always taken, and
`_NSSIterations` reports the rounds that ran). Measured on the device plugin
with the default geometry (DIV2K, 18 images, 256x256 crops, PSNR to the clean
image), against the joint filter at its 12 rounds:

| `cfast=1`, rounds | sigma 5 / 15 / 25 / 50 / 75 / 100 | Time |
|---|---|---|
| 3 | -0.01 / -0.05 / +0.02 / +0.21 / +0.37 / +0.44 dB | 0.17 |
| 4 (default) | +0.14 / +0.15 / +0.20 / +0.29 / +0.40 / +0.46 dB | 0.22 |
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
`group_size` is larger than `block_size^2`; with a small group (for example
`group_size=8`) the joint dictionary is no larger, and three planes are three
times the groups, so `cfast=0` is the faster call there.

## Algorithm and paper

Xu, Zhang, Yang & Zhang, *Trilateral Weighted Sparse Coding for Real-world
Image Denoising*, ECCV 2018. The plugin implements the complete three-weight ADMM
objective with the selected author preprocessing; there is no legacy runtime
mode, and the removed `lambda1` parameter is rejected (including zero).

## Defaults rationale and performance

The plugin deliberately uses one stable geometry (`block 8, step 1, group 90, window 60,
iters 12`) across noise levels — sigma controls strength, not geometry. This is
**far** heavier than the geometry used in the plugin's benchmark harness
(`step 8, group 8, iters 2`, ~1337 ms/frame, ~0.75 fps single-core 1080p
GRAYS). The true default has ~64x more reference positions, ~11x larger groups,
and 6x the iterations; it has not been formally timed and should be expected to
take minutes per frame. Override the three levers explicitly for interactive
work.

## Pitfalls

- The balanced default is a quality-first configuration, not a realtime one —
  always check whether you actually want `block_step=1`.
- `estimate_sigma` and explicit `sigma` are mutually exclusive (creation-time
  error).
- `ps_num > group_size` is an error after resolution.

## CUDA

`core.nss_cuda.TWSC` takes the same arguments. What is specific to the device
(extra arguments, temporal output, memory, numerics, speed) is in the
[CUDA page](cuda/twsc.md).
