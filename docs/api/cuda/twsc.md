# nss_cuda.TWSC

Trilateral weighted sparse coding on the device: given or blind sigma, Gray / YUV / RGB, spatial and temporal.

Every argument is described on this page. The algorithm, the paper and the
reasoning behind the defaults are on the CPU page, [`nss.TWSC`](../twsc.md); the
model and the defaults are the same on both plugins.

```python
out = core.nss_cuda.TWSC(clip, sigma=25)                    # balanced default
fast = core.nss_cuda.TWSC(clip, sigma=25, block_step=8, group_size=8, iters=2)
blind = core.nss_cuda.TWSC(clip, estimate_sigma=1)          # per-frame sigma
```

## Signature

```python
core.nss_cuda.TWSC(clip clip[, float[] sigma = 3.0, int estimate_sigma = 0,
                   int block_size = 8, int block_step = 1, int group_size = 90,
                   int search_window = 60, int bm_range, int radius = 0,
                   int ps_num = 2, int ps_range = 4, float lambda2 = 1.0,
                   clip rclip = None, int iters = 12, float delta = 0.0,
                   int admm_iter = 10, float rho = 0.5, float mu = 1.1,
                   float tol = 1e-6, int cfast = 1,
                   string temporal_mode = "rolling", int rolling_chunk = 4,
                   int rolling_cache_chunks = 1, int rolling_cache_limit = 16,
                   int memory_limit_mb, int device_id = 0, int num_streams = 1])
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
| `iters` | 12 | [1, 64] | Outer iterations (re-match and re-solve per round). Nearly linear cost; 12 is the balanced default, 2–4 for quick work. |
| `radius` | 0 | [0, 16] | Temporal radius. >0 makes the filter temporal: finished, normal-height frames by default (`temporal_mode`, below). |
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

## Diagnostics

Each output frame stamps: `_NSSSigma` (per-plane, in 8-bit units),
`_NSSGroups`, `_NSSADMMMaxIterGroups`, `_NSSSvdDoubleGroups`,
`_NSSSylvesterResidual`, `_NSSBlockSize`, `_NSSGroupSize`, `_NSSBlockStep`,
`_NSSSearchWindow`, `_NSSIterations`, `_NSSCFast` (1 when the fast color
path ran) and `_NSSModelVersion=5`.

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

**Rounds.** A plane of the fast path behaves like a Gray clip: it needs far
fewer rounds than the joint filter. Measured on the device plugin (DIV2K, 18
images, 256x256 crops, PSNR to the clean image, sigma 5 / 15 / 25 / 50),
against the joint filter with the same arguments:

| Arguments (`block_step=4` in all) | `cfast=1` against `cfast=0` |
|---|---|
| defaults, `iters=12` | -0.54 / -0.71 / -0.78 / -0.87 dB |
| defaults, `iters=5` | +2.11 / +2.85 / +3.12 / +3.33 dB |
| `block_size=4, group_size=32, search_window=20, iters=3` | +1.79 / +2.97 / +3.52 / +4.27 dB |

With `iters=5` the fast path is 0.1 to 0.2 dB above the joint filter's 12
rounds; with the default 12 rounds it over-smooths. **Set `iters` to 3 to 5
when the fast path applies**, or `cfast=0` to keep the joint result. Filtering
R, G and B alone, without the transform, measured 2 dB below the joint
filter.

**When it pays.** A group's dictionary has `min(rows, group_size)` atoms, with
`rows = block_size^2` per channel. The fast path shrinks it only when
`group_size` is larger than `block_size^2`; with a small group (for example
`group_size=8`) the joint dictionary is no larger, and three planes are three
times the groups, so `cfast=0` is the faster call there.

On the device the transform and its inverse are one kernel each on the
uploaded planes, and the three planes run one after the other on the
instance's stream. RGB 1080p on an RTX 5080, `sigma=3`, `block_size=4,
block_step=4, group_size=32, search_window=20`: 9.8 fps at `iters=2` and 6.6
at `iters=3` with the fast path, 6.1 and 4.1 without. The joint filter needs
`group_size=64` to reach the same quality at that block size (2.4 and 1.65
fps). The default call on a 128x128 RGB frame takes 1.6 s with the fast path
and 2.3 s without; `block_step=8, group_size=8, iters=2` at 1080p runs at 33
fps with it and 36 without (see "When it pays").

## Device parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `device_id` | 0 | device index | CUDA device the instance runs on. One instance uses one device; `core.nss_cuda.Backend(device_id)` reports whether it is supported. |
| `num_streams` | 1 | [1, 16] | Frames (or temporal chunks) the instance has in flight on the device at once. Each stream owns its device buffers, so memory grows with it. |
| `memory_limit_mb` | none | > 0 | Caps the instance's device and pinned host memory. The internal batches are fitted to it and the output does not change; a limit that cannot hold the streams is a creation error. Without it the filter takes what its plan needs. |
| `temporal_mode` | `"rolling"` | rolling / legacy | With `radius > 0`: `"rolling"` returns finished, normal-height frames; `"legacy"` returns the fat intermediate for `VAggregate`, as the CPU plugin does. |
| `rolling_chunk` | 4 | [1, 64] | Frames accumulated per rolling chunk. |
| `rolling_cache_chunks` | 1 | [1, 64] | Finished chunks kept for later frame requests at first. Given alone, the cache stays at this size. |
| `rolling_cache_limit` | 16 | [1, 64], >= `rolling_cache_chunks` | What the chunk cache may grow to: it keeps one more chunk each time two requests miss on chunks it dropped recently. Under `memory_limit_mb` it grows only into what the limit leaves. |

How the streams, the memory limit and the rolling output behave across filters is in the [shared notes](README.md#shared-by-every-filter).

## On the device

- **Device-resident.** The blind noise estimate, the joint matching, the
  per-group dictionary and ADMM solve, the aggregation and every round run on
  the device.
- **Temporal output.** The fat intermediate is reduced with
  `nss_cuda.VAggregate` inside the filter.
- **Numerics.** The dictionary comes from an FP32 eigendecomposition of the
  Gram matrix of the smaller group side (Householder tridiagonalization and
  implicit QL), and the solver runs in FP32 (FP64 was measured and gave the
  same agreement with the CPU).
  - The output is not bit-identical to the CPU, but is inside the 60 dB gate
    (93–143 dB on the frozen references, 142 dB on the fast color path) and
    run-to-run identical.
- **Frame properties.** `_NSSSigma`, `_NSSGroups` and the shape properties
  match the CPU. `_NSSADMMMaxIterGroups` counts this backend's own
  non-converged groups; `_NSSSvdDoubleGroups` and `_NSSSylvesterResidual` are
  CPU-solver diagnostics and are reported as 0.
- **Speed.** The default settings remain very heavy (one 64x64
  eigendecomposition per pixel position per round): about 0.5 s for a 128x128
  frame on an RTX 5080 (1.6 s in RGB; 2.3 s with `cfast=0`), against about
  190 s on 16 CPU threads. The fast call above (`block_step=8, group_size=8,
  iters=2`) runs 1080p at about 97 fps in gray and 33 fps in RGB (36 with
  `cfast=0`).
- **Memory.** Besides its device buffers an instance keeps pinned host
  staging for two frames in flight, so that one stream stays busy while
  another thread copies; under a `memory_limit_mb` that leaves no room for
  the second set it runs with one.

## Pitfalls

- The balanced default is a quality-first configuration, not a realtime one:
  always check whether you actually want `block_step=1`.
- `estimate_sigma` and explicit `sigma` are mutually exclusive (creation-time
  error).
- `ps_num > group_size` is an error after resolution.
- `temporal_mode="legacy"` output needs a `VAggregate` call; the default mode
  returns finished frames.
