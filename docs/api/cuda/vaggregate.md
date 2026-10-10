# nss_cuda.VAggregate

Reduces the fat intermediate of a temporal filter (`temporal_mode="legacy"`,
`radius > 0`) to normal-height frames. The device filters return finished
frames by default, so this call is only needed for the legacy form or to mix
backends.

```python
fat = core.nss_cuda.BM3D(clip, sigma=25, radius=1, temporal_mode="legacy")
out = core.nss_cuda.VAggregate(fat, clip, radius=1)
# The intermediate of either plugin is accepted.
out = core.nss_cuda.VAggregate(core.nss.BM3D(clip, sigma=25, radius=1), clip, radius=1)
```

## Signature

```python
core.nss_cuda.VAggregate(clip clip, clip src[, int radius = 0, int[] planes,
                         int allow_legacy = 0, int memory_limit_mb,
                         int device_id = 0, int num_streams = 1])
```

## Parameters

| Parameter | Default | Range | Meaning |
|---|---|---|---|
| `clip` | required | clip | The fat intermediate: the output of a temporal filter with `radius > 0` and `temporal_mode="legacy"` (or of the CPU plugin). Its height is `src` height x `(2 * radius + 1)` x 2. |
| `src` | required | clip | The clip the intermediate was computed from. It gives the output format and size, and the planes that are not aggregated. |
| `radius` | 0 | [1, 16] in practice | The radius the intermediate was produced with. It must match the one recorded in the frame properties. |
| `planes` | all planes | unique plane indices | The planes to aggregate. A plane that is not listed is copied from `src`. |
| `allow_legacy` | 0 | 0 / 1 | 1 also accepts an intermediate without any `_NSSFat*` identity properties (output of old builds). A partially tagged frame is always rejected. |
| `memory_limit_mb` | none | > 0 | Caps the host workspace of the instance. |
| `device_id` | 0 | device index | Accepted for the common signature; no device is used. |
| `num_streams` | 1 | [1, 16] | Accepted for the common signature; no device is used. |

The intermediate carries its model, model version, radius and layout as frame
properties, and a frame whose identity does not match the current build is
rejected: intermediates of another model version (for example NLH before
version 6) must be regenerated.

## On the device

- **`nss_cuda.VAggregate` runs on the host.** Its inputs are already host
  frames, and uploading 2(2R+1) planes per frame costs several times more than
  the sum itself. It sums slices in the CPU's order and divides with IEEE
  rounding.
- **Interop.** `nss.VAggregate` and `nss_cuda.VAggregate` accept each other's
  BM3D output. They agree within a few ulp: the CPU's fast-math division may
  be up to 2 ulp off IEEE rounding.
