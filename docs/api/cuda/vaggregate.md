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

The arguments are those of `nss.VAggregate`. `device_id` and `num_streams`
are accepted but no device is used.

## On the device

- **`nss_cuda.VAggregate` runs on the host.** Its inputs are already host
  frames, and uploading 2(2R+1) planes per frame costs several times more than
  the sum itself. It sums slices in the CPU's order and divides with IEEE
  rounding.
- **Interop.** `nss.VAggregate` and `nss_cuda.VAggregate` accept each other's
  BM3D output. They agree within a few ulp: the CPU's fast-math division may
  be up to 2 ulp off IEEE rounding.
