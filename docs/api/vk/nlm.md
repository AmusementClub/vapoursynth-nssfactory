# nss_vk.NLM (preview)

Non-local means on a Vulkan device: all channel modes, temporal `d`, `wmode`
and `rclip`, on CPU frames or, on VapourSynth R80+, on GPU frames. The output
is always normal-height.

Algorithm, paper and the reasoning behind the defaults: [`nss.NLM`](../nlm.md).
Every argument means what it does for [`nss_cuda.NLM`](../cuda/nlm.md).

```python
out = core.nss_vk.NLM(clip, h=1.2)                          # defaults
denoised = core.nss_vk.NLM(clip, d=1, a=4, s=3, h=1.5, channels="Y")
resident = core.nss_vk.NLM(clip.std.GPUUpload(), d=1)       # VapourSynth R80+: stays on the GPU
```

## Signature

```python
core.nss_vk.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2,
                string channels = "AUTO", int wmode = 0, float wref = 1.0,
                clip rclip = None,
                int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

On VapourSynth R80+ `clip` and `rclip` take CPU or GPU frames (`vnode:all`);
both must be the same kind ([GPU frames](README.md#gpu-frames-vapoursynth-r80)).

## Speed

1080p, default arguments, RTX 5080 (see the [overview](README.md#speed)):

| Frames | gray | YUV420 | RGB |
|---|---:|---:|---:|
| VapourSynth R81, CPU frames | 804 | 710 | 209 |
| vapoursynth-classic, direct transfers | 851 | 836 | 381 |
| R81, GPU frames | 841 | 839 | 505 |

At its defaults NLM is bound by its kernel (`d=1`: about 1.1 ms a 1080p plane
on the RTX 5080), so GPU frames gain most where the transfers were the limit
(RGB).

## Notes

- **Accuracy.** 142 dB or better against the CPU plugin on 1295 live frames
  (the same model, exact box sums); identical from run to run, and the same
  bits on GPU frames as on CPU frames.
- **Two paths.** The defaults and anything with `2d + 1 <= 17` frames and a
  tile that fits 48 KiB of shared memory run one dispatch per frame, with the
  window frames kept on the device once for every request that reads them.
  Larger windows, patches or `d` take a slower path with three frames
  resident.
- **`d=1`.** The weights of a frame pair are computed once for both frames
  when frames are rendered in order (198 MiB of maps per set at 1080p with
  `a=2`, one set more than there are streams); `memory_limit_mb` takes them
  away first, without changing the output.

## Pitfalls

As for [`nss_cuda.NLM`](../cuda/nlm.md#pitfalls): `h` is not in sigma units
(the default barely touches noise of sigma 3 or more), and the Bisquare
`wmode`s need a much larger `h`.
