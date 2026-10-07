# nss_cuda.NLM

Non-local means on the device: all channel modes, temporal `d`, and `rclip`. The output is always normal-height.

The parameters mean what they mean in [`nss.NLM`](../nlm.md), which
also has the algorithm, the defaults rationale and the pitfalls. This page has
the call and what is specific to the device.

```python
out = core.nss_cuda.NLM(clip, h=1.2)          # defaults
denoised = core.nss_cuda.NLM(clip, d=1, a=4, s=3, h=1.5, channels="Y")
```

## Signature

```python
core.nss_cuda.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2,
                  string channels = "AUTO", int wmode = 0, float wref = 1.0,
                  clip rclip = None,
                  int memory_limit_mb, int device_id = 0, int num_streams = 1])
```

`device_id`, `num_streams` and `memory_limit_mb` are described in the
[shared arguments](README.md#arguments).

## On the device

- **Device-resident.** The whole frame runs on the device: clamped distance
  maps, the (2s+1)² box sums, Welsch weights, the symmetric accumulation and
  the `wref` normalization.
- **One launch per frame.** A frame is one kernel launch: a block of threads
  owns a 32 x 32 tile of the output and keeps its sums in registers through
  every offset of every frame pair, so the frames are read from device memory
  once and only the result is written.
- **Window frames stay on the device.** A frame of the clip is uploaded once,
  however many windows it is part of, in any order of requests and with any
  number of streams. Threads stage their uploads and copy their results out
  while the stream works on other frames.
- **Weight maps are computed once with `d = 1`.** The map of a frame against
  the next one is also the next frame's map against it, so a stream keeps
  what a frame computed for the frame after it. This helps requests in
  order only and costs `2 (2a + 1)²` planes per stream (396 MiB at 1080p
  with the default `a = 2`); it is the first thing `memory_limit_mb` takes
  away, and it is left out when it would need more than a quarter of the
  device's memory. The output is the same with and without it.
- **Memory.** By default the device holds `2d + 4` frames of the clip (with
  one stream). Under `memory_limit_mb` it holds fewer, down to one window. For
  `d` above 8, a patch or search window too large for a tile's shared memory
  (48 KiB: the defaults use 16), or a limit below one window, the filter falls
  back to one launch per offset with only the centre frame and the current
  backward/forward pair resident; the smallest accepted limit is never above
  what that takes.
- **Numerics.** The GPU uses the same model as the CPU, exact box sums and
  the CPU's fast exponential operation for operation.
  - The output is not bit-identical to the CPU (the CPU takes its sums over
    rows as running sums), but is far inside the 60 dB gate (above 144 dB
    measured).
  - The output is run-to-run identical: every pixel is accumulated by one
    thread in a fixed offset order.
