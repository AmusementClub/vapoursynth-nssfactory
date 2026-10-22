# nss_cuda.BM4D

BM4D on the device: the CPU model's transforms and fixed-point sums, with
the matching by FP16 tensor-core distances (compute capability 8.0 and
later), so near ties may take other cubes than the CPU's: the output is about
70 dB from `nss.BM4D`'s with the same PSNR to the clean source. Every run
gives the same bits.

Algorithm, paper, parameters and quality: [`nss.BM4D`](../bm4d.md).

```python
out = core.nss_cuda.BM4D(clip, sigma=5)
```

## Speed

1080p, RTX 5080, one instance; 160 frames of an anime episode with grain of
sigma 5 added:

| Clip | fps |
|---|---:|
| gray | 40 |
| YUV420 | 27 |

About 40 times the CPU plugin on a 16-thread Ryzen 9 5950X (gray: 1.0 fps).
Asking for one frame computes its 32-frame chunk; the chunk's other frames
come from the cache. Once the requests reach the last three quarters of a
chunk, the next chunk's frames are fetched and uploaded while this one
computes.

## Signature

```python
core.nss_cuda.BM4D(clip clip[, float[] sigma = 3.0, int memory_limit_mb,
                   int device_id = 0, int num_streams])
```

`num_streams` is accepted for the common signature; chunks compute one at a
time (the next chunk's upload overlaps).

## Notes

- **Close to the CPU, not bit for bit** (`tests/cuda/test_cuda_bm4d_live.py`):
  the cube distances are |r'|^2 + |c'|^2 - 2 r'.c' over FP16 values centred
  on the volume's mean (matrix products on the tensor cores), not the CPU's
  sums of squared differences; the transforms keep the CPU's arithmetic and
  the sums are 64-bit fixed point, exact in any order. On 64 frames of anime
  with grain of sigma 5: PSNR to the clean source 46.6635 (CPU) and 46.6631
  dB, a mean difference of 0.05 level. GPUs without sm_80 code, and volumes
  FP16 cannot hold, take the CPU's exact matching.
- **Memory**: the chunk's 48 input and 32 output frames of the largest plane
  on the device, and 48 more input frames when the clip is longer than one
  chunk (the next chunk's upload), plus about 1 GB of work for a
  1080p tile volume (about 50 MB of it for the tensor-core search).
