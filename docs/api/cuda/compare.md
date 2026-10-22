# Against other plugins

`nss` and `nss_cuda` against the established VapourSynth plugins for the same algorithms, on the same input with the
same arguments:

- [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA): `bm3dcuda`, `bm3dcuda_rtc`, `bm3dcpu`.
- [VapourSynth-WNNM](https://github.com/WolframRhodium/VapourSynth-WNNM): `wnnm`.
- [vs-nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc): `nlm_ispc`.
- [vs-nlm-cuda](https://github.com/AmusementClub/vs-nlm-cuda): `nlm_cuda`.

The last section compares r2 with r1. Measured 2026-10-21 with the release builds of r2 (CUDA 12.9, the toolkit of
the published packages).

## Summary

- **Same algorithm, same result.** At the same arguments the outputs match the other plugins: the denoised PSNR agrees
  to 0.001 dB for spatial BM3D, CBM3D, WNNM and NLM.
- **Temporal filters denoise better.** Temporal BM3D (`radius=1`) is +0.93 dB above all four BM3DCUDA builds, and
  temporal WNNM +0.33 dB above VapourSynth-WNNM. The temporal search differs; see the notes.
- **Faster on the GPU** (1080p, against the fastest other GPU plugin):
  - BM3D 1.35×, BM3D two stages 2.78×, CBM3D 1.22×, temporal BM3D 2.48×.
  - NLM 1.56×, NLM `d=1` 3.0×.
  - WNNM, which has no other GPU implementation: 19× the CPU plugin VapourSynth-WNNM.
- **On the CPU:**
  - WNNM is 3.0× VapourSynth-WNNM, and NLM is 1.3–1.45× vs-nlm-ispc per thread.
  - BM3D runs at 0.87–0.94× of bm3dcpu, and at 0.53× for temporal BM3D, which does more search for the better result
    above.

## Setup

| Item | Value |
|---|---|
| Host | RTX 5080 (driver 595.91.07, also drives the desktop), Ryzen 9 5950X (16C/32T), Ubuntu 26.04, VapourSynth R78 |
| `nss`, `nss_cuda` | r2 release build of `master` `65d833d` (package workflow, CUDA 12.9 / GCC 13): `libnss.so` `fee9ecde…`, `libnss_cuda.so` `792ea277…` |
| r1 | the published `nssfactory-r1-linux-x86_64-cuda` package, `libnss_cuda.so` `3de42783…` |
| BM3DCUDA | b28d362, `sm_120`, upstream release flags (`bm3dcuda` `74a23460…`, `bm3dcuda_rtc` `5afc1a9c…`); `bm3dcpu` from vsrepo (`2b237a43…`) |
| VapourSynth-WNNM | vsrepo `v3-0-g52c764c` with MKL (`0080dc59…`), `MKL_NUM_THREADS=1` |
| vs-nlm-ispc | vsrepo `v3-0-gce27c1a` (`d67cc0ee…`) |
| vs-nlm-cuda | `7633714`, built with its upstream flags for `120-real` (`5bb9de22…`) |
| Input | 1920×1080 frames of `ds/MAPPA.png` (gray, BT.601 luma) and `ds/Ufotable.png` (RGB or full-range BT.709 YUV 4:4:4) with seeded Gaussian noise of the given sigma (8-bit units); temporal cases use a 5-frame pan. Every plugin gets the same frames |
| Arguments | Identical per case, spelled in each plugin's vocabulary. BM3D pins `block_step=8, bm_range=7` on every side (BM3DCUDA's block and group are fixed at 8). "Two stages" is `final=1` against `BM3D(ref=BM3D(...))` |
| Speed | 1080p frames per second, each implementation × case in its own process, two rounds in alternating order, three timed passes of at least 3 s per round after a warm-up; the median of the six passes. GPU plugins at 8 VapourSynth threads and their default streams; CPU plugins at 32 threads |

## Output

Denoised PSNR against the clean image (dB). In parentheses: the agreement with `nss` (CPU) on the same input (PSNR of
the difference, dB; 60 or more is the backends' own gate).

| Case | Noisy | `nss` | `nss_cuda` | `bm3dcuda` | `bm3dcuda_rtc` | `bm3dcpu` |
|---|---:|---:|---:|---:|---:|---:|
| BM3D, gray, σ5 | 34.15 | 40.471 | 40.471 (108) | 40.471 (97) | 40.471 (97) | 40.470 (84) |
| BM3D, gray, σ25 | 20.18 | 29.142 | 29.141 (77) | 29.142 (144) | 29.142 (144) | 29.141 (85) |
| BM3D, gray, σ25, step 4 | 20.18 | 30.605 | 30.605 (93) | 30.605 (105) | 30.605 (105) | 30.605 (84) |
| BM3D two stages, gray, σ5 | 34.15 | 40.371 | 40.371 (94) | 40.371 (91) | 40.371 (90) | 40.370 (67) |
| BM3D two stages, gray, σ25 | 20.18 | 29.426 | 29.426 (74) | 29.426 (84) | 29.426 (84) | 29.427 (59) |
| CBM3D, YUV444, σ10 (`chroma=1`) | 28.13 | 36.363 | 36.363 (102) | 36.363 (106) | 36.363 (106) | 36.363 (91) |
| BM3D, YUV444, σ10, per plane | 28.13 | 36.825 | 36.825 (94) | 36.825 (95) | 36.825 (95) | 36.825 (89) |
| **BM3D, gray, σ10, `radius=1`** | 28.13 | **36.478** | **36.478** (103) | 35.549 (42) | 35.549 (42) | 35.549 (42) |

| Case | Noisy | `nss` | `nss_cuda` | `wnnm` |
|---|---:|---:|---:|---:|
| WNNM, gray, σ5 | 34.15 | 35.417 | 35.417 (92) | 35.417 (81) |
| WNNM, gray, σ25 | 20.18 | 21.857 | 21.857 (113) | 21.857 (83) |
| WNNM, gray, σ25, step 4, group 16 | 20.18 | 23.686 | 23.686 (94) | 23.686 (84) |
| **WNNM, gray, σ10, `radius=1`** | 28.13 | **29.735** | **29.735** (91) | 29.403 (49) |

| Case | Noisy | `nss` | `nss_cuda` | `nlm_ispc` | `nlm_cuda` |
|---|---:|---:|---:|---:|---:|
| NLM, gray, σ5, `d=0` | 34.15 | 41.640 | 41.640 (143) | 41.640 (141) | 41.640 (142) |
| NLM, gray, σ25, `d=0, a=3` | 20.18 | 32.807 | 32.807 (141) | 32.807 (139) | 32.807 (139) |
| NLM, RGB, σ5, `d=0` | 34.15 | 41.983 | 41.983 (144) | 41.983 (138) | 41.991 (65) |
| NLM, gray, σ10, `d=1` | 28.13 | 36.022 | 36.022 (141) | 36.022 (139) | 36.022 (139) |

## Speed

GPU plugins, frames per second at 1080p (ratio: `nss_cuda` against the fastest other GPU plugin):

| Case | `nss_cuda` | `bm3dcuda` | `bm3dcuda_rtc` | `nlm_cuda` | Ratio |
|---|---:|---:|---:|---:|---:|
| BM3D, gray, σ5 | **717** | 530 | 530 | | 1.35 |
| BM3D two stages, gray, σ5 | **659** | 237 | 238 | | 2.78 |
| CBM3D, YUV444, σ10 | **201** | 165 | 164 | | 1.22 |
| BM3D, gray, σ10, `radius=1` (rolling) | **579** | 234 | 232 | | 2.48 |
| NLM, gray, σ5, `d=0` | **670** | | | 428 | 1.56 |
| NLM, RGB, σ5, `d=0` | **201** | | | 123 | 1.63 |
| NLM, gray, σ10, `d=1` | **414** | | | 138 | 3.00 |
| WNNM, gray, σ5 | **693** | | | | (no other GPU WNNM) |
| WNNM, gray, σ10, `radius=1` | **605** | | | | |

CPU plugins, 32 threads (ratio: `nss` against the other CPU plugin):

| Case | `nss` | `bm3dcpu` | `wnnm` | `nlm_ispc` | Ratio |
|---|---:|---:|---:|---:|---:|
| BM3D, gray, σ5 | 243 | 259 | | | 0.94 |
| BM3D two stages, gray, σ5 | 116 | 125 | | | 0.93 |
| CBM3D, YUV444, σ10 | 91 | 105 | | | 0.87 |
| BM3D, gray, σ10, `radius=1` | 54 | 101 | | | 0.53 |
| WNNM, gray, σ5 | **109** | | 36 | | 3.0 |
| WNNM, gray, σ10, `radius=1` | **42** | | 29 | | 1.47 |
| NLM, gray, σ5, `d=0` | **332** | | | 28 | (see notes) |
| NLM, RGB, σ5, `d=0` | **179** | | | 20 | (see notes) |
| NLM, gray, σ10, `d=1` | **74** | | | 6.0 | (see notes) |

## Notes

- **Temporal search.**
  - `nss` follows V-BM3D / VapourSynth-WNNM's predictive search: it carries the best matches of each frame to seed the
    next.
  - BM3DCUDA's `BM3Dv2` searches the neighbouring frames its own way.
  - On this pan our search finds better groups: +0.93 dB for BM3D, +0.33 dB for WNNM.
  - On the CPU the extra search costs time (0.53× of `bm3dcpu`); on the GPU `nss_cuda` is still 2.5× faster than
    `bm3dcuda`.
  - `nss` on the CPU at `radius=1` is `temporal_mode="legacy"` + `nss.VAggregate`. The CPU's `"rolling"` gives the same
    output but does not scale across frame requests: 14 fps.
- **WNNM speed.** VapourSynth-WNNM is a CPU plugin (with MKL); the 19× is GPU against CPU. CPU against CPU, `nss` is
  3.0× faster.
- **vs-nlm-ispc** does not get faster with more VapourSynth threads on this host. Single-threaded, `nss` was 1.29–1.45×
  faster in the 2026-10-17 run; the NLM code has not changed since.
- **vs-nlm-cuda RGB.**
  - With float input, its RGB distance loses the colour-dependent weight of KNLMeansCL's red-mean formula:
    `inv_divisor` is 0 for float, so `w = 0`.
  - Its RGB output therefore differs from the others (65 dB), while `nss`, `nss_cuda` and `nlm_ispc` agree at
    138–144 dB.
  - Integer input is not affected.
- **`bm3dcpu`** differs from the BM3DCUDA GPU builds by itself (84 dB basic, 59–67 dB two stages); the Wiener stage
  amplifies it.
- **Toolkit.**
  - The release packages are built with CUDA 12.9 for driver compatibility.
  - A build with CUDA 13.3 gives the same output and the same speed within 4% for every filter except NLM: about 1.35×
    faster with neighbouring frames (`d=1`, the default) and 1.08× on RGB.
  - The NLM rows above are the release build's.
- **Limits.**
  - One GPU and one CPU.
  - Speed figures with the GPU also driving the desktop.
  - The third-party plugins are their published builds, so their compile flags are not ours to choose.

## r2 against r1

`nss_cuda` at the default arguments, 1080p, sigma 3 on `ds/MAPPA.png`; both release packages (CUDA 12.9), frames per
second, two alternating pairs, each timed for at least 7 s:

| Filter | YUV420 r1 → r2 | RGB r1 → r2 | Output |
|---|---|---|---|
| LSSC | 196 → **250** (1.28×) | 125 → **148** (1.18×) | model 4 (better text and detail) |
| TWSC | 66 → **76** (1.15×) | 36 → **44** (1.25×) | same |
| BM3D | 516 → 494 (0.96×) | 280 → 279 (1.00×) | default `bm_range` 7 → 9 (+0.06–0.14 dB) |
| NLM | 416 → 416 | 218 → 217 | same |
| WNNM | 497 → 496 | 281 → 281 | same |
| NCSR | 387 → 386 | 214 → 213 | same |
| NLH | 36 → 36 | 30 → 30 | model 7 (better blind estimate) |
| MCWNNM | | 196 → 195 | same |

At `radius=1` (same arguments as the tables above, gray): BM3D 512 → 579 fps (1.13×), WNNM 513 → 605 (1.18×).

## Reproducing

```sh
# quality, then speed (each case and plugin in its own process), on a host with every plugin
THIRD_PARTY_SO=<bm3dcuda.so>:<bm3dcuda_rtc.so>:<bm3dcpu.so>:<wnnm.so>:<nlm_ispc.so> \
NSS_SO=<libnss.so> NSS_CUDA_SO=<libnss_cuda.so> NSS_CUDA_R1_SO=<r1 libnss_cuda.so> NLM_CUDA_SO=<libvsnlm_cuda.so> \
  python tests/compare_third_party.py quality --report quality.json
python tests/compare_third_party.py bench-all --out bench.jsonl --rounds 2 --runs 3 --min-seconds 3 --r1
python tests/compare_third_party.py summarize --quality quality.json --bench bench.jsonl
```
