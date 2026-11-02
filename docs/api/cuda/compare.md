# Against other plugins

`nss` and `nss_cuda` against the established VapourSynth plugins for the same algorithms, on the same input with the
same arguments:

- [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA): `bm3dcuda`, `bm3dcuda_rtc`, `bm3dcpu`.
- [VapourSynth-WNNM](https://github.com/WolframRhodium/VapourSynth-WNNM): `wnnm`.
- [vs-nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc): `nlm_ispc`.
- [vs-nlm-cuda](https://github.com/AmusementClub/vs-nlm-cuda): `nlm_cuda`.

## Summary

- **Same algorithm, same result** for WNNM and NLM: at the same arguments the denoised PSNR agrees to 0.001 dB.
- **Spatial BM3D and CBM3D denoise a little better** since BM3D model 3 (Haar along the group, as the paper;
  BM3DCUDA and bm3dcpu use a DCT there, as model 2 did): +0.05 to +0.13 dB at the same arguments. Model 2 matched
  them to 0.001 dB.
- **Temporal filters denoise better.** Temporal BM3D (`radius=1`) is +0.99 dB above all four BM3DCUDA builds, and
  temporal WNNM +0.33 dB above VapourSynth-WNNM. The temporal search differs; see the notes.
- **Speed** depends on the GPU, host and arguments; the [script](#reproducing) times every plugin on your own
  host. WNNM has no other GPU implementation.

## Setup

| Item | Value |
|---|---|
| Host | RTX 5080 (driver 595.91.07, also drives the desktop), Ryzen 9 5950X (16C/32T), Ubuntu 26.04, VapourSynth R78 |
| `nss`, `nss_cuda` | r2 release build of `master` `65d833d` (package workflow, CUDA 12.9 / GCC 13): `libnss.so` `fee9ecde…`, `libnss_cuda.so` `792ea277…` |
| BM3DCUDA | b28d362, `sm_120`, upstream release flags (`bm3dcuda` `74a23460…`, `bm3dcuda_rtc` `5afc1a9c…`); `bm3dcpu` from vsrepo (`2b237a43…`) |
| VapourSynth-WNNM | vsrepo `v3-0-g52c764c` with MKL (`0080dc59…`), `MKL_NUM_THREADS=1` |
| vs-nlm-ispc | vsrepo `v3-0-gce27c1a` (`d67cc0ee…`) |
| vs-nlm-cuda | `7633714`, built with its upstream flags for `120-real` (`5bb9de22…`) |
| Input | 1920×1080 frames of `ds/MAPPA.png` (gray, BT.601 luma) and `ds/Ufotable.png` (RGB or full-range BT.709 YUV 4:4:4) with seeded Gaussian noise of the given sigma (8-bit units); temporal cases use a 5-frame pan. Every plugin gets the same frames |
| Arguments | Identical per case, spelled in each plugin's vocabulary. BM3D pins `block_step=8, bm_range=7` on every side (BM3DCUDA's block and group are fixed at 8). "Two stages" is `final=1` against `BM3D(ref=BM3D(...))` |

## Output

Denoised PSNR against the clean image (dB). In parentheses: the agreement with `nss` (CPU) on the same input (PSNR of
the difference, dB; 60 or more is the backends' own gate).

| Case | Noisy | `nss` | `nss_cuda` | `bm3dcuda` | `bm3dcuda_rtc` | `bm3dcpu` |
|---|---:|---:|---:|---:|---:|---:|
| BM3D, gray, σ5 | 34.15 | 40.553 | 40.553 (104) | 40.471 (48) | 40.471 (48) | 40.470 (48) |
| BM3D, gray, σ25 | 20.18 | 29.219 | 29.219 (81) | 29.142 (35) | 29.142 (35) | 29.141 (35) |
| BM3D, gray, σ25, step 4 | 20.18 | 30.608 | 30.608 (90) | 30.605 (42) | 30.605 (42) | 30.605 (42) |
| BM3D two stages, gray, σ5 | 34.15 | 40.451 | 40.451 (92) | 40.371 (50) | 40.371 (50) | 40.370 (50) |
| BM3D two stages, gray, σ25 | 20.18 | 29.480 | 29.480 (79) | 29.426 (37) | 29.426 (37) | 29.427 (37) |
| CBM3D, YUV444, σ10 (`chroma=1`) | 28.13 | 36.493 | 36.493 (110) | 36.363 (42) | 36.363 (42) | 36.363 (42) |
| BM3D, YUV444, σ10, per plane | 28.13 | 36.914 | 36.914 (96) | 36.825 (43) | 36.825 (43) | 36.825 (43) |
| **BM3D, gray, σ10, `radius=1`** | 28.13 | **36.541** | **36.541** (108) | 35.549 (41) | 35.549 (41) | 35.549 (41) |

The BM3D rows use model 3 from a dev-host build (GCC 15, CUDA 13.3 `120-real`; the third
plugins as in the setup). Under model 2 the `nss` columns equalled BM3DCUDA's to 0.001 dB with an agreement of 59 dB or
more; the Haar transform along the group moves them by more than the gate, so the agreement column now reads 35–50 dB
for the third plugins while `nss_cuda` against `nss` stays at 79–110 dB.

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

## Notes

- **Temporal search.**
  - `nss` follows V-BM3D / VapourSynth-WNNM's predictive search: it carries the best matches of each frame to seed the
    next.
  - BM3DCUDA's `BM3Dv2` searches the neighbouring frames its own way.
  - On this pan our search finds better groups: +0.93 dB for BM3D, +0.33 dB for WNNM. It searches more, which
    costs time on the CPU.
  - `nss` on the CPU at `radius=1` is `temporal_mode="legacy"` + `nss.VAggregate`. The CPU's `"rolling"` gives the same
    output but does not scale across frame requests.
- **VapourSynth-WNNM** is a CPU plugin (with MKL).
- **vs-nlm-cuda RGB.**
  - With float input, its RGB distance loses the colour-dependent weight of KNLMeansCL's red-mean formula:
    `inv_divisor` is 0 for float, so `w = 0`.
  - Its RGB output therefore differs from the others (65 dB), while `nss`, `nss_cuda` and `nlm_ispc` agree at
    138–144 dB.
  - Integer input is not affected.
- **`bm3dcpu`** differs from the BM3DCUDA GPU builds by itself (84 dB basic, 59–67 dB two stages); the Wiener stage
  amplifies it.
- **Limits.**
  - One GPU and one CPU.
  - The third-party plugins are their published builds, so their compile flags are not ours to choose.

## Reproducing

```sh
# quality, then (optionally) speed on your own host, each case and plugin in its own process
THIRD_PARTY_SO=<bm3dcuda.so>:<bm3dcuda_rtc.so>:<bm3dcpu.so>:<wnnm.so>:<nlm_ispc.so> \
NSS_SO=<libnss.so> NSS_CUDA_SO=<libnss_cuda.so> NLM_CUDA_SO=<libvsnlm_cuda.so> \
  python tests/compare_third_party.py quality --report quality.json
python tests/compare_third_party.py bench-all --out bench.jsonl --rounds 2 --runs 3 --min-seconds 3
python tests/compare_third_party.py summarize --quality quality.json --bench bench.jsonl
```
