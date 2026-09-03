# vapoursynth-nssfactory

A work-in-progress VapourSynth factory for classical NSS (non-local self-similarity) denoisers — NLM, BM3D, WNNM, MCWNNM, TWSC, NLH, NCSR, LSSC — honoring the pre-AI era of these algorithms.

CPU plugin `libnss.so`, namespace `nss`. Linux x86-64, AVX2 minimum. Constant Gray / YUV / RGB 32-bit float clips only.

This project is WIP. ARM NEON, CUDA, and Vulkan support may be added later.

## Usage

```python
core.nss.NLM(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2, string channels = "AUTO", int wmode = 0, float wref = 1.0, clip rclip = None])

core.nss.BM3D(clip clip[, clip ref, float[] sigma = 3.0, int[] block_size = 8, int[] group_size = 8, int[] block_step, int[] bm_range = 7, int radius = 0, int[] ps_num, int[] ps_range = 4])
core.nss.BM3Dv2(...)  # BM3D then VAggregate when radius > 0

core.nss.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 0, int adaptive_aggregation = 1, clip rclip = None])
core.nss.WNNMv2(...)

core.nss.MCWNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, int residual = 1, int adaptive_aggregation = 0, clip rclip = None, int admm_iter = 10, float rho = 3.0, float mu = 1.001, int iters = 2, float delta = 0.1])
core.nss.MCWNNMv2(...)

core.nss.TWSC(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, float lambda1 = 0.0, float lambda2 = 3.0, clip rclip = None, int iters = 2, float delta = 0.1])

core.nss.NLH(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 16, int bm_range = 20, int radius = 0, int ps_num = 2, int ps_range = 4, int q = 4, clip rclip = None])
core.nss.NLHv2(...)

core.nss.NCSR(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, clip rclip = None, int iters = 2, float delta = 0.1])

core.nss.LSSC(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int radius = 0])

core.nss.VAggregate(clip clip, clip src[, int radius = 0, int[] planes])

core.nss.Version()  # returns version:data
```

`v2` filters are create-time sugar: the matching filter, then `VAggregate` when `radius > 0`.

NLM currently supports `wmode=0` (Welsch) only. TWSC rejects `lambda1 != 0`.

## Compilation

CMake ≥ 3.24, C++20, VapourSynth API4 headers. [Highway](https://github.com/google/highway) 1.4.0 is fetched at configure time.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Install `libnss.so` into the VapourSynth plugin directory.

## License

GPLv2. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The length-16/32/64 DCT kernels in `src/cpu/bm/dct_codelet_*.hpp` are generated
by [FFTW](https://www.fftw.org/) genfft (`gen_r2r`) and are distributed under
GPLv2 or later. Copyright (c) 1997-1999, 2003, 2007-14 Massachusetts Institute
of Technology and Matteo Frigo. They are mapped onto Highway and are not linked
against `libfftw3`. Regenerate with `tools/gen_dct_codelets.sh`.
