# Vulkan preview release checks

The r3 preview contains NLM, BM3D and PNLM. Build-only CI and a CTest
summary containing skips do not establish device-runtime admission.

## Create/render/free stability

`test_vk_stress.py` exercises Gray, YUV420 and RGB on padded rows, concurrent
frame requests in scrambled order, temporal boundaries, and repeated node
creation and destruction. BM3D includes spatial, rolling and legacy output.
Each recreated node must produce the same finite pixels; GPU-frame output
must remain resident and download to the CPU-frame result's exact bits.

Run the offline coverage guards without VapourSynth or NumPy:

```sh
python3 tests/vk/test_vk_stress_harness.py
```

Configure the Vulkan build with `NSS_VK_GPU_FRAMES_PYTHON` set to an R80/R81
Python executable to register `test_vk_stress_r8x` as well as the default
Python's `test_vk_stress`. Both use 25 cycles and a 600-second CTest timeout.
The R8x test requires GPU frames and fails if they are unavailable. The
default test runs CPU frames even on R78; without a supported plugin device
it returns 77. A supported device whose Backend probe fails is a failure.

For a longer admission run, use the same built plugin on both cores (paths
below are examples relative to the repository). Run with plugin autoloading
disabled so an older installed plugin cannot substitute for the candidate:

```sh
python tests/vk/test_vk_stress.py \
  --plugin build-vk/src/vulkan/libnss_vk.so --iterations 500
/home/owen/src/vs-envs/r81/python tests/vk/test_vk_stress.py \
  --plugin build-vk/src/vulkan/libnss_vk.so --iterations 500 --require-gpu
```

Require exit **0** from each command; 77 is not release admission. Repeat the
GPU-frame command with R80. Capture stdout, stderr and the exit status; run
under an external timeout for unattended soaks. Use a separate run with
synchronization validation enabled and verify that the validation layer
actually loaded on both the plugin device and the core's device.

This soak checks lifetimes and deterministic output. It does **not** measure
a device-memory leak bound or exercise driver failure injection. Record
memory telemetry separately before claiming memory stability.

## Other release evidence

- Build Linux and Windows archives and check that each includes the plugin,
  README, LICENSE, NOTICE and the complete `docs/api/` tree. Vulkan pages
  refer to their CPU and CUDA counterparts through relative links.
- On Windows with vapoursynth-classic R57.A14+, run the NLM, BM3D and PNLM
  transfer paths, verify direct transfers really activate, and compare with
  fallback staging. Include `nss_cuda`'s direct path. A Windows compile or
  a Linux classic run does not satisfy this check. The owner deferred this
  gate on 2026-10-29; it remains unvalidated and full V12 coverage is not claimed.
- Measure 1080p NLM, BM3D and PNLM against `nss_cuda` and the 32-thread CPU:
  Gray/YUV420/RGB, ordinary CPU frames, classic direct transfers and Vulkan
  resident GPU frames. Keep upload/filter/download timings distinct from
  resident chains. Use preloaded identical inputs, warmup and alternating
  paired timings; record parameters, deterministic hashes and outliers.
- Freeze source identity (including dirty changes), plugin hashes, core and
  driver versions, device, toolchain and input identity alongside results.
  Existing development measurements are not measurements of the release
  candidate. Keep AMD/Intel marked untested until device evidence exists.

CUDA matcher items #4, #8 and #17 retain their output-quality and paired-timing
requirements. Their best-effort scheduling does not waive those checks.
