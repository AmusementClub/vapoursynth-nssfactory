# ARM ISA feasibility lab

These tools are opt-in experiments. They are not built by the normal CMake
targets and do not enable SVE or SME in the production plugin. C++20 is retained.

Use a verified native C4A Spot environment and fresh directories. The full
Highway target remains NEON for the compiler experiments below; generated
instructions can still use SVE when compiler flags permit it. `Backend()` is a
Highway dispatch diagnostic, not an instruction census.

| Build | `CXXFLAGS` |
|---|---|
| Current reference | empty |
| Neoverse V2, NEON restriction | `-mcpu=neoverse-v2+nosve` |
| Neoverse V2, SVE allowed | `-mcpu=neoverse-v2` |
| Neoverse V2, SVE fixed length | `-mcpu=neoverse-v2 -msve-vector-bits=128` |

Run each through `tests/arm_native_gate.py` with new build/output paths. Archive
`CMakeCache.txt`, `compile_commands.json`, source hashes and binaries: the CXXFLAGS
environment is material to these builds. The resulting binaries target this
verified C4A environment; they are not general ARM release artifacts.

- `capabilities.cpp`: Linux HWCAP/prctl checks and actual SVE float lanes. It
  does not execute an unsupported ISA. Requested VLs are tested and restored.
- `sve_types.cpp`: raw sizeless arrays intentionally fail; define
  `NSS_SVE_FIXED_TYPES` with `-msve-vector-bits=128` for the sized-type control.
- `compile_audit.py`: syntax-check current CPU compile commands as Highway
  SVE2_128 without editing source. Compilation is not functional admission.
- `sme_compile.S`: assembler-only SME instruction canary. Do not call it as a
  production ABI wrapper. `sme_gemm.cpp` is the separate ACLE implementation.
- `instruction_loops.S` + `counter_probe.cpp`: opcode-count controls for the
  PMU, not an application throughput benchmark.
- `../c4a_profile.py`: fixed CPU0, preloaded integration inputs, frame-gated
  counters and sampling, checked output hashes and non-multiplexed event groups.
  First calibrate a reference, then freeze the same configuration/frame counts
  and use `--fixed-frames` for comparative PMU collection.
- `paired.py`: seven or fifteen alternating full-filter pairs using identical
  frame counts and source hashes; preserves all pixel deviations. It never
  grants a release or optimization gate.
- `gemm_trace.cpp`, `trace_worker.py`, `collect_extras.py`: explicitly preloaded
  Linux shape tracing within frame requests. Traced pixels must equal the
  uninstrumented reference. Instrumented elapsed times are not timing evidence.
  NN dimensions mean output m×n with inner k; TN means output k×n with inner m.
- `disassemble.py`: static SVE instruction inventory and sampled hot-symbol
  annotations. Static counts do not establish execution or speedup.

The Apple SME test compiles only `sme_gemm.cpp` with `-march=armv8.6-a+sme`.
Compile `sme_gemm_bench.cpp` normally and link it with that object and the
native-tested NEON static libraries. The driver checks macOS SME capability
before executing the kernel. The prototype uses FP32 SME outer products on
SME2-capable hardware; it does not claim SME2-specific tuning. Timing includes
B packing and streaming-mode transitions, with caller-owned allocation outside
both paths. It reuses warm buffers, does not pin a macOS core or measure cache
residency, and is not a full LSSC performance result.

Measured results are condensed in the top-level README's release and evidence
summary; the dated evaluation report is maintained outside the source tree.
