# Isolated TWSC SVD-library experiment

This lab compares the frozen production checkout with custom FP64 Jacobi at
`1e-13`, system OpenBLAS `DGESDD`, and (group replay only) `DGESVD`. It does not
modify the production build or admit a new backend. `prepare.py` patches only
an explicitly named `/tmp/nss-svd-*/source-lab` copy.

Library calls use economy factors, LP64 Fortran interfaces, one BLAS thread,
the existing scaling/rank-floor policy, and unchanged production validation.
Library modes replace the large batched route and scalar calls with n >= 24.
The 8-column shared WNNM/NCSR path is outside this experiment. The tighter
custom mode changes FP64 convergence in both scalar and lane implementations.
All candidates are numerical experiments, not byte-identical substitutions.

The nine full-frame cases use saved real DIV2K crops and deterministic noise.
Seven alternating triplets share a baseline within each round. Source frames
are materialized before timing; every timed pixel payload is retained. Both
whole-process and timed windows require CPU1 idle >= 99% and zero steal on
CPU0/CPU1. Failed windows remain in raw results and prevent formal cell
qualification; they are not silently dropped. Confidence intervals are paired
bootstrap intervals for the median. A lower bound above 1.02 is timing
evidence only, not numerical, cross-platform, or production admission.

`direct_confirmation.py` additionally compares DGESDD with the untouched
production plugin in seven alternating pairs for RGB G48, Gray G48, RGB
default sigma25 and RGB default sigma75. These cells were selected before the
primary paired results. This is a separate comparison and must not be pooled
with the triplet experiment or used to erase a failed earlier window.

Stock-versus-instrumented checks establish pixel equality on all cases, with
seven paired timing controls on Gray/RGB G48. Captures are diagnostic runs,
separate from formal timing. Their filenames contain shape and zero-based
per-shape call ordinal: the first 16 groups and every 256th group thereafter.
Capture runs include warmup and the timed frame, not just first-iteration data.

Group replay selects 16 groups spread through each saved capture sequence.
It measures decomposition, unchanged validation, coding, and their combined
cost. Coding uses a declared uniform nominal noise level; it is a controlled
factor-consumption diagnostic, not a replay of each production group's actual
noise weights. It uses ten ADMM iterations and zero restored means. Full-frame
cases test the real iteration budgets and weights, including unequal RGB.
Known-spectrum controls cover repeated spectra, geometric decay, rank
deficiency, zero matrices, and the maximum 768x256 shape. Independent Python
checks reconstruct input matrices, check orthogonality and compare spectra.
Kernel pilots choose one common repetition count for all four methods in each
cell, targeting at least 1.2 seconds for the fastest combined measurement.
Pilots are retained separately and excluded from paired statistics.

ResourceVector accounts adapter allocations; internal OpenBLAS allocations
are not integrated into the plugin's resource budget. This is one reason the
lab is not a production integration. The required numerical and host tests,
library/version hashes, complete raw payloads, failed attempts, and exact VM
cleanup records belong in the final evidence archive.

## Execution and evidence

The checked-in scripts are research tooling. Reproduction requires a frozen
source archive, GCC/CMake/Highway, VapourSynth R75, NumPy/SciPy for gates, and
a public LP64 OpenBLAS shared library. Run only on an isolated build tree;
`prepare.py` deliberately rejects the working checkout. Use the CMake flags
and package versions retained with the dated experiment, rather than treating
the host's current defaults as the same environment.

The sequence is fixture freezing, stock and lab builds, numerical/public API
gates, `campaign.py run`, `kernel_campaign.py`, `direct_confirmation.py`, and
`library_path_gate.py run`. Timing stages run sequentially. Kernel replay
starts a separately recorded quiet environment epoch; it does not repair any
failed primary timing window. Direct confirmation saves all its timed pixels
(two frames for RGB G48, seven for Gray G48, one for each default cell).

After export, `analyze.py` computes paired intervals and quality metrics;
`audit.py` independently checks retained payload hashes, repeat determinism,
zero source fills during timing, and every first-round matrix factorization.
The latter uses the stored FP32 matrices as its numerical reference. Preserve
raw failed windows and incomplete stages along with successful records.
