# Isolated NCSR alignment lab

This directory contains experimental source and a patch for the frozen source
archive documented in `../ncsr_alignment_plan.md`. It is not compiled by the
production build, does not add a public filter parameter, and is not a complete
TIP 2013 reproduction. Author MATLAB code was not copied into these files.

To reproduce, extract the baseline source archive to a **new external source
directory**, then run there:

```sh
git apply --check /absolute/path/to/tests/ncsr_alignment_lab/experiment.patch
git apply /absolute/path/to/tests/ncsr_alignment_lab/experiment.patch
```

Configure/build/test this external source with the same compiler, Highway source
and VapourSynth runtime as its baseline. The patch adds one compilation unit
through the existing NCSR source glob, adds a CTest self-check, and connects the
private frame and shrinkage hooks. Never apply it over unreviewed overlapping
production edits.

The private environment variable `NSS_NCSR_ALIGNMENT_FLAGS` selects bits from
`ncsr_alignment_lab.hpp`; zero/unset leaves legacy behavior. Supported experiments
are spatial (`radius=0`), without `rclip`; the harness selects the same explicit
parameters for paired variants. This code is a bounded research implementation,
not a hardened public environment-variable API.

Run `../ncsr_alignment.py` with both baseline and candidate plugin paths, identical
saved fixtures, CPU0 affinity and an idle SMT sibling. Every worker preloads
source frames, warms once, measures complete subsequent frames, and verifies
exact output repeatability. Training is not cached across frames or excluded
from timing. The trace contains per-inner-iteration estimated sigma, threshold
summary and stage timing.

Important adaptations: single-scale capped training samples, patch-DC features,
six deterministic clustering iterations, standard covariance eigenvectors,
sparse-class DCT fallback, and clamping negative residual-noise energy rather
than taking its absolute value. Performance reflects this implementation and
compiler, not a lower bound on faithful NCSR or optimized native implementations.

Source mirrors here are for inspection. `experiment.patch` is the integration
artifact, and downloaded source/plugin SHA256 records identify measured builds.
