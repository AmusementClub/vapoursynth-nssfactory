# NCSR alignment experiment contract — 2026-09-08

This is an isolated exploratory campaign, not a production default change or
an implementation of every detail of the TIP 2013 author program.

## Frozen baseline

Current HEAD c3d08d29f4c5637451b15aeaf922cb028a6c48e4 plus the existing dirty
working tree, captured before this experiment. Source SHA256
`d67e630f7aa1729957139d2aeb4315bc8778035f25b5aef50062306c1f499383`.
Do not overwrite existing production edits. All C++ changes are applied to an
external extracted source copy and retained as a reproducible patch/archive.

## Feature axes

- A (bit 1): use a residual-noise estimate after feedback; unweighted second
  moments around weighted beta, subtract estimated noise variance, and use
  noise-dependent threshold coefficient 0.56/0.59/0.64. Clamp a negative
  residual-noise energy to zero (explicit engineering adaptation; the author
  program uses absolute value). Matching bandwidth is distinct from shrink sigma.
- W (bit 2): mean-distance-scaled bandwidth, noise-dependent floor, replace
  minimum-distance self weight by second-neighbor weight, then normalize.
- B (bit 4): construct a neighborhood for each selected target, guarantee target
  inclusion, use one target-coordinate PCA basis, and reconstruct/aggregate only
  that target. Coverage is a separate variable; compare under identical step.
- C (bit 8, requires B): replace per-group PCA by per-class full orthogonal PCA
  bases learned from the current image. Deterministic at most 8192 single-scale
  training patches, DC-removed features, up to 70 clusters, six Lloyd iterations,
  standard symmetric covariance eigensolver, flat class and sparse-class DCT
  fallback. This retains the clustered dictionary idea but is NOT the author's
  multiscale Gaussian-highpass training recipe or elementwise covariance repair.
- Reuse (bit 16, requires B): refresh neighborhoods/dictionaries every three inner
  iterations. Original noisy input stays fixed. Learned dictionaries, matching,
  and every filtering pass remain inside the timed frame boundary.
- D: change budgets separately: step 8/4/2/1, range 7/15/30, iters 2/3/6/9,
  group 8/16, then a bounded combined author-like budget at sigma 25. This is not
  a strict author-parameter recipe for all noise levels.

No author MATLAB source is copied into the plugin. Reference outputs from the
previous campaign may anchor quality only; old host runtimes are not mixed into
new same-host timing ratios.

## Experiments and checks

1. Build original and instrumented plugins on one freshly verified C4 Spot.
2. Validate eigensolver, weights, noise estimation, zero-sigma reconstruction,
   full standard CTest, and byte-equal hook-disabled outputs.
3. Pilot every feature on house128 sigma25 before starting the matrix.
4. Main matrix: identical saved three-image by four-noise inputs from the author
   comparison; default and hook control, A, W, AW, B, AWB, AWB+reuse, C+reuse.
   Three reversed/forward-order repeat rounds, with one warm frame then three
   fully computed timed frames per worker. Source frames are all preloaded.
5. D: three sigma25 images; step, window, iterations and group tested separately
   plus combined budget. Apply bounded per-worker timeout, retain failures.
6. Verify every raw-output SHA and independently recompute PSNR/SSIM. Repeated
   output hashes must match. Flag CPU1 activity above 1% and any steal; only use
   eligible same-host pairs for time summaries. Do not hide individual quality
   regressions behind averages.
7. Confirm representative results at complete house256 sigma25. If feasible,
   add full-HD native fixtures for low-budget paths; do not extrapolate tiny
   images to video performance or represent unrun combinations as passes.
   Add a same-step=1 B/W/WB/AWB comparison against legacy step=1 across the full
   twelve-case matrix: sparse target-only output has fewer overlapping estimates,
   so it cannot alone establish whether target-specific beta is useful.
8. Download/hash-verify evidence, preserve source patches and rejected variants,
   delete only the exact owned temporary Spot and its automatic boot disk.

## Interpretation

Implementation fidelity, numerical stability, quality, runtime, and environment
validity are distinct conclusions. No minimum speedup is required for admission
to this experiment, but no experiment is promoted to defaults by this task.

A/W experimental finishers use explicit scalar loops with double statistical
accumulation, while still using the existing batched group-PCA path. Consequently
their measured speed difference includes the changed implementation of the
finisher; it is not a hardware-independent cost of just replacing a coefficient.
Hook-disabled byte parity checks legacy behavior, not equivalence of A/W outputs.
