# LSSC paper-first reference track

Status: **P1 fixed-dictionary SC/pilot/SSC reference implemented. Full LSSC is NOT implemented.**

The opt-in [native exploration track](NATIVE_EXPLORER.md) now provides bounded
native pursuit/aggregation, mixed-precision controls and explicitly approximate
penalized learning. It does not change frozen P1/P2 or the production filter.

Recorded stage results (the dated reports are maintained outside the source
tree): P1 fixed-dictionary validation passed 44 pytest cases and beat the old
plugin default by ≈4.8 dB on the 128² matrix while remaining ≈0.6 dB below the
full author reference, with retained σ50 SSIM regressions; P2 confirmed the
author's SPAMS residual-energy rule against an independent build (26/26 probes)
without promoting any policy; native mixed-precision exploration reached ≈6–7×
over the FP64 path on M4 Max with bounded precision differences; the author's
full learning budget costs ≈26–29× over zero learning on House images. Actual
ICCV MEX reconciliation remains pending, and P2 alternatives live separately in
`p2_reference.py`; the P1 algorithm is frozen.

Size/noise pressure testing uses `scaling.py`: independent persistent worker
processes for P1/P2, serial alternating pairs, instrumented warm passes excluded
from timing, exact warm/repeat output hashes, and variant-specific OS peak RSS.
All denoising/trace allocations remain timed. Its process ceilings default to
900 seconds per call and 24 GiB RSS per worker; failures retain partial evidence.
For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-paper/bin/python tests/lssc_paper/scaling.py \
  --clean /external/barbara.png --dictionary /external/dict_n9.mat \
  --cases 128x128:25 256x256:25 512x512:25 --pairs 3 \
  --out artifacts/lssc-paper-scaling/example
```

The native image is cropped without rescaling. Requests exceeding its original
size tile only the clean image and add independent noise; these are explicitly
labeled size-stress fixtures, not natural full-resolution image-quality evidence.
Sigma sweeps with one dictionary hold patch geometry fixed, not author defaults.
This is still the fixed-dictionary stage: it cannot answer the additional cost of
image-adaptive/grouped learning or predict a production C++ implementation.

This track is developed in isolation from the production filter. Earlier
frozen-plugin outputs may be consumed as external comparison evidence.

The purpose is to establish an inspectable quality baseline before accepting
algorithm-changing speed/quality tradeoffs. This directory is experimental test
tooling: it does not register a VapourSynth filter, change `nss.LSSC`, or add a
runtime dependency to the C++ plugin.

## Sources and proof boundaries

- Mairal et al., ICCV 2009, [Non-local Sparse Models for Image Restoration](https://www.di.ens.fr/~fbach/iccv09_mairal.pdf):
  Eq. (6) similarity sets, Eq. (7) group residual constraints, Eq. (8) dictionary
  learning, Eq. (9) counting aggregation, Sec. 3.3 practical implementation,
  Sec. 4.1 synthetic-noise parameters.
- Tropp, Gilbert and Strauss, [Algorithms for simultaneous sparse approximation,
  Part I: Greedy pursuit](https://web.eecs.umich.edu/~martinjs/papers/TGS05-Algorithms-Simultaneous-I.pdf),
  Algorithm 1 / Sec. 3: S-OMP selects the atom maximizing **sum of absolute
  correlations** across residual columns and refits by least squares. An L2
  correlation score would be a different choice and is not silently substituted.
- The author's external `denoise.m` selects 512-atom dictionaries with patch sides
  9, 12 and 16. Its `instructions.txt` explicitly warns that the package differs
  slightly from the paper's experiment and uses a different seed protocol.
- [SciPy chi2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
  supplies the chi-square quantile; [loadmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html)
  reads an externally supplied dictionary.

The implementation here was written from mathematical descriptions. No author
MEX implementation, author source code, prelearned dictionary, or benchmark image
is included. The author's downloaded package says **academic use only**. Its
assets must not simply be bundled into a product release; distribution of model
assets is a separate unresolved requirement.

## P1 mathematical contract

1. Input is a finite grayscale image in nominal `[0,1]` units, allowing out-of-range
   AWGN samples. The CLI sigma uses 8-bit units and is converted exactly once by
   `/255`. No clipping or noise regeneration is performed.
2. The dictionary is externally supplied, `m x atoms`, float64, with unit-norm
   columns. Invalid norms are rejected, not silently corrected. The provided
   author dictionaries are loaded without retraining, recentering, or alteration.
3. Extract **all valid stride-one patches**. Locations use raster order; pixels
   within each patch use MATLAB/Fortran column-major order. Odd sizes and sides
   9/12/16 are supported independently of the old power-of-two API.
4. Subtract each noisy patch mean. For a group with `g` patches, use the paper's
   `epsilon = sigma^2 * chi2.ppf(tau, m*g)`, default `tau=0.8`. Retain the stated
   degrees of freedom; do not silently replace `m` with `m-1` after centering.
5. Fit shared-support S-OMP using float64 least squares. Each patch has its own
   amplitudes/signs. Stop at the residual constraint, not a fixed 8-atom/16-step
   budget. Rank/span exhaustion is an explicit failure, not mean-only fallback.
6. The initial SC pilot is singleton S-OMP/OMP followed by dense aggregation.
   Re-extract patches from the aggregated pilot image for matching.
7. Group by the **literal** Eq. (6) plus Sec. 4.1 interpretation:
   `raw SSD <= (32*sigma)^2/m`. Each raster-first unassigned seed collects
   unassigned matches in its semi-local window. No fixed cluster count and no
   cap on group size. Every patch belongs to exactly one final group.
8. Reconstruct final groups from the **original noisy patches**, not pilot patches
   or ground truth. Add back the original patch means and sum/count every
   overlapping estimate. No Gaussian/Kaiser window, source blending, clipping,
   deblocking filter or sharpening is inserted.

Ground truth is an optional CLI argument used only **after** denoising for metrics
and previews. It is absent from the `denoise` function's interface.

### Choices not established as author-equivalent

These are visible reference definitions, not already accepted engineering tradeoffs:

- The paper does not completely specify its greedy/disjoint clustering schedule.
  This reference uses raster seeds and seed-relative SSD. Group members are not
  guaranteed to satisfy a pairwise threshold with every other member.
- An even window of side `w` uses offsets `[-floor(w/2), floor((w-1)/2)]` on each
  axis; this convention has not been checked against the MEX binary.
- The valid-patch boundary policy produces nonuniform coverage near borders.
  Coverage is normalized explicitly. The author's border policy and pixel packing
  must still be established before claiming its numerical equivalence.
- The paper prints a `/m` in the similarity threshold, while the public wrapper
  passes `(32*sigma)^2` into a binary. Its internal distance normalization is not
  available in the wrapper. P1 uses the literal printed raw-SSD convention and
  records the exact threshold. **Do not tune away this ambiguity against clean
  images and then call the result paper-exact.**
- Using an aggregated SC pilot for matching is an explicit realization of
  improved matching, not proof of the author's exact intermediate pipeline.
- The reference uses SVD least squares and a machine-precision feasibility
  tolerance. Floating-point agreement with other solvers must be tested rather
  than assumed.

## What is intentionally missing

- Initial image-adaptive dictionary learning (LSC).
- Grouped L1,2 dictionary learning with the Eq. (8) normalization and constraints.
- A source-verified mapping of author iteration counts, stopping parameters and
  grouping details to the mathematical contract.
- MATLAB/Octave/MEX output parity, an optimized C++ implementation, a public API,
  video/temporal acceptance, and any speed/quality tradeoff approval.

**Passing these tests proves the P1 contract, not complete paper reproduction.**

## Run locally

Create an isolated environment; do not install research dependencies into the
plugin's runtime:

```sh
uv venv .venv-paper
uv pip install --python .venv-paper/bin/python -r tests/lssc_paper/requirements.txt
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-paper/bin/python -m pytest -q tests/lssc_paper
```

Use an external dictionary and a saved float32 input. Replace the illustrative
paths below; no assets are downloaded automatically:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-paper/bin/python tests/lssc_paper/run.py \
  --input /path/to/noisy.f32 --width 128 --height 128 --sigma 25 \
  --dictionary /path/to/dict_n9.mat --clean /path/to/clean.f32 \
  --out artifacts/lssc-paper/example

OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-paper/bin/python tests/lssc_paper/audit.py \
  --run artifacts/lssc-paper/example --input /path/to/noisy.f32 \
  --dictionary /path/to/dict_n9.mat
```

An output directory must be new. Failed runs remain visible and are never
silently overwritten. `--stage sc` exposes the fixed-dictionary, non-grouped
baseline separately. Each run saves output/pilot float32 images, a float64 stage
trace, source/input/dictionary hashes, complete group/support counts, residual
constraints, quality metrics, and a fixed-scale preview.

The audit separately evaluates pseudoinverse LS fits to original noisy patches,
recomputes chi-square constraints, and rebuilds images by a separate patch-loop
aggregator. The independent aggregator has a different float64 summation order;
it checks an explicit forward-rounding bound plus the final half-float32-ULP and
records float32 midpoint disagreements. This is separate from the **exact output
SHA256** requirement for repetitions of the same implementation. Tests also
distinguish the L1 SOMP selection from an L2 variant,
check rank-deficient failure, preserve individual amplitudes, reject incomplete
coverage, and test non-power-of-two block sizes.

The bounded `campaign.py` consumes the prior saved `fixtures.json` format and
optionally old result files. It verifies fixture/output hashes, runs each case
twice by default, checks deterministic output, audits every stage, and produces
comparisons. Its local Python timing is **not** comparable to the prior C4 timing.

## Work plan and acceptance order

1. **P1 / fixed-D reference (current):** establish independent solver, geometry,
   grouping and reconstruction tests; run the old 3-image x 4-noise matrix and a
   full 256 image; retain bad cases and inspect native-scale/zoomed outputs.
2. **P2 / fidelity reconciliation:** resolve patch orientation, borders, matching
   units and grouping choices against independent source/intermediate evidence.
   If unresolved, retain the limitation instead of labeling author equivalence.
3. **P3 / learned baseline:** implement L1 image-adaptive learning and grouped
   L1,2 online learning with measured optimization convergence. Compare SC, LSC
   and LSSC explicitly. An approximate penalty solve is not automatically the
   constrained Eq. (8) objective.
4. **P4 / robust quality baseline:** expand to original full benchmark images,
   multiple saved noise seeds, low-noise cases, fine lines/textures, and intended
   real/video content. Keep full-reference, current candidate and predecessor in
   every comparison. Visual judgement is primary; PSNR/SSIM are diagnostics.
5. **P5 / engineering:** first optimize algorithms without deliberately changing
   their outputs. Then permit one algorithm-changing tradeoff per experiment,
   with blind A/B views and paired same-host timing. Do not accumulate small losses
   unnoticed by comparing only against the immediately preceding candidate.

No P2-P5 stage is marked complete by P1 unit tests or an attractive sample PSNR.
