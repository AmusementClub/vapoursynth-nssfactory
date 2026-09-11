# TWSC model version 3 / NLH model version 5

These models replace the version 2 implementations. The contribution buffer
layout is unchanged; `VAggregate` validates the producing model's version.
TWSC requires version 3, NLH requires version 5, and other existing models remain
at version 2. NLH v5 changes the default presets and retains v4's linear-threshold
math; v4 replaced the v3 quadratic-threshold contract.
There is no compatibility algorithm switch.

## Reference identity and scope

- TWSC: [ECCV 2018 paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/XU_JUN_A_Trilateral_Weighted_ECCV_2018_paper.pdf),
  [author repository](https://github.com/csjunxu/TWSC-ECCV2018/tree/5e23808ba916885de66541119784c5b3e68a607a).
  The full three-weight objective and ADMM are retained. Demeaning, noise-corrected
  singular values, and residual-MSE noise feedback follow the selected author
  preprocessing. This combination is not a claim of equation-by-equation paper
  reproduction or MATLAB output identity.
- NLH: [paper](https://arxiv.org/pdf/1906.06834),
  [author repository](https://github.com/njusthyk1972/NLH/tree/e36d833ba39cdffc66dd048896687da31a6ea4bb).
  Versions 4 and 5 use the engineering threshold `k_h*hard_strength*sigma_channel`
  after sigma/255 conversion and color-domain variance propagation. This
  deliberately supersedes version 3's literal `tau*sigma²` rule. The iterative
  stages, structural mask, repeated Wiener gains and full pixel aggregation
  remain. The author's MEX refinement is not used. The author program is an
  external research comparison, not an oracle for this calibrated model.
- Production C++ and the NumPy/SciPy reference are independently implemented.
  Author MATLAB/MEX files and evaluation images are not plugin distribution files.
  RGB spatial filtering is the paper-facing color path; native YUV, temporal
  processing and reference clips are documented engineering extensions.

## Parameters and migration

Public sigma is an 8-bit standard deviation; all mathematical formulas use the
normalized sigma/255. No BM3D 0.75 factor applies. Scalar sigma broadcasts; shorter
plane arrays repeat their last value. Explicit zero bypasses that input plane
exactly. Samples are not clipped; nonfinite source or reference samples fail.
Preset boundaries use double-precision working-domain sigma, propagated from
the original API values before FP32 normalization. Gray therefore distinguishes
50 from 50+1e-9; explicit RGB noise first undergoes color-variance propagation.
A positive sigma too small to remain positive after FP32 normalization raises
an error; it cannot silently become the explicit-zero bypass. Representable
positive small noise still uses the documented internal inverse floor.

TWSC's defaults use RMS over active channels:

| Sigma RMS | Block | Group | Outer iterations |
|---|---:|---:|---:|
| <=20 | 7 | 70 | 8 |
| >20 and <=40 | 8 | 90 | 12 |
| >40 and <=60 | 8 | 120 | 12 |
| >60 | 9 | 140 | 14 |

An explicit block/group/iteration overrides only that field. Sigma defaults to 3;
`estimate_sigma=1` enables the NLH equation-based estimator per input plane and
conflicts with explicit sigma. `lambda2=1`, `delta=0`, `admm_iter=10`, `rho=.5`,
`mu=1.1`, `tol=1e-6`. `lambda1` has been removed, even when its value is zero.
Block is 1..16, group 1..256, outer iterations 1..64, ADMM iterations 1..1000.
`lambda2>=0`, `delta` is 0..1, rho/tol positive, and mu>=1. Sigma>100 is an
interface extension, not paper experiment coverage.

NLH omits sigma by default and estimates it once for each input frame/channel.
`noise_model="auto"` selects AWGN for Gray and real for RGB/YUV; `"awgn"` and
`"real"` are explicit alternatives. AWGN uses the maximum working-channel sigma
of the request's center frame to choose the <=50 or >50 preset.

| Preset | Basic/Wiener block | Step | Group | q | Window | Basic rounds | Wiener gains | hard_strength | Wiener sigma scale |
|---|---|---|---|---|---|---:|---:|---:|---:|
| AWGN <=50 | 8 / 16 | 6 / 15 | 16 / 16 | 4 / 4 | 40 / 40 | 4 | 2 | 1 | .32 |
| AWGN >50 | 8 / 15 | 6 / 7 | 16 / 16 | 4 / 4 | 40 / 40 | 5 | 2 | 0.7071067811865476 | .64 |
| Real | 7 / 16 | 4 / 10 | 16 / 16 | 2 / 4 | 40 / 40 | 2 | 2 | .125 | .64 |

All three use `lambda_basic=.6`. Exact fixtures are in
[`nlh_presets_v5.json`](../tests/data/nlh_presets_v5.json). The selection balances
noise levels and datasets; it does not promise a quality gain on every image.
The real preset trades lower-noise texture quality for stronger high-noise
results and lower cost. Low-noise blind estimates can amplify texture loss;
the sealed-test worst RGB case loses 7.327 dB relative to v4. See the
[evaluation report](../NLH_DEFAULTS_OPTIMIZATION_20260909.md).

Block, step, group, q and search_window accept one or two integers: one broadcasts;
two select Basic/Wiener separately. Blocks are 2..16, groups {2,4,8,16,32,64},
q {2,4,8,16} and q<=block². Basic rounds and Wiener gain applications each accept
1..64; the default search covered 1..7 and 1..3 respectively. Step must not exceed
the resolved block. Omitted blocks fit the smallest active plane geometry in the
request; omitted steps are capped at the resolved block, and omitted q is reduced
by powers of two when needed. Explicit incompatible shapes fail. Blind estimation
requires at least 8x8 per plane for its fixed bootstrap. Explicit sigma=0 still
supports exact identity on smaller images.

Explicit parameters override individual fields. Remaining omissions continue to
use the chosen preset, including on temporal requests whose estimated sigma
changes. Diagnostic properties describe the resolved options actually executed.
Basic/Wiener share no mutable reference. TWSC retains dense reference step1.
Search_window is an actual side
length (1..129), TWSC default60, NLH preset40. Explicit old `bm_range=r`
(1..64) converts to `2*r+1`; using both names fails. Radius remains 0..16;
`ps_num` defaults to 2 (1 for explicit TWSC group1), `ps_range=4` (1..64).
The shared `memory_limit_mb` resource contract applies; allocation failures
raise errors rather than reducing groups, iterations or search windows.

Old scripts must remove `lambda1`. To retain an old spatial sampling grid,
explicitly pass block, group, block_step and bm_range; algorithm outputs still
change. For the former NLH sigma=3 behavior, now pass `sigma=3` explicitly.
No old threshold or single-pass behavior is available.

The linear constant remains `k_h=2.025`. The old v4 defaults used
`hard_strength=1` and Wiener scale `.08`, selected against single-call default
BM3D paired-noise response at unchanged input sigma. V5 keeps sigma's standard
deviation meaning and reports BM3D at the same sigma; the new presets are not
an equal-response calibration on every image.

To reproduce v4 pixels, explicitly supply every old field, using the same input,
sigma or blind estimate, noise model, reference clip and temporal settings:

```python
# Gray AWGN, working sigma <=50. Use b=10, k=5 above 50;
# use b=7, k=2, noise_model="real" for old RGB/YUV auto defaults.
b, k = 8, 4
out = core.nss.NLH(
    clip, sigma=25, noise_model="awgn",
    block_size=[b, b], block_step=[1, 1], group_size=[16, 16],
    q=[4, 4], search_window=[40, 40], basic_iters=k,
    wiener_iters=2, lambda_basic=.6, hard_strength=1., wiener_sigma_scale=.08,
)
```

All three explicit recipes are in
[`nlh_presets_v4.json`](../tests/data/nlh_presets_v4.json). Omit sigma to retain
blind estimation; a blind AWGN sequence can require different low/high recipes
per frame according to its estimated working sigma. V5 still labels these outputs
with model version 5. V4 and v5 temporal contribution buffers cannot be mixed.

NLH v3 callers must remove `hard_tau`; its numeric values are not silently
reinterpreted. Use `sigma` as the normal strength control. Optional
`hard_strength` multiplies the calibrated linear threshold only; it does not
change the sigma supplied to Wiener or the noise estimator. `hard_strength=0`
disables hard coefficient rejection but retains the Basic structural mask and
the Wiener stage, so it is not an identity setting. Model version 5 is required
for temporal contributions, including explicit-zero paths.

## Formula, implementation and independent evidence map

| Operation / decision | Production source | Independent check |
|---|---|---|
| TWSC unweighted dictionary SVD after demeaning | `cpu/twsc/linalg.cpp`, `full.cpp` | known-spectrum/residual/orthogonality fixtures in `test_alignment_math.py` |
| First column noise lambda2*RMS; later lambda2*sqrt(abs(RMS²-residualMSE)) | `cpu/twsc/image.cpp` | `alignment_reference.py::twsc_contributions` |
| W3 diagonal sqrt(max(s²-n*sigma_ref²,0)) | `cpu/twsc/full.cpp` | independent full-group SVD/ADMM reference |
| W1=diag(sigma_channel^-1/2), W2=diag(sigma_column^-1/2) | `cpu/twsc/full.cpp` | fixed dictionary, unequal row/column noise fixtures |
| Minimize ||W1(Y-D*S*C)W2||F² + ||C||1 | `cpu/twsc/full.cpp` | SciPy Schur Sylvester, every ADMM iterate, objective and long-run KKT |
| Three Frobenius stopping tests; finite iteration cap is normal unconverged completion | `cpu/twsc/full.cpp` | converged and capped fixed-D cases |
| Restore mean and aggregate with inverse column noise | `cpu/twsc/image.cpp` | independent spatial and synchronous temporal image references |
| NLH equation (3)-(4) distance estimator, excludes self, global average, no sqrt2 correction | `cpu/common/image.cpp` | direct pixel-distance bootstrap reference |
| Basic mix lambda*previous+(1-lambda)*original; rematch each round | `cpu/nlh/image.cpp` | independent image reference |
| Orthonormal Haar, linear threshold k_h*hard_strength*sigma, then last-two-row/nonfirst-column structural mask | `cpu/nlh/full.cpp`, `haar.cpp` | explicit NumPy Haar matrices, sample/noise scale covariance, all supported q/group lengths, no DC exemption |
| Wiener gains R²/(R²+(scale*sigma)²), repeated wiener_iters times against fixed Basic | `cpu/nlh/full.cpp` | direct matrix reference; no Wiener structural mask or DC exemption |
| Every q*m restored value contributes to its original pixel; uniform counts | `cpu/nlh/full.cpp`, `image.cpp` | independent full num/count matrices and final images |
| Color, zero planes, format sizes, reference-only matching, version, memory and temporal requests | `host/full_image.cpp` | `test_full_image_plugin.py` public matrix / error cases |

Source paths in this table are relative to `src/`; test paths are under `tests/`.
The mathematical reference does not call production finishers or decomposition.

With P=D*S, each coefficient step solves A*C+C*B=E, where
A=P'W1²P, B_j=.5*rho*max(sigma_j,1e-6), and
E=P'W1²Y+.5*(rho*Z-U)*max(sigma_j,1e-6). Auxiliary Z uses soft threshold 1/rho;
U updates by rho*(C-Z), followed by rho*=mu. Exactly zero corrected singular
directions are excluded. An all-zero active subspace returns the group mean.

## Numerical and optimization boundaries

The TWSC workspace is isolated from other filters' 256x32 SVD capacity and
supports at most 768x256. Householder QR + one-sided Jacobi SVD uses relative
correlation tolerance 1e-6, squared rank floor 1e-14, bounded power-of-two
rescaling and at most 128 sweeps. FP32 factorization must meet reconstruction
residual 5e-5 and effective U/V orthogonality 5e-4; otherwise FP64 recomputes it.
Repeated/clustered spectra (gap<=1e-4 of maximum) use a common FP64 path for
single/batch consistency. Uniform-row-noise groups above 32 columns use FP64
directly to retain the orthogonal-dictionary property. Failure after the guarded
path is an error. Statistics, Gram matrices, eigensolves and residual checks use
FP64; centered samples and final groups have explicit FP32 boundaries.

The AVX2 target validates large SVDs (`n>32`, `min(m,n)>=32`) with SIMD across
independent reconstructed samples and orthogonality dot products. Every dot
product and final error sum retains the scalar operation order, with FMA
contraction disabled. The residual/rank/orthogonality limits are unchanged.
Other targets and the generic control retain the scalar validator. See the
[scoped timing and sampling report](../docs/twsc-speed-20260909.md).

The symmetric eigensolve is cached per group, then reused at each ADMM step.
Sylvester relative residual must be <=1e-10; a Cholesky fallback is available.
For off-diagonal Gram Frobenius norm <=1e-12 of full Gram norm, an elementwise
equivalent ADMM path uses a conservative Sylvester residual upper bound against
the full Gram. This is a bound, not a directly measured residual. Positive-noise
inverse floors are 1e-6 normalized units; explicit zero input planes bypass.

Ordered batches reuse workspace and the guarded 8-column cross-group SVD where
eligible. Uniform-noise large groups also use independent FP64 SIMD lanes with
the scalar reduction and rotation order, up to 16 groups per ordered batch.
Scalar and lane decompositions disable FMA contraction to preserve this order;
padding a short lane batch does not alter a real group's result. The same residual
and orthogonality checks remain mandatory. Memory budgets reduce batch capacity only. Highway FP64 matrix
products, distance selection and Haar butterflies retain their math. Configure
`-DNSS_ALIGNMENT_GENERIC=ON` to build a same-model validation control using
general scalar distance/Haar/math and dense Sylvester paths; this is a build
control, not a public preset or legacy algorithm. Speed claims require matching
inputs, parameters, compiler, ISA and complete pipeline timing on the same host.

### NLH admitted paths

Haar processes independent rows with SIMD for q8/q16, and q4/group>=32;
the existing q4/group16 kernel remains. On AVX3, q2/group>=8 also uses this row
path; q8/q16 with actual groups 8/16/32/64 additionally transform independent
columns with gather/scatter. Butterfly and rounding order within each row/column
is preserved. Smaller boundary groups retain the general routes.

AVX3 PixelMatch uses repeated full-row SSD plus SIMD selection for block² in
65..256, actual group 8..64 and q 2..16. This admits all blocks 9..16 under those
conditions. It retains the original column/FMA order, index tie-break and
exceptional-distance insertion fallback. Other targets and smaller groups retain
the previous upper-triangle method. Multi-channel blind estimation shares matching
selectors within equal geometries and preserves each channel's reduction order.

The C4 timing evidence covers AVX3; the functional AVX2/NEON checks do not imply
matching speedups on those targets. No block-specific complete filter or fused
Haar/filter/aggregation algorithm was added. See the configuration-specific
[report](../docs/nlh-defaults-optimization-20260909.md).

## Image conventions and extensions

Windows use offsets [-floor(W/2),ceil(W/2)-1], clipped to valid block origins.
Reference positions are raster ordered, including the final valid x/y origin
when step does not land there. Self is first; ties use frame/y/x order. TWSC uses
the actual available group. NLH keeps the largest power-of-two group not exceeding
the available count (possibly one at tiny boundaries); only that subset is
filtered and aggregated. Reduced boundary groups do not prove full-group runs.
Uncovered pixels retain their original sample.

TWSC jointly processes active same-sized planes: RGB/YUV444 together and
subsampled chroma together on its native grid. Zero-sigma planes are excluded.
NLH converts RGB to fixed full-range BT.601 Y/Cb/Cr with centered chroma, propagates
explicit independent input-plane noise variances, and matches using shared luma.
Blind bootstrap uses b8/g16/q4/W40/step1 with shared luma selectors but each
channel's own intensities. Native YUV grids and sample values are retained;
area-averaged luma guides chroma matching. Explicit zero RGB planes are restored
exactly after inverse conversion.

Each temporal request loads the genuine clipped n±radius window. All intermediate
rounds fully process every reference frame within that fixed window and aggregate
synchronously. Only the final center reference produces outgoing contributions;
`VAggregate` performs the existing cross-request combination. Missing temporal
slots are zero; identity planes contribute only at the center. `rclip` supplies
block and NLH pixel matching guides, including the noise-estimation bootstrap,
never noisy data, TWSC dictionary
samples, or the internal fixed Basic Wiener coefficients.

Frame properties expose `_NSSModelVersion`, `_NSSSigma` (working-domain 8-bit
units), `_NSSBlockSize`, `_NSSGroupSize`, `_NSSBlockStep`, `_NSSSearchWindow`,
`_NSSIterations`, `_NSSGroups`, `_NSSADMMMaxIterGroups`, `_NSSSvdDoubleGroups`,
and `_NSSSylvesterResidual`. Requested group size is distinct from actual
boundary group sizes; max-iteration counts are not numerical-failure counts.
NLH additionally exposes `_NSSQ`, `_NSSLambdaBasic`, `_NSSHardStrength`,
`_NSSHardCoefficient` (the effective linear multiplier), and `_NSSWienerSigmaScale`
for reproducible runs. Private omitted-value sentinels are never exposed.
