# Noise profile 1 (BM3D), semantic version 2

BM3D users supply sigma in 8-bit sample units on every backend. The model entry
conversion is `(user_sigma * (1 / 255)) * 0.75`, with that expression order retained.
Kernel entry points receive effective sigma. The fused 8×8×8 transform multiplies
it by coefficient gain 64 exactly once. The orthonormal generic/direct/portable
routes use the same effective sigma with their own transform normalization.
`NoiseProfile` in `include/nss/contracts.hpp` is the common source of these values.
BM3D frame properties advertise `_NSSNoiseProfile=1`; the other models advertise
0 and do not apply BM3D's calibration factor.

The hard stage retains DC and coefficients at or above `2.7 * sigma_effective`.
Wiener uses the provided reference coefficients, preserves DC, and uses squared
Wiener gains for aggregation weight. The configured group is the transform length;
only k actual patches contribute. Missing transform slots are zero padded, without
inventing matches. The standalone 128-bit portable route is callable explicitly
and has the same fixed-match contract. No ISA-specific user-sigma compensation is
permitted.

Compatibility consists of three distinct measurements:

1. Fixed matches: generic/direct/fused/portable against an independent double
   cosine-transform oracle, including k=1…8, separate strides, DC, nextafter
   threshold neighbors and sigma
   0/0.1/1/3/10/25/50. This is the backend numerical-equivalence contract.
2. The pinned bm3dcpu Basic, fixed-reference Final and Basic→Final curves. The
   reference is WolframRhodium/VapourSynth-BM3DCUDA commit
   `e869cfae8d2322cf7a2b3c8056ed57e9308b2d33`; its source archive is
   `374163cf7e38c6017c041833237f0400e2e1e86ec48af3b4652f3b2677b10a5d`.
   Run `tests/c4_bm_migration.py` and retain its binary/input hashes and quality
   curves. Public matching/window/temporal policies differ; this is a calibrated
   comparison, not a claim of universal pixelwise compatibility.
3. Old NSS generic migration: for the former generic route that received
   `old_sigma / 255`, use `new_sigma = old_sigma * 4/3` to preserve its effective
   sigma. Old 8×8×8 spatial fused configuration keeps the same user sigma. Old
   temporal outputs that mixed query/contribution identities have no valid
   sigma-only compatibility adapter. They are not correctness goldens.

The migration factor belongs at the user's legacy-model conversion boundary,
never inside NEON/CUDA/Metal/Vulkan dispatch. Run the pinned reference comparisons
for each intended source, shape and stage, and retain their measurement records.

Threshold-neighbor tests retain the discrete keep/drop interpretation. A coefficient
within the declared 2e-5 forward-transform error band may lie on either side after
FP32 rounding, but its output must match one of those two valid masks, including
DC reconstruction and weight. Outside the band the independent mask is mandatory.
This does not relax candidate identity or permit unrelated image deviations.
