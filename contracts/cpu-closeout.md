# CPU closeout decisions (O00–O07)

These decisions preserve the accepted default CPU configuration. Correctness,
resource ownership and compiler fixes do not count as a new speculative
optimization campaign. `NSS_BM_EXPERIMENT=2305`, `NSS_AVX2_DEFAULTS=ON`, and
`NSS_AVX2_EXPERIMENT=0` remain the product defaults. The resulting AVX2 candidate
mask is 1789, filtered by the existing immutable algorithm/shape policy.
Historical BM and AVX2 speedup factors must not be multiplied.

| ID | Decision | Scope and reopening condition |
|---|---|---|
| O00 | KEEP / refresh evidence | Preserve accepted defaults. Archive original stage-1, corrected-math and final identities, then compare full workloads with paired wall time, CPU affinity, source/binary/input hashes, output/clean metrics and resource observations. |
| O01 | KEEP accepted reuse; DEFER homogeneous/lazy raster | Keep buffer reuse/ring/scratch. Reopen rejected organization variants only with a new measured whole-filter bottleneck. |
| O02 | KEEP cross-group SVD/Q-replay | Apply shared scale/rank corrections first. No new live-set or nondefault-shape performance claim is made. |
| O03 | KEEP NLM stripe; DEFER speculative rewrite | Reopen only for an implementation that removes actual horizontal/vertical or Welsch intermediate traffic without changing borders or normalization. |
| O04 | DEFER frame-pair/displacement reuse | Requires exact candidate/boundary proof, measured memory, and a full-frame improvement. |
| O05 | KEEP accepted AVX2 b12/b16 routes | Preserve their existing algorithm/shape permissions and prior two-machine acceptance record. No inference from these x86 results to ARM speed. |
| O06 | KEEP selected NLH/Haar and LSSC/MCWNNM GEMM | Existing q2/g8 fallback control costs remain disclosed in `docs/avx2-port-campaign.md`. Include those controls in the new regression matrix; further optimization is DEFER pending a new causal hotspot. |
| O07 | KEEP dedicated rolling workspace and budget | Eight worker handoffs use one serialized scratch. Cache/inflight/pinned/output accounting, budget errors and cleanup are tested. New cross-chunk contribution reuse is DEFER. |

The previously rejected same-layout first4/heap/ZMM variants, MCWNNM 16-tall
lockstep and global fast-math changes are not reopened. Internal index sorting
uses a total input-index order, so a stable-sort heap buffer can be removed
without changing the order of any group; the index vector remains budgeted.

O07 acceptance is correctness/resource safety plus disclosed full-workload cost.
It is not a requirement that every mathematical correction be faster than an
incorrect baseline. The corrected-math snapshot isolates costs of ownership,
accounting and finisher integration; the original stage-1 comparison records the
full product change, with semantic differences identified independently.

The completed integration has measured regressions in some configurations.
`CPU_CLOSEOUT` records the implementation, safety evidence and KEEP/DEFER
decisions. Performance acceptance is reported separately in
[the final Plan01 report](../PLAN01_REVIEW_20260906.md), with original and
confirmation workloads kept distinct. The current C4 reference-off controls and
the historical 5950X fallback costs are separate hardware/compiler observations.
