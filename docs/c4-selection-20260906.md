# Bounded Cross-Algorithm Selection

The user authorized a new selection policy: accept configuration speedups above
1.02x, keep each complete benchmark group around 30-60 seconds, and use at most
two Tel Aviv and one Montreal C4 Spot hosts concurrently. This supersedes the
previous all-control 1% nonregression admission requirement for this campaign.
It does not supersede numerical correctness or allow a failed environment to be
called valid evidence. Single-pair results are identified explicitly.

Implementation remains in the `codex/bm3d-correctness-c4-20260905` worktree.
Historical evidence and the original checkout are preserved.

## Independent Candidates

| Mask | Candidate | Lane |
|---:|---|---|
| 1 | SortedTopK, existing b8/group>=16 AVX3 selection | TLV1 |
| 2 | Cached-worst TopK, shared matcher policy | TLV1 |
| 2048 | Bounded patch/ref/work reuse and redundant scratch removal | TLV2 |
| 4096 | Lazy raster jobs | TLV2 |
| 8192 | Homogeneous batch dispatch and homogeneous bucket fast path | TLV2 |
| 16384 | Direct scratch contribution extraction | YUL |
| 32768 | Target ring with original fat contribution extraction | YUL |
| 256 | Target ring plus direct scratch extraction | YUL |
| 2305 | SortedTopK plus buffer reuse plus ring/scratch | YUL; AVX2 correctness on TLV1 |

Legacy masks 4 and 256 retain their combined meanings. The new masks allow
independent measurement without removing the older candidates. Zero remains
the pure-correctness reference; the selected default is now 2305.

## Measurement

Each host uses its own frozen baseline and candidate libraries, CPU0 affinity,
sample hashes, numerical matrix, and semantic tests. No timing ratios are pooled
across hosts. `c4_selection_configs.py` records the exact matrix: real MAPPA
1080p BM3D target cases, bounded block/temporal controls, and seven other
algorithms at 640x360. Ring applies only to rolling BM3D and is tested across
sequential/random access and radius/chunk sizes. WNNM, TWSC, NCSR and MCWNNM
use g32 to exercise the relevant matcher; NLH supports at most g16, so its
corrected case uses temporal g16 at 320x180. MCWNNM receives a 320x180 bounded
supplement when its original 640x360 case exceeds the budget. These are static
synthetic sources for shared algorithms, not natural-video performance claims.

The driver uses `--group-seconds 45 --selection-threshold 1.02`. Calibration
includes initialization and warm-up; measured frame time excludes them. Timed
frame counts are shared between A and B, rolling uses whole new chunks, and
partial pairs are discarded at timeout. Groups can finish below 30 seconds
when chunk granularity or minimum useful sampling prevents filling the budget.

`selection.json` is the new per-configuration decision. The historical
`decision.json` remains the old formal gate and must not be confused with it.
The new selection requires numerical checks and a valid per-case environment,
but not the old global control gate or seven-pair minimum. Confidence intervals,
pair counts, timing budget, and missing configurations remain in the evidence.

## Resources

- `nss-c4-select-tlv1-20260906`, `me-west1-b`
- `nss-c4-select-tlv2-20260906`, `me-west1-b`
- `nss-c4-select-yul-20260906`, `northamerica-northeast1-a`

All three were provisioned from their matching regional snapshots and passed
guest verification with real nonzero PMU counters. Exact instances and their
auto-delete boot disks are cleaned up after evidence download and verification.

## Results

SortedTopK's measured MAPPA chains are approximately 1.615x (g16), 1.675x (g32),
and 1.381x (bounded temporal g16). Shared g32 WNNM/TWSC/NCSR are about 1.05x;
temporal NLH g16 is 1.205x. Cached-worst's corresponding BM3D spatial chains
are 1.438x/1.390x; it also reaches 1.032x on the b4/g32 case. These configuration
results meet the new selection policy where environment and numerical checks
pass; they do not imply identical gains for all group sizes.

Buffer reuse alone gives about 1.026x on temporal BM3D, but approximately one
on the spatial targets. Lazy raster has no accepted configuration in its bounded
matrix. Separate scratch extraction and ring each show gains, with the combined
ring/scratch strongest so far (about 1.08-1.11x on completed cases).

The original g32 NLH configuration was invalid. Its run stopped after retaining
completed earlier cases. Input hashes were reverified, and only the remaining
cases resumed under the corrected configuration; the failed run remains in the
evidence. Driver overlays B/C and scripts are retained separately from the
original source archive, rather than silently rewriting in-flight inputs.

The final combination is SortedTopK + bounded reuse + ring/direct scratch
(`NSS_BM_EXPERIMENT=2305`). Fresh CMake configuration selects this value; existing
caches retain their old value until explicitly changed. Resetting the cache
entry produced a plugin byte-identical to the measured combination. Default
spatial b8/g8, LSSC and MCWNNM remained approximately neutral; the results below
must not be generalized to every public default configuration.

| Final configuration | Baseline / candidate | Pairs | Scope |
|---|---:|---:|---|
| BM3D b8/g16 chain | 1.66154x | 7 | MAPPA 1080p |
| BM3D b8/g32 chain | 1.71491x | 3 | MAPPA 1080p |
| BM3D b8/g64 chain | 1.55088x | 7 | Synthetic 640x360 |
| BM3D b4/g32 chain | 1.03717x | 7 | Synthetic 320x180 |
| BM3D temporal g16 chain | 1.46148x | 7 | Synthetic 640x360, radius 1 |
| WNNM g32 | 1.05590x | 7 | Synthetic 640x360 |
| TWSC g32 | 1.03261x | 3 | Synthetic 640x360 |
| NCSR g32 | 1.04646x | 3 | Synthetic 640x360 |
| NLH temporal g16 | 1.21326x | 7 | Synthetic 320x180, radius 1 |
| Rolling r1/chunk8 sequential | 1.12672x | 1 | MAPPA 1080p |
| Rolling r1/chunk2 sequential | 1.09602x | 7 | Synthetic 640x360 |
| Rolling r4/chunk8 sequential | 1.10499x | 3 | Synthetic 320x180 |
| Rolling random cases | 1.08689-1.09499x | 1-5 | Bounded 320x180 / 160x96 |

Single-pair entries are accepted under the user's new policy, with explicitly
limited repeatability evidence. They are not old seven-pair formal-gate passes.
No natural-video performance claim is made. Cross-host ratios were never pooled.

Cached-worst is accepted for its qualifying configurations and retained as mask 2,
but SortedTopK is preferred in the default combination because it is faster on
the main b8 and shared g32 targets. MCWNNM's bounded supplement was approximately
1.00173x for SortedTopK, 0.97240x for cached-worst, and 0.99880x for the selected
combination. Neither lazy raster nor homogeneous dispatch reached 1.02x in the
completed matrix; both remain disabled in the default combination.

## Verification and Evidence

All eight independent masks and the combined mask passed 16/16 CTest, the
independent semantic suite and 230/230 numerical comparisons. The combined
AVX2 build also passed 16/16 CTest, semantic tests and 230/230 comparisons.
Default and AVX2 rolling regressions had max_abs=0; VAggregate regressions passed.
The selected default's complete plugin SHA256 is
`12b267a529be26716dc295e8d8a1a232223fd02f2dfee5b542daadced84cc204`.

The tested final runtime was verified against archive SHA256
`243acea4ba422ce817c378abe9a3b5bab76cff8147708e260c3261ef1a2db391`;
documentation and test/orchestration updates after that archive do not change
the verified runtime files. The initial source archive was
`787650f019b1a417c9f340b808177e85f0a66fdfce4ca8867bfa5def7bf89d7e`.

Evidence is in `artifacts/c4/selection-20260906`. All 497 payload files passed
local SHA256 verification. `selection-audit.json` recomputes 91 completed
configuration results from raw pairs and independently checks CPU1/steal and
the new threshold; 37 entries qualify, including overlapping standalone and
combined configurations. `verification.json` records source and test checks.
Completed group wall times were 20.40-49.29 seconds; chunk/minimum-pair granularity
caused the shorter groups. Budget-expired groups and superseded partial runs are
preserved and not represented as completed passes.

All three exact instances and their auto-delete boot disks were deleted. The
helper cleanup records are stored alongside the evidence; independent instance
and disk inventory queries also returned no matching resources.
Regional snapshots and unrelated resources are preserved. No merge or push was
performed; changes remain in the dedicated implementation worktree.
