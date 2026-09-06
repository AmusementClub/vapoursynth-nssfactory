# BM3D correctness and dual-region C4 campaign

Status: P0 passes both C4 hosts; optimization admission and final delivery remain
in progress. A screening pass is not a nine-image or final combination pass.

## Source and correctness contracts

The original dirty checkout is preserved. `d572c52` records B0, archive SHA256
`e28a51339429399b47809a8f24ccd6f096ff4bc2dbb796eb34a0471d8c7fc70f`.
`4154cdf` records the initial P0 implementation. Later P0 repairs cover aliased
Wiener references, partially disabled joint TWSC/MCWNNM output planes, and safe
frame-index arithmetic near INT_MAX. The original ignored documentation was
copied intact before appending this campaign; it was absent from B0's tracked
source export.

- Every temporal candidate is compared with the original center reference patch.
  Forward and backward prediction are independent. Each layer scans the raster
  union of its predicted windows once. Only the original self is forced into the
  retained group. Single-channel and joint-channel matchers share this control
  flow, retaining per-frame and per-channel strides.
- SearchConfig carries an effective half-open temporal range. Padded slots keep
  their historical buffer indexes but do not participate in search/contribution.
- Target n consumes centers max(0,n-r)..min(last,n+r), slice n-center+r. Each
  contribution uses its own stride; zero total weight falls back to source n.
  VAggregate declares general dependencies. Disabled fat planes contribute only
  to their center slice; no cross-frame identity averaging is permitted.
- Rolling builds centers chunk+-r with source/ref support chunk+-2r. A two-stage
  reference expands dependencies again through its Basic graph. The independent
  target oracle is the authority; equality with the legacy path alone is not.
- BM3D host adaptation applies sigma_eff = 0.75*sigma_user/255 exactly once per
  public call. Fused 8x8/8-group filtering applies only the additional transform
  factor 64; inverse scale remains 4096. Generic Basic uses 2.7*sigma_eff, Wiener
  uses sigma_eff squared. Low-level BM3D group/direct/batch callers now supply
  sigma_eff. Other algorithms retain their original sigma units.

The F experiment source SHA256 is
`96dd4a177b7cbd107cc764a794d822fc0b42838c7eb72bb3b55f08d2297b3b9b`.
The canonical value is recorded in each source archive sidecar and source
manifest; those files, not this narrative, drive validation.

## Protocol

Each same-host gate alternates seven A/B pairs and extends once to fifteen when
its confidence interval straddles a decision boundary. True-chain geomean must
be >=1.01, with CI95 lower>1. Every control lower bound must be >=1/1.01.
CPU0 affinity, CPU1 idle>=99.9%, steal=0. Reports include per-case environment
checks; a favorable aggregate cannot override an invalid individual case.

The driver pre-generates source frames and warms the graph outside frame timing.
Timing includes required target aggregation and the actual Basic-to-Wiener
reference graph. Numerical dumps and output hashes occur outside timing. Hashes
are provenance, not numerical pass/fail criteria. max_abs>1e-5 or RMSE>1e-6
requires causal stage replay and clean-reference quality losses <=0.05 dB PSNR
and <=0.001 SSIM before further admission. No thresholds are silently widened.

Full target sets contain all nine 1080p samples and Basic, common-reference
Wiener, and true two-stage outputs. The control set has 51 cases: all b8 groups
and seven shared algorithms retain MAPPA 1080p, other blocks use bounded 320x180
stress, and temporal controls cover radius 1/4 and step 2/4/8. Expensive 1080p
controls time one new frame after two warm-up frames. Target sets time two new
frames after two warm-up frames, except rolling which rounds to full chunks.
Large-block/radius controls are intentionally bounded, not full-size Cartesian
products. Random rolling controls span multiple chunks with cache limit one.

## Candidate dispositions

This section records the earlier strict campaign. The user's subsequent
per-configuration >1.02x policy and the new selected default are documented in
[the September 6 selection report](c4-selection-20260906.md). The historical
gate failures below remain unchanged and do not describe the new default.

All experiment bits default off. No candidate passed the complete admission gate.

| Bit | Structure | Recorded screening result |
|---:|---|---|
| 1 | SortedTopK | H nine-image chain 1.68818x, but TWSC control lower CI 0.98735 and one CPU1 violation; not admitted |
| 2 | Cached-worst TopK | I nine-image chain 1.35690x; b16, LSSC, small-group/NCSR bounds and one CPU1 violation fail controls; not admitted |
| 4 | Bounded patch/ref/work reuse, lazy raster jobs, homogeneous dispatch | I revised nine-image chain 1.01252x, CI [1.01168,1.01303]; full controls fail, not admitted |
| 8 | DCT4/8 butterfly | Full-image b8 threshold transition replayed; independent 4x4 completion does not admit the unresolved full configuration surface |
| 16 | Frequency-tile group transform/shrink/inverse | Scalar prototype revised once to SIMD tiles; E chain 0.987x, not admitted |
| 32 | Paired first-four-row conservative SSD bound | Revised to preserve the original accumulation tree; E chain 0.858x, not admitted |
| 64 | Per-displacement double prefix SSD, bounded host batches | 224 small cases pass; MAPPA Basic 0.124x and numerical triage failure, not admitted |
| 128 | Bounded forward-DCT memo with real patch identity and epochs | Revised once from payload lookup to keyed epochs; E chain 0.929x, not admitted |
| 256 | Target ring and direct scratch contribution extraction | Corrected N/P timings are budget-limited; the full-image chain has one pair only (1.08536x), insufficient for admission |
| 512 | Existing 8x8 pair-in-register layout | H nine-image chain 1.01079x; singleton/b16 regressions and CPU1 violations remain after the one revision; not admitted |
| 1024 | Conservative DC-coefficient SSD bound, double accumulation | G chain 0.22061x; not admitted |

Failed or inconclusive experiments remain explicit opt-ins. No combined candidate
is promoted, and no unmeasured crossover is inferred.

## Numerical and measurement investigations

Both F hosts passed 16/16 CTest, 224 plugin comparison cases, the independent
public sigma/target-time/disabled-plane/concurrency oracles, and retained rolling
and VAggregate regressions. Public sigma tests cover every legal block and group
with sigma 0,0.5,1,3,5,10,20,40 against an independent double cosine-transform
reference. Temporal tests include radius 16, one-frame input, short arrays,
joint disabled channels, aliasing, boundary truncation, and actual two-stage
rolling dependencies.

The first timing driver left its parent unpinned. Parent array/bootstrap work ran
on CPU1 and invalidated those timings. A live input-hash change was detected and
failed closed. Subsequent drivers pin the parent, use RAM numerical dumps, flush
pending I/O before CPU snapshots, and confine system/user services to CPU0 during
formal timing. Initial invalid results are retained as evidence, not pooled.

The E ring screen accidentally timed two cached frames after warm-up. The F
protocol aligns the first measured frame to a new chunk boundary and rounds the
measured count to whole chunks. E ring numbers cannot support admission.

The user's September 6 duration revision budgets each complete A/B configuration
group for approximately 30-45 seconds. Driver N uses `--group-seconds 40`, probes
both variants, includes subprocess startup and warm-up in its budget, and fixes
the same calibrated frame count for both variants. Rolling still measures whole
new chunks. Calibration is separate from the paired samples. Seven pairs are
retained where they fit; smaller counts cannot pass the formal admission gate.
Automatic extension is disabled under a time budget, and an unfinished pair is
excluded on timeout. `budgets.json` retains the calibration and actual wall time.
Completed historical groups remain unchanged. Initial real C4 rolling groups
took 35.4 seconds (32 frames) and 32.7 seconds (24 frames), including calibration.
The P/Q driver probes four chunks for random access so calibration exercises cache
misses, and reserves headroom for larger random windows. If only the minimal
calibration A/B pair fits, it is retained as a one-pair diagnostic with its arrays.
Whole-chunk granularity and the minimum useful pair count can produce groups
shorter than 30 seconds; the driver does not pad the time with idle waits or call
an incomplete group a formal result. Timeout excludes the entire unfinished pair.
The final Q random-access checks took 40.00, 28.77, 28.26 and 27.86 seconds,
with 42, 64, 12 and 48 frames respectively. Completed pairs were 6, 1, 7 and 1;
these checks validate the time budget, not candidate admission. All six budget
tests passed locally and on C4, including timeout and insufficient-pair rejection.

Foundation M's pure-P0 AVX2 contract test exposed a test-only absolute tolerance
issue: 16.9999980927 versus 17 is one float rounding unit. The N test overlay uses
two float epsilons scaled by the expected magnitude for this aggregation oracle;
padding checks remain exact. This does not widen output admission thresholds.
The next AVX2 oracle found a real disabled-plane identity violation: reciprocal
division changed 0.05 by 3.72529e-9 even with total weight one. P directly retains
the numerator when total weight is one; disabled-plane comparisons remain exact.

F's broad SortedTopK experiment regressed b4/b12/b16 and small b8 groups. Retained
AVX3 disassembly showed the b4 hot function shrinking from 2,181 to 330 assembly
lines and calling SortedTopK::add through the PLT for each candidate, whereas
the baseline inlined it. The one H revision keeps the new sorted policy separate
from the established class, forces only its short insertion path inline, and
selects it for b8/group>=16. Temporal global/layer selection remains unchanged.
H's target nine-image chain CI was [1.68624,1.68992], but the full control gate
still did not establish TWSC's 1% regression bound after the permitted sampling.
No selective favorable rerun is used to override that disposition.

H controls were calibrated from frozen F baseline durations to target at least
150 ms per timed window. This mitigates sub-millisecond control noise without
changing the 1% or CPU1 gates. Dedicated guest workqueues and modifiable IRQs
were also assigned CPU0 between runs; their prior/new affinity and any managed
IRQ failures are recorded. A b32/g8 control still had CPU1 idle 0.998957 and
therefore remains invalid despite a neutral timing ratio.

The 8x8 butterfly's first full-image Basic case had max_abs 0.0005797744 and RMSE
0.00000327605. Fixed-match replay around (1900,200) found one coefficient only
7.63e-9 from the threshold; its keep decision changed the group weight from 1/60
to 1/61. The measured clean PSNR loss was 0.000003486 dB and SSIM loss 3.04e-8.
This explains that specific Basic/r0/b8/g16/sigma3 case only. It does not admit
other stages/configurations without their own numerical evidence.
The final replay audit restores the probe's round-trippable decimal values to
float32 before converting them to double, avoiding decimal serialization error
in the independent reference. The original classification is retained alongside
that audit; both identify the same single threshold transition.

Prefix SSD's MAPPA Basic/step2 failure had max_abs 0.0002926588, RMSE 6.68366e-7,
and 144 differing values, with maximum at (1112,138). It is not admitted; original
arrays are retained. Of 225 replayed queries around the affected output, one at
(1106,134) changes group order. The original SSD ties (1101,131) and (1107,134)
at 0.0315589979; prefix accumulation ranks the latter at 0.0315589942. The patch
set is unchanged but its group-axis transform order changes. This locates a
matching-stage cause without claiming complete reconstruction of every output
difference. The measured slowdown already fails performance.

## Final verification and diagnostics

YUL passed dynamic, AVX2 and AVX3 default-off builds: each has 16/16 CTest and
230/230 numerical comparisons against the independently assembled pure P0 source,
plus independent sigma, temporal, disabled-plane, aliasing and concurrency oracles.
Dynamic and AVX2 lab builds each passed 18/18 CTest and the semantic oracles.
TLV independently passed the dynamic build, 16/16 CTest and 230/230 comparisons.
Both hosts passed retained rolling (max_abs=0) and VAggregate plugin regressions.
Every corresponding complete plugin binary, not just its text section, matches
the pure P0 reference; the dynamic binary also matches across regions. With an
empty admitted candidate set this establishes default-off identity, not a new
performance gain. `final-verification.json` records binary and source hashes.

The pure reference is built from 4154cdf plus the later correctness repairs.
The J reference builder initially omitted BM3D's overflow-safe valid-slot end;
the P reference includes that repair before the final identity comparison.
The runtime source tested is P (SHA256
862eff2a8cc7cc3caa6b5e1a8fd714b6eb743c3d0fb4fa605bc86b228d1f6bab).

The I buffer candidate passed its nine-image target but failed b4/g32 Basic,
b16/g8 chain, and shared NLM regression bounds. Its target pass alone is not
admission. `remaining-bench-summary.json` recomputes these decisions from raw pairs.

Pinned BM3DCPU migration completed 1,080 cases over nine real-image sources,
five synthetic temporal conditions, eight sigma values, and three stages. For
nonzero sigma, static chain clean PSNR differences average +0.00155 dB, with
worst -0.00078 dB; synthetic temporal chain means range +0.263 to +0.371 dB.
These are migration diagnostics between implementations, not numerical identity
or speed gates. No natural video was supplied or certified.

The bounded B0-to-F correctness-cost diagnostic has spatial chain ratios near
one (g8 0.99764, g16 1.00700). For radius one, B0/B1 ratios are 0.73928 Basic,
0.76804 common-ref Wiener and 0.75616 chain: approximately 35%, 30% and 32% more
time for the corrected semantics. The chain has only three pairs under the new
budget. This is an earlier F artifact, not a final P performance claim, and it
cannot pass an optimization gate.

Memory/revisit diagnostics completed 640x360, 1080p and 4K with repeated hashes
unchanged. Baseline/cache process peaks were 416,192/444,116 KiB; the 4K process
peak was 1,209,164 KiB. Three revisit epochs and retained RSS are preserved, but
do not prove a steady-state leak bound. The bounded DCT memo recorded 640,843
hits and 46,821,557 misses (1.35% hits). Allocation counters are process-wide,
including framework and array activity, and timings include output hashing.

Nine PMU profiles retain nonzero hardware counts, perf data, annotations and zero
lost samples. For the g32 chain, sorted TopK reduces bad speculation from 19.0%
to 10.0%, while front-end share rises from 22.4% to 31.9%. This diagnostic agrees
with a reduced matching workload but does not override failed cross-algorithm
controls. `diagnostics-summary.json` records the migration, memory and SSD replay.

Evidence is under `artifacts/c4/bm3d-correctness-20260905`; 1,050 payload files
passed local SHA256 verification. The exact TLV and YUL campaign Spot instances
and their auto-delete boot disks were deleted. `cleanup-tlv.json` and
`cleanup-yul.json` retain the helper's absence checks. Regional snapshots and
unrelated resources were preserved. Changes remain in the implementation
worktree; the original checkout was not modified or merged.

BM3DCPU is pinned to e869cfae8d2322cf7a2b3c8056ed57e9308b2d33. Migration uses
explicit range 7; range 9 is a separate diagnostic. Synthetic translation,
brightness, occlusion and cuts do not establish natural-video acceptance.
