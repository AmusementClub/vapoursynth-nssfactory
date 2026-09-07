# Failure and resource contract, semantic version 2

All nine VapourSynth factories validate the original int64/double argument map
before conversion or node acquisition. Integer arguments must fit int; float
arguments must be finite and fit float. Each factory then validates its model's
legal shape and range. WNNM also validates predictive seed count/window.
Active planes smaller than a block are rejected. Invalid formats and unsupported
algorithm shapes are creation errors, not denoising bypasses.

Host frame/node references are owned by `FrameScope`/`NodeRef`. A frame scope
tracks each acquired reference separately, even when several references point to
the same underlying frame. It guards outputs until successful return. Both the
creation and get-frame C callbacks translate C++ exceptions to observable VS
errors. Matching/solver batch failures are errors, not empty-denominator copies.
Sigma-zero, unselected planes and the documented denominator-zero policy remain
legal bypasses. Public CPU batch status is 1 for a completed item and 0 for a
failed item; a positive function return identifies the first failed input index
plus one. This preserves the existing API.

Float frame strides must be positive, divisible by sizeof(float), at least the
visible row width, and have a plane span representable by the kernels' int
indexing. Fat height and derived LSSC grid/work sizes are checked before narrowing.
`Workspace::get` checks byte multiplication, reserves the replacement before
releasing the old block, and preserves the old pointer/capacity/content on failure.
Zero-size requests return a stable allocation. All dynamic vectors use checked
allocator byte counts. Array products with larger legal bounds use size_t or
checked uint64 intermediates. Host plane adapters copy only when an existing
channel-stride API cannot directly represent differing frame/reference strides.

`memory_limit_mb` applies per node, defaults to 1024 MiB, and accepts integer
values 1 through 1048576. It accounts for requested allocation capacity, not just
used vector length. Growth includes both old and replacement allocations until
the replacement succeeds. Exceeding the limit returns an error without reducing
group size, radius, iterations or model work.

`_NSSResourceBytes` is a six-element array, in this order:

| Index | Account | Lifetime |
|---|---|---|
| 0 | Workspace | Per-worker scratch for ordinary parallel filters; one serialized scratch for rolling, including its map allocation |
| 1 | Cached | Published rolling chunks owned by the node cache |
| 2 | Pinned | Evicted chunks still owned by requests |
| 3 | Inflight | Transient host/CPU vector capacities and chunks under construction |
| 4 | Output | SDK output frame bytes held by the callback |
| 5 | Framework | Borrowed input frame bytes, counted per acquired reference |

`_NSSResourcePeak` and `_NSSResourceLimit` are bytes. The snapshot is taken just
before successful output handoff; transient locals may still be alive. Peak is
node-wide owned peak (indices 0–4), and includes transient growth. Input reference
bytes are diagnostic and excluded from the admission limit because their memory
belongs to upstream nodes. SDK output bytes are measured after SDK allocation;
an over-budget output is immediately freed. Handoff removes that output from the
node's account; the VS frame cache subsequently owns it.

This is an NSS buffer budget, not a process RSS cap. The VS core/upstream caches,
allocator bookkeeping, small object/control-block metadata, bounded stack arrays,
and fixed solver thread-local arrays are outside this admission limit. The opt-in
BM transform-cache experiment has a separate process-thread lifetime. RSS and
framework reference bytes are reported separately; node teardown zeroes all
node-owned tracked capacities. Returning RSS to its exact initial value is not a
required allocator behavior.

Rolling fills run under `compute_mu`. Its Workspace is switched to serial mode
before publication, so eight successive VS workers reuse one buffer. Ordinary
fmParallel filters retain independent mutable scratch per thread. Chunk accounts
move from inflight to cached to pinned; shared ownership keeps evicted memory
charged until the final request releases it.

Tests: `test_workspace`, `test_resources`, `test_host_failure`,
`test_parameter_bounds_plugin.py`, `test_plan01_plugin.py`. The failure harness
compiles the original BM3D/rolling callbacks against a counted provider, injects
get/new/workspace failures and every observed C++ allocation position, and checks
zero surviving references/allocations. Real VS and sanitizer runs are separate
acceptance evidence.
