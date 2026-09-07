# Temporal contribution contract, version 2

A center c emits ordered contributions to destinations c-r…c+r. Its fat frame
stores alternating numerator/denominator planes vertically, one pair per slice.
Slice s means destination c+s-r. VAggregate for target t requests only actual
centers max(0,t-r)…min(last,t+r) and reads slice t-c+r. Repeated boundary slots
must not be counted as additional centers or matches. Source/reference requests
use the shared overflow-safe temporal helpers.

Every producer (BM3D, WNNM, MCWNNM, TWSC, NCSR, NLH), including sigma-zero paths,
sets `_NSSFatVersion=2`, `_NSSFatRadius`, `_NSSFatCenter`, `_NSSFatLayout=1`,
`_NSSModel`, `_NSSModelVersion=2`, and `_NSSNoiseProfile`. VAggregate rejects
missing/partial/conflicting identities and mixed models. `allow_legacy=1` permits
only wholly untagged external synthetic contributions whose destination semantics
the caller explicitly guarantees. It does not accept a conflicting version and
cannot repair an old wrong-axis producer. Final outputs remove fat layout tags.

Rolling consumes the same ordered destination contributions through a bounded
ring. For C output targets, producer-center support expands by 2r and source
support by 4r. Every dependency layer applies that rule again: a Basic→Final
reference graph has additional upstream expansion, rather than reusing the
single-stage source window. Random seeking never relies on an unpinned cache hit
observed during arInitial. Normal VS dependency scheduling still applies.

Zero sigma is an exact identity contribution only at the center's own target;
other slices have zero numerator and denominator. Unselected planes copy the
source. A target with no positive denominator copies its source sample. These are
intentional model policies, distinct from failed matching/solver status.

Real tests use weighted identity contributions and independent center/slice
impulses, first/last/single-frame clips, maximum radius, unselected planes,
reference clips, two-stage graphs, random repeated requests and concurrent
cache eviction. Fat-versus-rolling agreement is supplemental and is not used as
the sole correctness oracle. CPU and future GPU implementations may use different
physical layouts while preserving destination identity and contribution order.
