# Matching contract, semantic version 2

The query is the fixed center-frame patch. Every temporal candidate's distance
is measured against that query, never against the neighbor frame's own patch at
the query coordinate. The genuine self match has distance zero and ordinal zero
and remains the first identity. Other candidates sort by finite distance first,
then temporal index, y, x and traversal ordinal. Nonfinite candidates rank after
finite candidates; they cannot displace a full finite top-k.

The center frame contributes its spatial search result. Each temporal direction
starts independently from the center's selected predictive seeds. For each next
frame, form the union of seed windows, visit distinct coordinates in raster order,
select the frame's seed count, and feed those candidates to the global group.
Overlap contributes one identity, and the two temporal directions do not share
mutable seed state. Only valid temporal slots participate.

The independent oracle enumerates a set of coordinates, computes scalar distances
and sorts with a test-owned comparator. Dyadic fixtures make distances exactly
representable: frame/coordinate identity and distance must match exactly across
actual production kernels. It covers Gray and multichannel data, independent
channel strides, short groups/windows, boundaries, overlaps and blocks
1/2/4/8/12/16. Existing nonfinite, spatial and separate-query-stride tests remain.
Near-tie numerical tests require a stated distance error band; non-near-tie
identity changes are failures, not generic cross-ISA tolerance.
