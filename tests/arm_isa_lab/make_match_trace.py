#!/usr/bin/env python3
"""Create a private plugin source with checked matcher capture/replay hooks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
shutil.copytree(args.source, args.out, ignore=shutil.ignore_patterns('.git', 'build*', 'artifacts', '__pycache__'))
here = Path(__file__).resolve().parent
for name in ('match_trace.hpp', 'match_trace.cpp'):
    shutil.copy2(here / name, args.out / 'src/cpu' / name)
changed = []

def edit(relative, change):
    path = args.out / relative
    old = path.read_text(); new = change(old)
    if old == new: raise RuntimeError('no instrumentation change: ' + relative)
    path.write_text(new)
    changed.append(dict(path=relative, before=hashlib.sha256(old.encode()).hexdigest(), after=hashlib.sha256(new.encode()).hexdigest()))

def once(s, old, new):
    if s.count(old) != 1: raise RuntimeError('source anchor not unique: ' + old[:70])
    return s.replace(old, new)

edit('src/cpu/bm/ssd.cpp', lambda s: '#include "cpu/match_trace.hpp"\n' + once(s,
    '    return HWY_DYNAMIC_DISPATCH(SpatialMatch)(ref, stride, width, height, bx, by, block, bm_range, group, out, avx2_features | NSS_AVX2_REQUESTED);',
    '    const int count = HWY_DYNAMIC_DISPATCH(SpatialMatch)(ref, stride, width, height, bx, by, block, bm_range, group, out, avx2_features | NSS_AVX2_REQUESTED);\n'
    '    return nss_trace_matches(1, bx, by, block, group, width, height, nss_trace_hash(ref, width, height, stride), out, count);'))

def temporal(s, nch=False):
    name = 'predictive_match_nch' if nch else 'predictive_match'
    s = once(s, 'int ' + name + '(', 'int ' + name + '_raw(')
    if nch:
        wrapper = '''
int predictive_match_nch(const float* const* refs, const int* strides, int nch, int ntemp, int width, int height,
                         int bx, int by, int t0, const SearchConfig& cfg, Match* out) {
    const int count = predictive_match_nch_raw(refs, strides, nch, ntemp, width, height, bx, by, t0, cfg, out);
    std::uint64_t hash = 0;
    for (int c = 0; c < nch; ++c) for (int t = 0; t < ntemp; ++t)
        hash = hash * 1099511628211ull ^ nss_trace_hash(refs[c*ntemp+t], width, height, strides[c]);
    return nss_trace_matches(4, bx, by, cfg.block, cfg.group, width, height, hash, out, count);
}
'''
    else:
        wrapper = '''
int predictive_match(const float* const* refs, const int* strides, int ntemp, int width, int height, int bx, int by,
                     int t0, const SearchConfig& cfg, Match* out, unsigned avx2_features) {
    const int count = predictive_match_raw(refs, strides, ntemp, width, height, bx, by, t0, cfg, out, avx2_features);
    std::uint64_t hash = 0;
    for (int t = 0; t < ntemp; ++t) hash = hash * 1099511628211ull ^ nss_trace_hash(refs[t], width, height, strides[t]);
    return nss_trace_matches(2, bx, by, cfg.block, cfg.group, width, height, hash, out, count);
}
'''
    pos = s.rindex('}  // namespace nss')
    return '#include "cpu/match_trace.hpp"\n' + s[:pos] + wrapper + s[pos:]

edit('src/cpu/bm/match.cpp', temporal)
def mch(s):
    s = once(s, '''    return HWY_DYNAMIC_DISPATCH(SpatialMatchNch)(refs, strides, nch, width, height, bx, by, block, bm_range, group,
                                                 out);''', '''    const int count = HWY_DYNAMIC_DISPATCH(SpatialMatchNch)(refs, strides, nch, width, height, bx, by, block, bm_range, group, out);
    std::uint64_t hash = 0;
    for (int c = 0; c < nch; ++c) hash = hash * 1099511628211ull ^ nss_trace_hash(refs[c], width, height, strides[c]);
    return nss_trace_matches(3, bx, by, block, group, width, height, hash, out, count);''')
    return temporal(s, True)
edit('src/cpu/mcwnnm/match_nch.cpp', mch)
edit('src/cpu/nlh/match16.cpp', lambda s: '#include "cpu/match_trace.hpp"\n' + once(s,
    '''    return HWY_DYNAMIC_DISPATCH(NlhSpatialMatch16Batch)(ref, stride, width, height, items, count, matches,
                                                        match_stride, counts);''',
    '''    const int status = HWY_DYNAMIC_DISPATCH(NlhSpatialMatch16Batch)(ref, stride, width, height, items, count, matches, match_stride, counts);
    const auto hash = nss_trace_hash(ref, width, height, stride);
    for (int i = 0; i < count; ++i)
        counts[i] = nss_trace_matches(5, items[i].bx, items[i].by, block, 16, width, height, hash, matches + i * match_stride, counts[i]);
    return status;'''))
edit('src/cpu/nlh/pixel_match.cpp', lambda s: '#include "cpu/match_trace.hpp"\n' + once(s,
    '    HWY_DYNAMIC_DISPATCH(PixelMatch)(group, m, n, lda, q, idx);',
    '    HWY_DYNAMIC_DISPATCH(PixelMatch)(group, m, n, lda, q, idx);\n'
    '    nss_trace_indices(m, n, q, nss_trace_hash(group, m, n, lda), idx);'))
edit('CMakeLists.txt', lambda s: s + '\ntarget_sources(nss_cpu PRIVATE src/cpu/match_trace.cpp)\n')
(args.out / 'match-trace-manifest.json').write_text(json.dumps(dict(changes=changed, diagnostic_only=True), indent=2))
