#!/usr/bin/env python3
"""Capture actual BM3D/NCSR batch inputs and outputs in a private plugin."""
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
for name in ('group_trace.hpp', 'group_trace.cpp'):
    shutil.copy2(Path(__file__).with_name(name), args.out / 'src/cpu' / name)
path = args.out / 'src/cpu/batch.cpp'; before = text = path.read_text()
text = '#include "cpu/group_trace.hpp"\n' + text
for name in ('bm3d_filter_group_batch', 'ncsr_filter_group_batch', 'mcwnnm_filter_group_batch', 'twsc_pca_soft_batch'):
    anchor = 'int ' + name + '('
    if text.count(anchor) != 1: raise RuntimeError('ambiguous function: ' + name)
    text = text.replace(anchor, 'int ' + name + '_raw(')
text += '''
namespace nss {
int bm3d_filter_group_batch(Bm3dFilterBatchItem* items, int count) {
    return nss_trace_bm_groups(items, count, bm3d_filter_group_batch_raw);
}
int ncsr_filter_group_batch(NcsrFilterBatchItem* items, int count) {
    return nss_trace_ncsr_groups(items, count, ncsr_filter_group_batch_raw);
}
int mcwnnm_filter_group_batch(McwnnmFilterBatchItem* items, int count) {
    return nss_trace_mc_groups(items, count, mcwnnm_filter_group_batch_raw);
}
int twsc_pca_soft_batch(TwscPcaBatchItem* items, int count) {
    return nss_trace_twsc_groups(items, count, twsc_pca_soft_batch_raw);
}
}
'''
path.write_text(text)
batch_after = text
path = args.out / 'src/cpu/bm/dct8.cpp'
dct = path.read_text()
start = dct.index('void bm3d_filter_group(')
opening = dct.index('{', start)
closing = dct.index('\n}', opening)
body = dct[opening + 1:closing]
dct = '#include "cpu/group_trace.hpp"\n' + dct[:opening + 1] + '\n    nss_trace_bm_call(patches, lda, group, k, block, sigma, wiener, ref_patches, weight_out, work, [&] {\n' + body + '\n    });' + dct[closing:]
path.write_text(dct)
path = args.out / 'src/cpu/twsc/encode.cpp'
text = path.read_text()
start = text.index('int twsc_pca_soft(')
opening = text.index('{', start)
closing = text.index('\n}', opening)
body = text[opening + 1:closing]
text = '#include "cpu/group_trace.hpp"\n' + text[:opening + 1] + '\n    return nss_trace_twsc_call(group, m, n, lda, sigma, work, work_floats, col_sigma, col_w, row_w, [&] {\n' + body + '\n    });' + text[closing:]
path.write_text(text)
cmake = args.out / 'CMakeLists.txt'
cmake.write_text(cmake.read_text() + '\ntarget_sources(nss_cpu PRIVATE src/cpu/group_trace.cpp)\n')
(args.out / 'group-trace-manifest.json').write_text(json.dumps(dict(diagnostic_only=True,
    before_sha256=hashlib.sha256(before.encode()).hexdigest(), after_sha256=hashlib.sha256(batch_after.encode()).hexdigest()), indent=2))
