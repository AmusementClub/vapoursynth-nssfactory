#!/usr/bin/env python3
"""Create one isolated ARM experiment from a frozen source directory.

This does not edit the production checkout or grant performance admission.
The manifests make the exact single-hypothesis change reviewable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"source anchor is not unique: {old[:90]!r}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", choices=("svd4", "jacobi-split4", "gemm8x8", "gemm-row-panel", "sorted-spatial", "dct8-four"), required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be new")
    shutil.copytree(args.source, args.out, ignore=shutil.ignore_patterns(
        ".git", "build*", "artifacts", "__pycache__", ".omx", "ds"))
    changed = {"svd4": "src/cpu/wnnm/svd_batch8.cpp", "sorted-spatial": "src/cpu/bm/matcher.hpp",
               "dct8-four": "src/cpu/bm/dct8.cpp"}.get(args.candidate, "src/cpu/wnnm/jacobi8.cpp")
    path = args.out / changed
    original = text = path.read_text()
    if args.candidate == "svd4":
        text = once(text, "constexpr int kBatchLanes = 16;", """#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_WITHOUT_AES
constexpr int kBatchLanes = 4;
#else
constexpr int kBatchLanes = 16;
#endif""")
    elif args.candidate == "gemm8x8":
        text = once(text, "    if (N == 16) {", """    if (N == 16
#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_WITHOUT_AES
        || (N == 4 && n >= 64)
#endif
    ) {""")
    elif args.candidate == "gemm-row-panel":
        outer = '    for (; j0 + 4 <= n && mr <= m; j0 += 4) {'
        start = text.index(outer)
        inner = text.index('        for (int i0 = 0; i0 + mr <= m; i0 += mr) {', start)
        body_start = text.index('\n', inner) + 1
        body_end = text.index('\n        }\n        // The panel kernel', body_start)
        body = text[body_start:body_end]
        panel = '''#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_WITHOUT_AES
    // Keep one complete 16-row A panel hot across long LSSC products.
    // Each output still visits k in its original order. Incomplete row
    // panels retain the original implementation and all column tails.
    if (n >= 64 && m >= mr && m % mr == 0) {
        for (int i0 = 0; i0 < m; i0 += mr) {
            for (int panel_j = j0; panel_j + 4 <= n; panel_j += 4) {
                const int j0 = panel_j;
''' + body + '''
            }
        }
        j0 = (n / 4) * 4;
    }
#endif
'''
        text = once(text, outer, panel + outer)
    elif args.candidate == "sorted-spatial":
        text = once(text, "    CandidateTopK topk(out + 1, wanted - 1);", """#if defined(__aarch64__) || defined(_M_ARM64)
    SpatialSortedTopK topk(out + 1, wanted - 1);
#else
    CandidateTopK topk(out + 1, wanted - 1);
#endif""")
    elif args.candidate == "dct8-four":
        # Only the generic eight-point algebra becomes available at four
        # lanes; the FixedTag<8> patch loaders stay behind their old guard.
        start = text.index("#if HWY_MAX_BYTES >= 32\n", text.index("static const float* DctMatrix"))
        text = text[:start] + text[start:].replace("#if HWY_MAX_BYTES >= 32", "#if HWY_MAX_BYTES >= 16", 1)
        text = once(text, "using D8 = hn::FixedTag<float, 8>;", "#endif\n#if HWY_MAX_BYTES >= 32\nusing D8 = hn::FixedTag<float, 8>;")
        anchor = "#if HWY_MAX_BYTES >= 32\n        if (n == 8) {\n            Dct8Packed(d, x, y, inverse);"
        replacement = """#if HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_WITHOUT_AES
        if (n == 8) {
            Dct8Packed(d, x, y, inverse);
            for (int outb = 0; outb < n; ++outb) {
                for (int lane = 0; lane < lanes; ++lane)
                    base[(v0 + lane) * line_stride + outb * sample_stride] = y[outb * L + lane];
            }
            continue;
        }
#endif
""" + anchor
        text = once(text, anchor, replacement)
    else:
        begin = text.index("#else\n    for (int sweep = 0; sweep < 32; ++sweep)")
        end = text.index("\n#endif", begin)
        fallback = text[begin + len("#else\n"):end]
        split = once(fallback, "    for (int sweep", "    const hn::CappedTag<float, 4> d4;\n    for (int sweep")
        start = split.index("                float app = 0.f, aqq = 0.f, apq = 0.f;")
        stop = split.index("                if (app <= rank_floor", start)
        split = split[:start] + """                const auto p0 = hn::LoadU(d4, U8 + p * 8);
                const auto p1 = hn::LoadU(d4, U8 + p * 8 + 4);
                const auto q0 = hn::LoadU(d4, U8 + q * 8);
                const auto q1 = hn::LoadU(d4, U8 + q * 8 + 4);
                const float app = hn::ReduceSum(d4, hn::Add(hn::Mul(p0, p0), hn::Mul(p1, p1)));
                const float aqq = hn::ReduceSum(d4, hn::Add(hn::Mul(q0, q0), hn::Mul(q1, q1)));
                const float apq = hn::ReduceSum(d4, hn::Add(hn::Mul(p0, q0), hn::Mul(p1, q1)));
""" + split[stop:]
        start = split.index("                for (int i = 0; i < 8; ++i) {")
        stop = split.index("\n            }\n        }", start)
        split = split[:start] + """                const auto vcs = hn::Set(d4, cs), vsn = hn::Set(d4, sn);
                for (int i = 0; i < 8; i += 4) {
                    const auto up = hn::LoadU(d4, U8 + i + p * 8);
                    const auto uq = hn::LoadU(d4, U8 + i + q * 8);
                    hn::StoreU(hn::Sub(hn::Mul(vcs, up), hn::Mul(vsn, uq)), d4, U8 + i + p * 8);
                    hn::StoreU(hn::Add(hn::Mul(vsn, up), hn::Mul(vcs, uq)), d4, U8 + i + q * 8);
                    const auto vp = hn::LoadU(d4, V8 + i + p * 8);
                    const auto vq = hn::LoadU(d4, V8 + i + q * 8);
                    hn::StoreU(hn::Sub(hn::Mul(vcs, vp), hn::Mul(vsn, vq)), d4, V8 + i + p * 8);
                    hn::StoreU(hn::Add(hn::Mul(vsn, vp), hn::Mul(vcs, vq)), d4, V8 + i + q * 8);
                }""" + split[stop:]
        text = text[:begin] + "#elif HWY_TARGET == HWY_NEON || HWY_TARGET == HWY_NEON_WITHOUT_AES\n" + split + "\n#else\n" + fallback + text[end:]
    path.write_text(text)
    manifest = dict(candidate=args.candidate, changed_file=changed,
                    before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    after_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    hypothesis_only=True, performance_admission=False)
    (args.out / "arm-candidate.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
