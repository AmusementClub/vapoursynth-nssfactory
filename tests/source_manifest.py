#!/usr/bin/env python3
"""Write a reproducibility manifest for a source/build/candidate comparison.

The manifest deliberately records dirty state and never treats a release
snapshot as evidence for a different checkout. Optional plugin/input,
quality/performance and exception files are attached by hash.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def run(root: Path, *args: str) -> str:
    return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files(root: Path) -> list[Path]:
    # Include staged/committed and non-ignored working-tree files. A dirty
    # integration file must be part of the identity, not merely reported as a
    # status line beside an otherwise clean source digest.
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root)
    return [root / value for value in raw.decode().split('\0') if value]


def source_digest(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b'\0' + sha256(path).encode() + b'\n')
    return digest.hexdigest()


def cmake_cache(build: Path | None) -> dict[str, str]:
    if build is None:
        return {}
    path = build / 'CMakeCache.txt'
    if not path.exists():
        return {}
    result = {}
    pattern = re.compile(r'^([^:#]+):[^=]*=(.*)$')
    for line in path.read_text(errors='replace').splitlines():
        match = pattern.match(line)
        if match:
            result[match.group(1)] = match.group(2)
    keys = ('CMAKE_BUILD_TYPE', 'CMAKE_CXX_COMPILER', 'CMAKE_CXX_FLAGS',
            'NSS_HWY_TARGET_MODE', 'NSS_AVX2_DEFAULTS', 'NSS_AVX2_EXPERIMENT',
            'NSS_BM_EXPERIMENT', 'NSS_TWSC_MIDGROUP_BATCH', 'NSS_GEMM_MULTI_ACCUM',
            'NSS_GIT_DESCRIBE')
    return {key: result[key] for key in keys if key in result}


def artifact(path: str | None) -> dict | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f'artifact does not exist: {resolved}')
    return {'path': str(resolved), 'size': resolved.stat().st_size, 'sha256': sha256(resolved)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--build')
    parser.add_argument('--plugin')
    parser.add_argument('--input', action='append', default=[])
    parser.add_argument('--baseline-label', default='')
    parser.add_argument('--candidate-label', default='')
    parser.add_argument('--isa-symbol', action='append', default=[])
    parser.add_argument('--quality-json')
    parser.add_argument('--performance-json')
    parser.add_argument('--exceptions-json')
    args = parser.parse_args()

    root = Path(args.source_root).expanduser().resolve()
    files = tracked_files(root)
    status = run(root, 'git', 'status', '--porcelain=v1', '--untracked-files=all')
    try:
        commit = run(root, 'git', 'rev-parse', 'HEAD')
    except subprocess.CalledProcessError:
        commit = None
    manifest = {
        'schema': 'nssfactory.reproducibility-manifest.v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(root),
        'source_commit': commit,
        'source_dirty': bool(status),
        'source_status': status.splitlines(),
        'source_file_count': len(files),
        'source_tree_sha256': source_digest(files, root),
        'source_files': {path.relative_to(root).as_posix(): sha256(path) for path in files},
        'build': cmake_cache(Path(args.build).expanduser().resolve() if args.build else None),
        'isa_symbols': args.isa_symbol,
        'baseline_label': args.baseline_label,
        'candidate_label': args.candidate_label,
        'plugin': artifact(args.plugin),
        'inputs': [artifact(path) for path in args.input],
        'quality': artifact(args.quality_json),
        'performance': artifact(args.performance_json),
        'exceptions': artifact(args.exceptions_json),
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'out': str(out), 'source_tree_sha256': manifest['source_tree_sha256'],
                      'source_dirty': manifest['source_dirty']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
