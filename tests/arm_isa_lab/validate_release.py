#!/usr/bin/env python3
"""Verify the recorded Plan02 release decision and its local evidence identities."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--source-root', type=Path,
                        help='Explicit frozen release source to validate after the checkout advances; evidence stays under --root')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve() if args.source_root else root
    result = dict(passed=False, checks=[], validated_source_root=str(source_root),
                  validates_current_checkout=source_root == root,
                  scope='Recorded native release matrix for validated_source_root; no deployment or later-source claim.')

    def check(name, condition):
        result['checks'].append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise RuntimeError(name)

    def read(name):
        return json.loads((root / name).read_text())

    try:
        ready = read('plan02-readiness.json')
        check('release levels', all(ready.get(k) is True for k in ('n_functional', 'n_accelerated', 'n_release')))
        check('historical Plan01 failure retained', ready['plan01_no_regression_passed'] is False)
        check('no unresolved release gates', not ready['open_items'] and all(v['passed'] for v in ready['gates'].values()))
        production = ready['source_identity']['production_files']
        current = {str(p.relative_to(source_root)) for folder in ('src', 'include', 'cmake')
                   for p in (source_root / folder).rglob('*') if p.is_file() and p.suffix in ('.cpp', '.hpp', '.h', '.cmake')}
        current.add('CMakeLists.txt')
        check('complete production source set', current == set(production))
        check('production matches tested source', all(sha(source_root / p) == h for p, h in production.items()))
        for path, digest in ready['evidence_files'].items():
            check('evidence: ' + path, (root / path).is_file() and sha(root / path) == digest)
        for item in ready['required_reports']:
            value = read(item['path'])
            for key in item.get('field', ['passed']):
                value = value[key]
            check(item['name'], value == item.get('equals', True))
        bench = read('neon_bench_results.json')
        check('same-host performance gate', bench['passed'] and len(bench['platforms']) == 3)
        for platform, group in bench['platforms'].items():
            for row in group['results']:
                raw = read(row['report'])
                measured = next(v for v in raw['results'] if v['config']['name'] == row['name'])
                check(platform + ' ' + row['name'] + ' complete', raw['complete'] and len(measured['pairs']) >= 15)
                ratios = [p['baseline']['ms'] / p['candidate']['ms'] for p in measured['pairs']]
                check(platform + ' ' + row['name'] + ' paired median', abs(statistics.median(ratios) - row['speedup']) < 1e-12)
                check(platform + ' ' + row['name'] + ' no regression', measured['ci95'][0] >= 1 / 1.01)
                check(platform + ' ' + row['name'] + ' inputs', all(p['baseline']['input_sha256'] == p['candidate']['input_sha256'] and
                      all(p[s]['timed_source_fills'] == 0 for s in ('baseline', 'candidate')) for p in measured['pairs']))
                if platform != 'macos':
                    check(platform + ' ' + row['name'] + ' environment', measured['minimum_cpu1_idle'] >= .95 and measured['cpu0_no_steal'] and measured['scheduler_ticks_observed'])
                if platform == 'x86':
                    check('x86 ' + row['name'] + ' exact outputs', measured['exact'])
        for package in ready['packages']:
            manifest = read(package['manifest'])
            check(package['platform'] + ' installed package', manifest['passed'] and manifest['installed_binary_unchanged'])
            check(package['platform'] + ' measured binary', manifest['plugin_sha256'] == package['plugin_sha256'])
            check(package['platform'] + ' package digest', sha(root / package['archive']) == package['archive_sha256'])
        for resource in ready['cleanup']['records']:
            value = read(resource)
            check('resource cleanup ' + value['instance'], value['instance_absent'] and value['disk_absent'])
        check('both temporary VMs checked', len(ready['cleanup']['records']) == 2)
        result['passed'] = True
    except (OSError, ValueError, KeyError, StopIteration, RuntimeError) as error:
        result['error'] = str(error)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=result['passed'], checks=len(result['checks']), error=result.get('error'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
