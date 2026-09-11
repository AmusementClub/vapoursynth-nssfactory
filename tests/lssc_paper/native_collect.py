#!/usr/bin/env python3
"""Freeze final source, exact library, test evidence and an artifact hash manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from native import default_library
from native_campaign import snapshot
from run import sha


def junit(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    failed = sum(any(child.tag in ('failure', 'error') for child in case) for case in cases)
    skipped = sum(any(child.tag == 'skipped' for child in case) for case in cases)
    return {'tests': len(cases), 'passed': len(cases) - failed - skipped, 'failed': failed, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--build', required=True)
    parser.add_argument('--asan', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    build, asan = Path(args.build).resolve(), Path(args.asan).resolve()
    verification = json.loads((root / 'verification-v1.json').read_text())
    cleanup = json.loads((root / 'c4-cleanup.json').read_text())
    if not verification['complete'] or not cleanup['instance_absent'] or not cleanup['disk_absent']:
        raise RuntimeError('evidence/cleanup is not complete')
    out = root / 'final-delivery'
    out.mkdir(exist_ok=False)
    record = snapshot(out, default_library().resolve())
    copies = {root / 'final-pytest.xml': 'pytest.xml', root / 'final-ctest.xml': 'ctest.xml',
              root / 'final-asan.xml': 'asan.xml', build / 'CMakeCache.txt': 'CMakeCache.txt',
              build / 'Testing/Temporary/LastTest.log': 'ctest.log',
              asan / 'Testing/Temporary/LastTest.log': 'asan.log'}
    for source, name in copies.items():
        shutil.copy2(source, out / name)
    record.update(schema='nss.native-delivery.v1', native_library_sha256=sha(out / record['library_file']),
                  verification_sha256=sha(root / 'verification-v1.json'),
                  verified_campaigns=verification['campaigns'], verified_benchmark_calls=verification['calls'],
                  c4_created=False, c4_instance_absent=True, c4_disk_absent=True,
                  pytest=junit(out / 'pytest.xml'), ctest=junit(out / 'ctest.xml'),
                  sanitizer=junit(out / 'asan.xml'), no_default_promotion=True)
    if any(record[name]['failed'] for name in ('pytest', 'ctest', 'sanitizer')):
        raise RuntimeError('failed test evidence')
    (out / 'delivery.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    manifest = root / 'SHA256SUMS'
    sidecar = root / 'SHA256SUMS.sha256'
    if manifest.exists() or sidecar.exists():
        raise FileExistsError('artifact manifest already exists')
    lines = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                value = hashlib.file_digest(stream, 'sha256').hexdigest()
            lines.append(f'{value}  {path.relative_to(root)}\n')
    manifest.write_text(''.join(lines))
    sidecar.write_text(f'{sha(manifest)}  {manifest.name}\n')
    print(json.dumps(record | dict(manifest_files=len(lines), manifest_sha256=sha(manifest)), indent=2))


if __name__ == '__main__':
    main()
