#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Run the review's plugin/ISA gates and reject missing or unexecuted results.

This is admission for these named gates only, not the full release matrix.
Every invocation retains JUnit, logs and a decision in a fresh output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from gate_evidence import GateEvidence


NATIVE = ('common', 'mcwnnm', 'batch', 'twsc_svd_validation')


def required_tests(isas):
    return ['test_full_image_plugin', *[f'test_{name}_{isa}' for isa in isas for name in NATIVE]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_results(junit, required, invocation_id, plugin_sha256, expected_cases=84):
    cases = ET.parse(junit).getroot().findall('.//testcase')
    selected = []
    plugin_report = None
    for name in required:
        matches = [case for case in cases if case.get('name') == name]
        if len(matches) != 1:
            raise ValueError(f'{name}: expected exactly one result, got {len(matches)}')
        case = matches[0]
        if case.get('status') != 'run' or any(case.find(tag) is not None for tag in ('skipped', 'failure', 'error')):
            raise ValueError(f'{name}: required gate did not execute successfully')
        output = case.findtext('system-out', '')
        if name == 'test_full_image_plugin':
            records = [json.loads(line.split(' ', 1)[1]) for line in output.splitlines()
                       if line.startswith('NSS_GATE_COMPLETED ')]
            if len(records) != 1:
                raise ValueError('plugin gate: missing or ambiguous completed-run record')
            record = records[0]
            if record.get('invocation_id') != invocation_id or record.get('passed') is not True:
                raise ValueError('plugin gate: stale invocation or unsuccessful completion')
            path = Path(record['summary_path'])
            if not path.is_absolute() or path.parent.name != record['run_id']:
                raise ValueError('plugin gate: invalid report identity')
            report = json.loads(path.read_text())
            if (report.get('completed') is not True or report.get('passed') is not True or
                report.get('plugin_sha256') != plugin_sha256 or
                any(report.get(key) != record.get(key) for key in ('run_id', 'invocation_id', 'summary_path'))):
                raise ValueError('plugin gate: incomplete, stale, failed, or wrong-binary summary')
            rows = report.get('cases', [])
            if (len(rows) != expected_cases or any(row.get('passed') is not True for row in rows) or
                len({row['name'] for row in rows}) != expected_cases):
                raise ValueError('plugin gate: missing, duplicate, or failed cases')
            plugin_report = str(path)
        else:
            isa = name.rsplit('_', 1)[1].upper()
            lanes = 8 if isa == 'AVX2' else 16
            if f'verified Highway target={isa} lanes={lanes}' not in output.splitlines():
                raise ValueError(f'{name}: requested target was not verified')
            if name.startswith('test_twsc_svd_validation_'):
                rows = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
                if len(rows) != 1 or rows[0].get('target') != isa or rows[0].get('available_8') is not True or rows[0].get('passed') is not True:
                    raise ValueError(f'{name}: missing expected group-eight admission')
        selected.append(dict(name=name, executed=True, passed=True))
    return dict(tests=selected, plugin_summary=plugin_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--plugin', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--ctest', default='ctest')
    parser.add_argument('--isa', action='append', choices=('avx2', 'avx3'),
                        help='Require only these ISAs on this host; default requires both')
    args = parser.parse_args()
    evidence = GateEvidence(args.out)
    required = required_tests(list(dict.fromkeys(args.isa or ('avx2', 'avx3'))))
    report = dict(passed=False, required=required, scope='named review gates only')
    env = dict(os.environ, NSS_GATE_INVOCATION_ID=evidence.invocation_id)
    prefix = [args.ctest, '--test-dir', str(args.build.resolve())]
    try:
        plugin_sha = digest(args.plugin)
        report['plugin_sha256'] = plugin_sha
        registry = subprocess.run(prefix + ['--show-only=json-v1'], capture_output=True, text=True, check=True)
        (evidence.path / 'registry.json').write_text(registry.stdout)
        registered = {test['name'] for test in json.loads(registry.stdout)['tests']}
        missing = set(required) - registered
        if missing:
            raise ValueError('required tests not registered: ' + ', '.join(sorted(missing)))
        junit = evidence.path / 'results.xml'
        command = prefix + ['-R', '^(' + '|'.join(map(re.escape, required)) + ')$',
                            '--no-tests=error', '--output-on-failure', '--output-junit', str(junit),
                            '--test-output-size-passed', '1048576', '--test-output-size-failed', '1048576']
        # No retries: a later success must not erase failed evidence.
        with (evidence.path / 'ctest.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        report['ctest_exit'] = result.returncode
        report.update(validate_results(junit, required, evidence.invocation_id, plugin_sha))
        if result.returncode != 0 or digest(args.plugin) != plugin_sha:
            raise ValueError('CTest failed or candidate binary changed during execution')
        report['passed'] = True
    except (OSError, ValueError, KeyError, ET.ParseError, subprocess.CalledProcessError) as error:
        report['error'] = str(error)
        print(f'Required review gates FAIL: {error}', file=sys.stderr)
    return 0 if evidence.finish(report) else 1


if __name__ == '__main__':
    raise SystemExit(main())
