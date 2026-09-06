#!/usr/bin/env python3
"""Collect bounded, frame-gated Topdown L1/L2/L3 and symbols after paired runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from c4_paired_bm import cpu_stat, environment_delta


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).with_name('c4_integration_pmu.py')
    source = args.summary or args.configs
    rows = json.loads(source.read_text())
    if args.configs:
        rows = [dict(config=c) for c in rows]
    selected = [r for r in rows if r['config']['name'].endswith('_1080') or
                r['config']['name'] in ('bm3d_g32_chain', 'bm3d_motion_r1')]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [args.baseline, args.candidate, source, driver, Path(__file__),
                          driver.with_name('c4_integration.py'), driver.with_name('c4_paired_bm.py'),
                          driver.with_name('bm_numerics.py')]}
    (out / 'inputs.json').write_text(json.dumps(manifest, indent=2))
    results = []
    for row in selected:
        config = dict(row['config'])
        # Same count for both variants. Aim for two seconds of baseline work;
        # expensive filters retain the irreducible one-frame diagnostic.
        measured = row.get('budget', {}).get('calibration')
        if measured is None:
            measured = []
            for plugin in (args.baseline, args.candidate):
                command = ['taskset', '-c', '0', sys.executable, str(driver.with_name('c4_integration.py')),
                           'worker', '--plugin', str(plugin), '--config', json.dumps(dict(config, frames=1))]
                measured.append(json.loads(subprocess.check_output(command, text=True, timeout=60)))
            (out / (config['name'] + '-calibration.json')).write_text(json.dumps(measured, indent=2))
        per_frame = max(r['ms'] for r in measured) / 1000
        config['frames'] = max(1, min(16, int(2 / max(per_frame, .001))))
        if config.get('access') == 'random':
            config['frames'] = max(config['frames'], 4 * config['kwargs'].get('rolling_chunk', 4))
        for label, plugin in (('baseline', args.baseline), ('candidate', args.candidate)):
            directory = out / config['name'] / label
            directory.mkdir(parents=True)
            ctl, ack = directory / 'control.fifo', directory / 'ack.fifo'
            os.mkfifo(ctl)
            os.mkfifo(ack)
            command = [sys.executable, str(driver), '--plugin', str(plugin), '--config', json.dumps(config),
                       '--control', str(ctl), '--ack', str(ack)]
            control = ['--delay=-1', '--control=fifo:' + str(ctl) + ',' + str(ack)]
            before = cpu_stat()
            start = time.monotonic()
            try:
                for level in ('TopdownL1', 'TopdownL2', 'TopdownL3', 'counters', 'record'):
                    if level == 'record':
                        perf = ['perf', 'record', '-q', *control, '-e', 'cycles:u', '-F', '997',
                                '-o', str(directory / 'perf.data')]
                    else:
                        events = ['-e', 'cycles:u,instructions:u,branches:u,branch-misses:u'] if level == 'counters' else ['-M', level]
                        perf = ['perf', 'stat', *control, *events, '-o', str(directory / (level + '.txt'))]
                    proc = subprocess.run(['taskset', '-c', '0', *perf, '--', *command],
                                          capture_output=True, text=True, timeout=60)
                    (directory / (level + '.stderr')).write_text(proc.stderr)
                    (directory / (level + '.json')).write_text(proc.stdout)
                    proc.check_returncode()
                    value = json.loads(proc.stdout)
                    assert value['pmu_boundary'] == 'frame_requests_only'
                with (directory / 'hot.txt').open('w') as output:
                    subprocess.run(['perf', 'report', '--stdio', '--no-children', '--percent-limit', '.25',
                                    '--sort', 'symbol,dso', '-i', str(directory / 'perf.data')],
                                   stdout=output, stderr=subprocess.STDOUT, check=True)
                assert (directory / 'perf.data').stat().st_size > 0
                assert 'Total Lost Samples: 0' in (directory / 'hot.txt').read_text()
                for level in ('TopdownL1', 'TopdownL2', 'TopdownL3', 'counters'):
                    text = (directory / (level + '.txt')).read_text()
                    assert 'not counted' not in text and 'not supported' not in text, level
                results.append(dict(name=config['name'], variant=label, config=config,
                                    seconds=time.monotonic() - start,
                                    environment=environment_delta(before, cpu_stat())))
                (out / 'summary.json').write_text(json.dumps(results, indent=2))
                print(json.dumps(results[-1]), flush=True)
            finally:
                ctl.unlink()
                ack.unlink()
    for path, digest in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    (out / 'complete.json').write_text(json.dumps(dict(completed=True, profiles=len(results))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--summary', type=Path)
    source.add_argument('--configs', type=Path)
    run(parser.parse_args())
