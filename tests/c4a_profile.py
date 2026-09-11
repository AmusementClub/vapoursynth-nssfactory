#!/usr/bin/env python3
"""Native C4A frame-boundary PMU profile. Diagnostic, never a speedup gate.

Uses the existing integration worker: inputs/preloading/hashing are outside
timing, output hashes must agree between wall/stat/sample runs, CPU0 is fixed.
Architectural raw event codes follow GCP's Axion STANDARD event table. No Intel
Topdown metrics, cache-line traffic estimates or exclusive stall percentages
are inferred from these counts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from c4_paired_bm import cpu_stat

EVENTS = {
    "cycles": 0x11, "instructions": 0x8, "branches": 0x21, "branch_misses": 0x22,
    "l1d_refill": 0x3, "l2d_refill": 0x17, "frontend_stall": 0x23,
    "backend_stall": 0x24, "backend_memory_stall": 0x4005,
    "neon_spec": 0x8005, "sve_spec": 0x8006, "fp32_spec": 0x8018,
}
GROUPS = {
    "retired": ("cycles", "instructions", "branches", "branch_misses"),
    "cache": ("cycles", "instructions", "l1d_refill", "l2d_refill"),
    "stall": ("cycles", "instructions", "frontend_stall", "backend_stall", "backend_memory_stall"),
    "vector": ("cycles", "instructions", "neon_spec", "sve_spec", "fp32_spec"),
}


def event(name):
    return f"armv8_pmuv3_0/event=0x{EVENTS[name]:x}/u"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_stat(path, names):
    values = {}
    for row in csv.reader(path.read_text().splitlines(), delimiter=';'):
        if not row or row[0].startswith('#') or len(row) < 3:
            continue
        for name in names:
            if row[2].strip() == event(name):
                if '<' in row[0]:
                    raise RuntimeError(f"unavailable event {name}: {row}")
                count = float(row[0])
                running = float(row[4]) if len(row) > 4 and row[4] else None
                if running is None or running < 99.:
                    raise RuntimeError(f"multiplexed/incomplete event {name}: {row}")
                values[name] = dict(count=count, running_percent=running)
    if set(values) != set(names) or values['cycles']['count'] <= 0 or values['instructions']['count'] <= 0:
        raise RuntimeError(f"missing or empty counters: {path}: {values}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--wall-repeats', type=int, default=3)
    parser.add_argument('--fixed-frames', action='store_true',
                        help='use each config frames value for comparable cross-build PMU runs')
    args = parser.parse_args()
    if platform.machine() != 'aarch64' or args.seconds <= 0 or args.wall_repeats < 1:
        parser.error('requires native aarch64 and positive time/repeats')
    args.out.mkdir(parents=True, exist_ok=False)
    plugin = args.plugin.resolve()
    driver = Path(__file__).with_name('c4_integration.py').resolve()
    pmu_driver = driver.with_name('c4_integration_pmu.py')
    inputs = {str(p): digest(p) for p in (plugin, args.configs, Path(__file__), driver, pmu_driver,
                                         driver.with_name('c4_paired_bm.py'), driver.with_name('bm_numerics.py'))}
    report = dict(schema='nssfactory.c4a-profile.v1', passed=False, cpu=0,
                  performance_admission=False, inputs=inputs, event_codes=EVENTS, results=[],
                  frames_policy='config' if args.fixed_frames else 'calibrated_per_case')
    (args.out/'processes-before.txt').write_text(subprocess.check_output(['ps','-eo','pid,comm,pcpu,rss,args'],text=True))
    processes = subprocess.check_output(['ps', '-eo', 'comm='], text=True).splitlines()
    if any(p.strip() in ('cc1plus', 'cc1', 'ninja') for p in processes):
        raise RuntimeError('competing build process must finish before profiling')

    def execute(name, command, directory, timeout=180):
        before = cpu_stat()
        start = time.monotonic()
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        after = cpu_stat()
        (directory/(name+'.stdout')).write_text(proc.stdout)
        (directory/(name+'.stderr')).write_text(proc.stderr)
        (directory/(name+'.command.json')).write_text(json.dumps(command, indent=2))
        proc.check_returncode()
        value = json.loads(proc.stdout)
        if value['timed_source_fills'] != 0:
            raise RuntimeError('input construction entered timing')
        ticks = [b-a for a,b in zip(before['cpu1'], after['cpu1'])]
        value['environment'] = dict(cpu1_idle=ticks[3]/max(1,sum(ticks[:8])),
                                    cpu0_steal_ticks=after['cpu0'][7]-before['cpu0'][7],
                                    before=before, after=after)
        value['process_wall_seconds'] = time.monotonic()-start
        return value

    try:
        configs = json.loads(args.configs.read_text())
        for original in configs:
            directory = args.out/original['name']
            directory.mkdir()
            worker = lambda cfg: [sys.executable, str(driver), 'worker', '--plugin', str(plugin), '--config', json.dumps(cfg)]
            calibration = execute('calibration', ['taskset','-c','0',*worker(dict(original,frames=1))], directory)
            frames = original['frames'] if args.fixed_frames else max(1, min(16, math.ceil(args.seconds/(calibration['ms']/1000))))
            if not isinstance(frames, int) or frames < 1:
                raise RuntimeError('fixed frame counts must be positive integers')
            config = dict(original, frames=frames)
            row = dict(config=config, calibration=calibration, wall=[], counters={}, sampling={})
            report['results'].append(row)
            for repeat in range(args.wall_repeats):
                row['wall'].append(execute(f'wall-{repeat}', ['taskset','-c','0',*worker(config)], directory))
            row['median_ms_per_frame'] = statistics.median(r['ms'] for r in row['wall'])
            expected = row['wall'][0]['sha256']
            expected_input = row['wall'][0]['input_sha256']
            for label, names in [*GROUPS.items(), ('record', ())]:
                ctl, ack = directory/(label+'.ctl'), directory/(label+'.ack')
                os.mkfifo(ctl); os.mkfifo(ack)
                control = ['--delay=-1', f'--control=fifo:{ctl},{ack}']
                command = [sys.executable, str(pmu_driver), '--plugin', str(plugin),
                           '--config', json.dumps(config), '--control', str(ctl), '--ack', str(ack)]
                if label == 'record':
                    perf = ['perf','record','-q',*control,'-e','cycles:u','-F','997','-o',str(directory/'perf.data')]
                else:
                    perf = ['perf','stat',*control,'-x',';','--no-big-num','-e',
                            '{'+','.join(event(n) for n in names)+'}','-o',str(directory/(label+'.csv'))]
                try:
                    value = execute(label, ['taskset','-c','0',*perf,'--',*command], directory)
                finally:
                    ctl.unlink(); ack.unlink()
                if value['pmu_boundary'] != 'frame_requests_only' or value['sha256'] != expected or value['input_sha256'] != expected_input:
                    raise RuntimeError(f'PMU pass changed boundary/input/output: {label}')
                if label == 'record':
                    row['sampling']['worker'] = value
                else:
                    row['counters'][label] = dict(events=parse_stat(directory/(label+'.csv'),names),worker=value)
            if any(v['sha256'] != expected or v['input_sha256'] != expected_input for v in row['wall']):
                raise RuntimeError('wall-repeat outputs differ')
            commands = {
                'hot.txt': ['perf','report','--stdio','--header','--no-children','--percent-limit','0.2','--sort','symbol,dso','-i',str(directory/'perf.data')],
                'records.txt': ['perf','script','-D','-i',str(directory/'perf.data')],
            }
            for filename, command in commands.items():
                with (directory/filename).open('w') as output:
                    subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True)
            import re
            dump = (directory/'records.txt').read_text()
            samples = len(re.findall(r'PERF_RECORD_SAMPLE\b',dump))
            lost = len(re.findall(r'PERF_RECORD_LOST(?:_SAMPLES)?\b',dump))
            if samples < 20 or lost:
                raise RuntimeError(f'inadequate/lost samples: {samples}/{lost}')
            row['sampling'].update(samples=samples,lost=lost)
            row['passed'] = True
            (args.out/'summary.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(name=config['name'],ms=row['median_ms_per_frame'],frames=frames,samples=samples)),flush=True)
        if any(digest(Path(p)) != h for p,h in inputs.items()):
            raise RuntimeError('profile input changed')
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
