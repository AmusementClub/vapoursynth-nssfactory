#!/usr/bin/env python3
"""Bounded C4 L1/L2 and DWARF collection at the worker's frame boundary.

Wall timing is collected without perf. Counter/sample timings are diagnostics.
L3 is unavailable. Failed and timeout cases retain logs and partial records.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded(command, timeout):
    """Kill/reap the entire perf/worker group and retain timeout diagnostics."""
    if timeout <= 0:
        raise TimeoutError('case budget exhausted before launch')
    env = dict(os.environ, LC_ALL='C', DEBUGINFOD_URLS='')
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True, env=env)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException as error:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            error.output, error.stderr = stdout, stderr
        raise
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


class CaseBudget:
    def __init__(self, seconds, clock=time.monotonic):
        self.clock = clock
        self.deadline = clock() + seconds

    def remaining(self, limit):
        result = min(limit, self.deadline - self.clock())
        if result <= 0:
            raise TimeoutError('total case budget exhausted')
        return result


def logged(command, directory, label, budget, limit):
    """Share one deadline between calibration, modes and report generation."""
    timeout = budget.remaining(limit)
    save(directory / (label + '.command.json'), dict(command=command, timeout_seconds=timeout))
    try:
        proc = bounded(command, timeout)
    except subprocess.TimeoutExpired as error:
        (directory / (label + '.stdout')).write_text(error.output or '')
        (directory / (label + '.stderr')).write_text(error.stderr or '')
        save(directory / (label + '.timeout.json'), dict(timed_out=True, seconds=timeout))
        raise
    (directory / (label + '.stdout')).write_text(proc.stdout)
    (directory / (label + '.stderr')).write_text(proc.stderr)
    proc.check_returncode()
    return proc


def cpulist(text):
    result = set()
    for part in text.strip().split(','):
        if not re.fullmatch(r'\d+(?:-\d+)?', part):
            raise ValueError('invalid CPU topology list')
        ends = list(map(int, part.split('-')))
        if ends[0] > ends[-1]:
            raise ValueError('reversed CPU topology range')
        result.update(range(ends[0], ends[-1] + 1))
    return result


def verify_topology(root=Path('/sys/devices/system/cpu')):
    # The current worker measures CPU1. Refuse a different actual topology.
    observed = {}
    for cpu in (0, 1):
        text = (root / f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()
        if cpulist(text) != {0, 1}:
            raise RuntimeError(f'CPU{cpu} siblings {text!r}; worker requires reciprocal CPU0/CPU1 SMT')
        observed[f'cpu{cpu}_thread_siblings'] = text
    if 0 not in os.sched_getaffinity(0):
        raise RuntimeError('CPU0 is outside the available affinity mask')
    return dict(cpu=0, sibling_cpu=1, verified=True, **observed)


def parse_counters(text, mode, minimum_running=90.):
    rows = [json.loads(line.strip().rstrip(',')) for line in text.splitlines()
            if line.lstrip().startswith('{')]
    counted = [row for row in rows if 'counter-value' in row]
    if not counted:
        raise ValueError('no measured counters')
    events, running, metrics = {}, [], {}
    for row in rows:
        if 'metric-value' in row:
            value = float(row['metric-value'])
            if not math.isfinite(value):
                raise ValueError('nonfinite perf metric')
            metrics[row.get('metric-unit', '')] = value
    for row in counted:
        value = float(str(row['counter-value']).replace(',', ''))
        percent = float(row['pcnt-running'])  # Missing coverage is not 100%.
        runtime = float(row['event-runtime'])
        if not all(map(math.isfinite, (value, percent, runtime))) or value < 0 or runtime <= 0:
            raise ValueError('invalid counter value/runtime')
        if not minimum_running <= percent <= 100:
            raise ValueError('insufficient counter-running coverage')
        events[row['event']] = value
        running.append(percent)
    if not any(value > 0 for value in events.values()):
        raise ValueError('all counters are zero')
    if mode == 'counters':
        for event in ('cycles:u', 'instructions:u'):
            if events.get(event, 0) <= 0:
                raise ValueError('missing positive ' + event)
    elif not metrics:
        raise ValueError('Topdown produced no metrics')
    return dict(events=events, metrics=metrics, minimum_counter_running_percent=min(running),
                minimum_required_running_percent=minimum_running)


def validate_window(value, pmu):
    if pmu and value.get('pmu_boundary') != 'frame_requests_only':
        raise ValueError('PMU boundary is not frame requests')
    if value.get('timed_source_fills') != 0:
        raise ValueError('source was evaluated inside timing')
    if value.get('timed_frames', 0) <= 0 or not math.isfinite(value['ms']) or value['ms'] <= 0:
        raise ValueError('invalid measured frame time/count')
    if not re.fullmatch(r'[0-9a-f]{64}', value.get('sha256', '')):
        raise ValueError('missing output identity')
    timed = value['timed_environment']
    if timed['cpu0_steal_ticks'] != 0 or timed['cpu1_steal_ticks'] != 0:
        raise ValueError('steal inside measured window')
    idle = timed['cpu1_idle']
    if timed['cpu1_total_ticks'] <= 0 or idle is None or not math.isfinite(idle) or not .99 <= idle <= 1:
        raise ValueError('measured sibling is busy or interval is unmeasurable')
    return dict(valid=True, sibling_cpu=1, minimum_sibling_idle=.99,
                scope='worker frame window plus adjacent perf-control acknowledgements')


def validate_samples(report, data):
    if data.stat().st_size <= 0:
        raise ValueError('empty perf data')
    losses = re.findall(r'Total Lost Samples:\s*([\d,]+)', report)
    if not losses or any(int(value.replace(',', '')) != 0 for value in losses):
        raise ValueError('missing sample-loss evidence or lost samples')
    samples = re.findall(r'Samples:\s*([\d,.]+)\s*([KMG]?)', report, flags=re.IGNORECASE)
    if not samples or not any(float(value.replace(',', '')) > 0 for value, _ in samples):
        raise ValueError('no recorded samples')


def run_case(args, original, directory):
    config = dict(original)
    row = dict(name=config['name'], config=dict(config), completed=False, accepted=False,
               l3_status='unavailable_by_C4_protocol', modes={},
               sampling=dict(event='cycles:u', frequency_hz=199, callgraph='dwarf,4096'))
    start = time.monotonic()
    budget = CaseBudget(args.case_seconds)
    worker = Path(__file__).with_name('c4_integration.py')
    driver = worker.with_name('c4_integration_pmu.py')
    try:
        calibration = logged(['taskset', '-c', '0', sys.executable, str(worker), 'worker',
                              '--plugin', str(args.plugin), '--config', json.dumps(dict(config, frames=1))],
                             directory, 'calibration', budget, args.timeout)
        measured = json.loads(calibration.stdout)
        if not math.isfinite(measured['ms']) or measured['ms'] <= 0 or measured['timed_source_fills'] != 0:
            raise ValueError('invalid calibration')
        save(directory / 'calibration.json', measured)
        config['frames'] = max(1, min(16, int(2000 / max(measured['ms'], 1))))
        config['_cpu_environment'] = True
        config.pop('_frame_times', None)
        if config.get('motion'):
            config['frames'] = max(3, config['frames'])
        row['measured_config'] = config
        ctl, ack = directory / 'control.fifo', directory / 'ack.fifo'
        os.mkfifo(ctl)
        os.mkfifo(ack)
        command = [sys.executable, str(driver), '--plugin', str(args.plugin), '--config', json.dumps(config),
                   '--control', str(ctl), '--ack', str(ack)]
        control = ['--delay=-1', '--control=fifo:' + str(ctl) + ',' + str(ack)]
        for mode in ('wall', 'TopdownL1', 'TopdownL2', 'counters', 'record'):
            if mode == 'wall':
                invocation = ['taskset', '-c', '0', sys.executable, str(worker), 'worker',
                              '--plugin', str(args.plugin), '--config', json.dumps(config)]
            else:
                if mode == 'record':
                    perf = ['perf', 'record', '-q', *control, '-e', 'cycles:u', '-F', '199',
                            '--call-graph', 'dwarf,4096', '-o', str(directory / 'perf.data')]
                else:
                    events = ['-e', 'cycles:u,instructions:u,branches:u,branch-misses:u'] if mode == 'counters' else ['-M', mode]
                    perf = ['perf', 'stat', '-j', *control, *events,
                            '-o', str(directory / (mode + '.stat.jsonl'))]
                invocation = ['taskset', '-c', '0', *perf, '--', *command]
            proc = logged(invocation, directory, mode, budget, args.timeout)
            value = json.loads(proc.stdout)
            save(directory / (mode + '.json'), value)
            row['modes'][mode] = value
            value['window_validation'] = validate_window(value, pmu=mode != 'wall')
            value['measurement_role'] = 'uninstrumented_wall_timing' if mode == 'wall' else 'perf_diagnostic_not_wall_benchmark'
            if mode not in ('wall', 'record'):
                value['counters'] = parse_counters((directory / (mode + '.stat.jsonl')).read_text(),
                                                    mode, args.minimum_running)
            save(directory / (mode + '.json'), value)
        hashes = {value['sha256'] for value in row['modes'].values()}
        inputs = {value['input_sha256'] for value in row['modes'].values()}
        if len(hashes) != 1 or len(inputs) != 1:
            raise ValueError('input/output differs across measurement modes')
        for label, switches in (('hot', ['--no-children', '--call-graph', 'none']),
                                ('callers', ['--children', '--call-graph', 'graph,0.5,caller'])):
            report = logged(['perf', 'report', '--stdio', '--header', *switches, '--percent-limit', '.25',
                             '--sort', 'symbol,dso', '-i', str(directory / 'perf.data')],
                            directory, label, budget, min(30, args.timeout))
            (directory / (label + '.txt')).write_text(report.stdout)
            validate_samples(report.stdout, directory / 'perf.data')
        row['record_to_uninstrumented_wall_ratio'] = row['modes']['record']['ms'] / row['modes']['wall']['ms']
        row['overhead_interpretation'] = 'single-window diagnostic; sampling time is excluded from wall performance claims'
        budget.remaining(args.timeout)
        row['completed'] = row['accepted'] = True
    except Exception as error:
        row['error'] = repr(error)
    finally:
        for name in ('control.fifo', 'ack.fifo'):
            (directory / name).unlink(missing_ok=True)
        row['elapsed_seconds'] = time.monotonic() - start
        save(directory / 'result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--case-seconds', type=float, default=600)
    parser.add_argument('--minimum-running', type=float, default=90.)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout, args.case_seconds)) or not 0 < args.minimum_running <= 100:
        parser.error('timeouts and counter-running threshold must be positive and finite')
    args.plugin, args.configs, args.out = args.plugin.resolve(), args.configs.resolve(), args.out.resolve()
    configs = json.loads(args.configs.read_text())
    names = [config['name'] for config in configs]
    if not configs or len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
        parser.error('case names must be unique, nonempty safe directory names')
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'topology.json', verify_topology())
    driver = Path(__file__).with_name('c4_integration_pmu.py')
    dependencies = [args.plugin, args.configs, Path(__file__), driver, driver.with_name('c4_integration.py'),
                    driver.with_name('c4_paired_bm.py'), driver.with_name('bm_numerics.py')]
    dependencies.extend(Path(config['sample']).resolve() for config in configs)
    identities = {str(path): sha(path) for path in dependencies}
    save(args.out / 'inputs.json', identities)
    results = []
    for config in configs:
        directory = args.out / config['name']
        directory.mkdir()
        row = run_case(args, config, directory)
        results.append(row)
        save(args.out / 'summary.json', results)
        print(json.dumps({key: row[key] for key in ('name', 'completed', 'accepted', 'elapsed_seconds')}), flush=True)
    unchanged = all(sha(Path(path)) == digest for path, digest in identities.items())
    accepted = unchanged and all(row['accepted'] for row in results)
    save(args.out / 'complete.json', dict(completed=True, profiles=len(results),
                                        input_unchanged=unchanged, all_accepted=accepted))
    raise SystemExit(0 if accepted else 1)


if __name__ == '__main__':
    main()
