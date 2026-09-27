"""Offline measurement-integrity tests; no perf, VM, VS or cloud credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import campaign_profile as profile


def counters(mode='counters', running=100.):
    events = ['cycles:u', 'instructions:u', 'branches:u', 'branch-misses:u'] if mode == 'counters' else ['TOPDOWN.SLOTS']
    rows = [dict(event=event, **{'counter-value': '12345', 'event-runtime': 2000000000,
                                'pcnt-running': running}) for event in events]
    if mode != 'counters':
        rows[0].update({'metric-value': '35.5', 'metric-unit': '% tma_retiring'})
        rows.append({'metric-value': '20.5', 'metric-unit': '% tma_backend_bound'})
    return '\n'.join(json.dumps(row) for row in rows)


def observation():
    return dict(ms=10., timed_frames=16, timed_source_fills=0,
                sha256='a' * 64, input_sha256='b' * 64,
                pmu_boundary='frame_requests_only',
                timed_environment=dict(cpu0_steal_ticks=0, cpu1_steal_ticks=0,
                                       cpu1_idle=1., cpu1_total_ticks=200))


class CounterTests(unittest.TestCase):
    def test_json_metric_only_rows_and_running_threshold(self):
        got = profile.parse_counters(counters('TopdownL1', 90.), 'TopdownL1')
        self.assertEqual(got['minimum_counter_running_percent'], 90.)
        self.assertEqual(len(got['metrics']), 2)
        with self.assertRaises(ValueError):
            profile.parse_counters(counters('TopdownL1', 89.99), 'TopdownL1')

    def test_missing_coverage_is_not_assumed_complete(self):
        row = json.loads(counters('TopdownL1').splitlines()[0])
        del row['pcnt-running']
        with self.assertRaises(KeyError):
            profile.parse_counters(json.dumps(row), 'TopdownL1')

    def test_bad_counts_metrics_runtime_and_missing_instructions_fail(self):
        for field, value in [('counter-value', '<not counted>'), ('counter-value', 'nan'),
                             ('event-runtime', 0), ('pcnt-running', 101),
                             ('metric-value', 'NaN')]:
            with self.subTest(field=field, value=value):
                row = json.loads(counters('TopdownL1').splitlines()[0])
                row[field] = value
                with self.assertRaises(ValueError):
                    profile.parse_counters(json.dumps(row), 'TopdownL1')
        with self.assertRaises(ValueError):
            profile.parse_counters(counters().splitlines()[0], 'counters')
        with self.assertRaises(ValueError):
            profile.parse_counters('', 'TopdownL2')


class WindowTests(unittest.TestCase):
    def test_measured_window_integrity(self):
        self.assertTrue(profile.validate_window(observation(), True)['valid'])
        for field, value in [('cpu0_steal_ticks', 1), ('cpu1_steal_ticks', 1),
                             ('cpu1_idle', .989), ('cpu1_idle', None),
                             ('cpu1_idle', float('nan')), ('cpu1_total_ticks', 0)]:
            with self.subTest(field=field, value=value):
                row = observation()
                row['timed_environment'][field] = value
                with self.assertRaises(ValueError):
                    profile.validate_window(row, True)

    def test_source_and_boundary_fail_even_with_quiet_cpu(self):
        for field, value in [('timed_source_fills', 1), ('pmu_boundary', 'whole_process'),
                             ('ms', 0), ('timed_frames', 0), ('sha256', '')]:
            with self.subTest(field=field):
                row = observation()
                row[field] = value
                with self.assertRaises(ValueError):
                    profile.validate_window(row, True)
        row = observation()
        del row['pmu_boundary']
        self.assertTrue(profile.validate_window(row, False)['valid'])

    def test_topology_requires_real_reciprocal_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cpu in (0, 1):
                p = root / f'cpu{cpu}/topology/thread_siblings_list'
                p.parent.mkdir(parents=True)
                p.write_text('0-1\n')
            with patch.object(profile.os, 'sched_getaffinity', return_value={0, 1}, create=True):
                self.assertEqual(profile.verify_topology(root)['sibling_cpu'], 1)
                (root / 'cpu1/topology/thread_siblings_list').write_text('1,17\n')
                with self.assertRaises(RuntimeError):
                    profile.verify_topology(root)
            self.assertEqual(profile.cpulist('0,2-4'), {0, 2, 3, 4})


class TimeoutTests(unittest.TestCase):
    def test_calibration_and_reports_share_a_deadline(self):
        now = [100.]
        budget = profile.CaseBudget(10., clock=lambda: now[0])
        self.assertEqual(budget.remaining(180.), 10.)
        now[0] += 8
        self.assertEqual(budget.remaining(30.), 2.)
        now[0] += 2
        with self.assertRaises(TimeoutError):
            budget.remaining(30.)

    def test_timeout_preserves_both_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, '-c',
                       'import sys,time; print("partial",flush=True); print("diagnostic",file=sys.stderr,flush=True); time.sleep(10)']
            with self.assertRaises(subprocess.TimeoutExpired):
                profile.logged(command, root, 'probe', profile.CaseBudget(3.), .25)
            self.assertIn('partial', (root / 'probe.stdout').read_text())
            self.assertIn('diagnostic', (root / 'probe.stderr').read_text())
            self.assertTrue(json.loads((root / 'probe.timeout.json').read_text())['timed_out'])

    @unittest.skipUnless(os.name == 'posix', 'process group semantics require POSIX')
    def test_timeout_kills_descendant_not_only_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'descendant-survived'
            child = f'import time; from pathlib import Path; time.sleep(.7); Path({str(marker)!r}).write_text("bad")'
            parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); print("started",flush=True); time.sleep(10)'
            with self.assertRaises(subprocess.TimeoutExpired):
                profile.bounded([sys.executable, '-c', parent], .25)
            time.sleep(.8)
            self.assertFalse(marker.exists())


class SampleTests(unittest.TestCase):
    def test_nonempty_file_is_not_sufficient_sample_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'perf.data'
            data.write_bytes(b'header')
            profile.validate_samples('# Total Lost Samples: 0\n# Samples: 2K of event cycles:u', data)
            for report in ('# Samples: 2K', '# Total Lost Samples: 1\n# Samples: 2K',
                           '# Total Lost Samples: 0\n# Samples: 0'):
                with self.subTest(report=report), self.assertRaises(ValueError):
                    profile.validate_samples(report, data)
            data.write_bytes(b'')
            with self.assertRaises(ValueError):
                profile.validate_samples('# Total Lost Samples: 0\n# Samples: 2K', data)


class CompositionTests(unittest.TestCase):
    def run_fake(self, root, tamper=None):
        seen = []
        def logged(command, directory, label, budget, limit):
            seen.append((label, command))
            if label in ('hot', 'callers'):
                return subprocess.CompletedProcess(command, 0, '# Total Lost Samples: 0\n# Samples: 100 of event cycles:u', '')
            row = observation()
            if label in ('TopdownL1', 'TopdownL2', 'counters'):
                (directory / (label + '.stat.jsonl')).write_text(counters(label))
            if label == 'record':
                (directory / 'perf.data').write_bytes(b'samples')
            if label == 'wall':
                row.pop('pmu_boundary')
            if tamper:
                tamper(label, row)
            return subprocess.CompletedProcess(command, 0, json.dumps(row), '')
        args = SimpleNamespace(case_seconds=10., timeout=5., minimum_running=90., plugin=Path('/fixture/libnss.so'))
        with patch.object(profile, 'logged', side_effect=logged):
            row = profile.run_case(args, dict(name='fixture', algorithm='nlh', kwargs={}, size=[16,16]), root)
        return row, seen

    def test_wall_is_uninstrumented_and_callers_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, seen = self.run_fake(root)
            self.assertTrue(row['accepted'], row.get('error'))
            commands = dict(seen)
            self.assertNotIn('perf', commands['wall'])
            record = commands['record']
            self.assertEqual(record[record.index('-F') + 1], '199')
            self.assertEqual(record[record.index('--call-graph') + 1], 'dwarf,4096')
            self.assertIn('--children', commands['callers'])
            self.assertIn('--no-children', commands['hot'])
            self.assertEqual(row['modes']['wall']['measurement_role'], 'uninstrumented_wall_timing')
            self.assertFalse((root / 'control.fifo').exists())

    def test_one_contaminated_window_or_hash_change_rejects_case(self):
        for tamper in ('busy', 'hash'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                def change(label, row):
                    if label == 'TopdownL2':
                        if tamper == 'busy':
                            row['timed_environment']['cpu1_idle'] = .5
                        else:
                            row['sha256'] = 'c' * 64
                root = Path(tmp)
                row, _ = self.run_fake(root, change)
                self.assertFalse(row['accepted'])
                self.assertFalse(row['completed'])
                self.assertTrue((root / 'TopdownL2.json').exists())
                self.assertTrue((root / 'result.json').exists())
                self.assertFalse((root / 'ack.fifo').exists())

    def test_nonfinite_worker_record_does_not_break_failure_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            def change(label, row):
                if label == 'wall':
                    row['ms'] = float('nan')
            root = Path(tmp)
            row, _ = self.run_fake(root, change)
            self.assertFalse(row['accepted'])
            self.assertTrue((root / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
