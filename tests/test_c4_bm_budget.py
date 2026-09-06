import types
import unittest
import json
from pathlib import Path
import subprocess
import tempfile
from contextlib import ExitStack
from unittest.mock import patch

import c4_paired_bm as bench


class BudgetTests(unittest.TestCase):
    def calibrate(self, wall, ms, quantum=0):
        row = dict(ms=ms, worker_wall_seconds=wall,
                   timed_frames=quantum or 1, rolling_chunk=quantum)
        with patch.object(bench, 'invoke_worker', return_value=row), patch.object(bench.time, 'monotonic', return_value=2):
            return bench.calibrate_group(types.SimpleNamespace(pairs=7), dict(frames=5000), 40)

    def test_fast_workload_is_bounded(self):
        config, pairs, _ = self.calibrate(.3, 10)
        self.assertEqual(pairs, 7)
        self.assertLessEqual(config['frames'], 128)
        self.assertGreater(config['frames'], 1)

    def test_rolling_keeps_whole_chunks(self):
        config, pairs, _ = self.calibrate(2, 200, 8)
        self.assertEqual(config['frames'] % 8, 0)
        self.assertGreaterEqual(config['frames'], 8)
        self.assertEqual(pairs, 7)

    def test_slow_group_reduces_pairs(self):
        _, pairs, _ = self.calibrate(5, 2000)
        self.assertEqual(pairs, 3)

    def test_impossible_group_skips(self):
        _, pairs, _ = self.calibrate(20, 10000)
        self.assertEqual(pairs, 0)

    def test_random_calibration_spans_multiple_chunks(self):
        row = dict(ms=20, worker_wall_seconds=1, timed_frames=32, rolling_chunk=8)
        config = dict(access='random', kwargs=dict(temporal_mode='rolling', rolling_chunk=8))
        with patch.object(bench, 'invoke_worker', return_value=row) as call, patch.object(bench.time, 'monotonic', return_value=2):
            effective, _, _ = bench.calibrate_group(types.SimpleNamespace(pairs=7), config, 40)
        self.assertEqual(call.call_args_list[0].args[2]['frames'], 32)
        self.assertGreaterEqual(effective['frames'], 32)
        self.assertEqual(effective['frames'] % 8, 0)

    def test_incomplete_sampling_never_passes(self):
        import numpy as np
        temporary_directory = tempfile.TemporaryDirectory
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                config = dict(name='chain', stage='two_stage', frames=5000)
                (root/'configs.json').write_text(json.dumps([config]))
                (root/'plugin.so').write_bytes(b'fixture')
                args = types.SimpleNamespace(out=str(root/'out'), configs=str(root/'configs.json'),
                    baseline=str(root/'plugin.so'), candidate=str(root/'plugin.so'), pairs=7,
                    group_seconds=40, extend=True, regression_only=False, semantic_change=False,
                    selection_threshold=1.02)
                stack.enter_context(patch.object(bench.os, 'sched_setaffinity', create=True))
                stack.enter_context(patch.object(bench.os, 'sync'))
                stack.enter_context(patch.object(bench.time, 'sleep'))
                stack.enter_context(patch.object(bench, 'cpu_stat', return_value={}))
                stack.enter_context(patch.object(bench, 'environment_delta', return_value=dict(valid=True)))
                stack.enter_context(patch.object(bench.tempfile, 'TemporaryDirectory',
                                                side_effect=lambda **kw: temporary_directory(dir=directory)))
                stack.enter_context(patch.object(bench, 'calibrate_group', return_value=(dict(config, frames=2), 3, [])))
                def invoke(args, name, config, limit):
                    if timeout and name == 'candidate':
                        raise subprocess.TimeoutExpired('fixture', limit)
                    if '_dump' in config:
                        np.save(config['_dump'], np.zeros((1, 1, 8, 8), dtype=np.float32))
                    return dict(ms=2 if name=='baseline' else 1, sha256='same', timed_frames=2)
                stack.enter_context(patch.object(bench, 'invoke_worker', side_effect=invoke))
                stack.enter_context(patch('builtins.print'))
                bench.run(args)
                decision = json.loads((root/'out/decision.json').read_text())
                self.assertFalse(decision['passed'])
                selection = json.loads((root/'out/selection.json').read_text())
                self.assertEqual(selection['selected'], [] if timeout else ['chain'])
                self.assertEqual(selection['missing'], ['chain'] if timeout else [])
                rows = (root/'out/raw.jsonl').read_text().splitlines()
                self.assertEqual(len(rows), 0 if timeout else 6)


if __name__ == '__main__':
    unittest.main()
