#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Offline guards: a stress run must not pass by silently skipping coverage."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import test_vk_stress as stress


class HarnessTest(unittest.TestCase):
    def run_stress(self, *, status=b'no (API < 4.3)', devices=1,
                   probe=1, require=False, output=None, support=b'native'):
        self.renders = []

        class Source:
            def __init__(self):
                self.std = self

            def GPUUpload(self):
                return self

        class Core:
            # R78 has no vulkan_devices member; reading it is a regression.
            std = types.SimpleNamespace(LoadPlugin=lambda **kw: None)
            nss_vk = types.SimpleNamespace(
                Backend=lambda: dict(device_count=devices, support=support,
                                     probe_ok=probe, gpu_frames=status),
                **{name: (lambda clip, **kw: clip) for name in ('NLM', 'BM3D', 'PNLM')})

        core = Core()
        if stress.text(status).startswith('yes'):
            core.vulkan_devices = [{'usable': True}]
        vs = types.SimpleNamespace(core=core, Error=RuntimeError,
                                   GRAYS=1, YUV420PS=2, RGBS=3)
        modules = {'vapoursynth': vs, 'numpy': types.ModuleType('numpy'),
                   'test_vk_gpu_frames': types.SimpleNamespace(make_clip=lambda *a, **kw: Source())}

        def render(node, gpu, np):
            self.renders.append(gpu)
            return output(gpu) if output else ['same']

        argv = ['--plugin', 'unused', '--iterations', '2']
        if require:
            argv.append('--require-gpu')
        with patch.dict(sys.modules, modules), patch.object(stress, 'render', render), \
                contextlib.redirect_stdout(io.StringIO()):
            return stress.main(argv)

    def test_pre43_still_renders_cpu(self):
        self.assertEqual(self.run_stress(), 0)
        self.assertEqual(self.renders, [False] * 30)

    def test_bytes_status_exercises_gpu(self):
        self.assertEqual(self.run_stress(status=b'yes (R81)', require=True), 0)
        self.assertEqual(self.renders.count(True), 30)
        self.assertEqual(self.renders.count(False), 30)

    def test_required_gpu_cannot_pass_on_cpu_only(self):
        self.assertEqual(self.run_stress(require=True), 1)
        self.assertEqual(self.renders, [])

    def test_no_device_skip_is_not_release_admission(self):
        self.assertEqual(self.run_stress(devices=0), 77)
        self.assertEqual(self.run_stress(devices=0, require=True), 1)
        self.assertEqual(self.run_stress(support=b'unsupported'), 77)

    def test_probe_failure_is_not_skip(self):
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            self.run_stress(probe=0)

    def test_gpu_output_mismatch_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'outputs differ'):
            self.run_stress(status='yes (R81)', output=lambda gpu: [str(gpu)])

    def test_empty_soak_rejected(self):
        for value in ('0', '-1'):
            with self.assertRaises(stress.argparse.ArgumentTypeError):
                stress.positive(value)


if __name__ == '__main__':
    unittest.main()
