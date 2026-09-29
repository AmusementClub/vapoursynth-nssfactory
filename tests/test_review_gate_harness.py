#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Harness regressions only: mocked case results are never plugin evidence."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from gate_evidence import GateEvidence
from run_review_gates import required_tests, validate_results


class HarnessTests(unittest.TestCase):
    def test_actual_suite_pass_then_fail(self):
        # Extract the actual Suite so its initialization, run list, exception
        # handling and report writing execute without scientific/VS imports.
        source = Path(__file__).with_name('test_full_image_plugin.py')
        tree = ast.parse(source.read_text())
        suite = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Suite')
        core = SimpleNamespace(std=SimpleNamespace(LoadPlugin=lambda **kw: None))
        vs = SimpleNamespace(core=core, __version__='MOCK', GRAYS=1, RGBS=2, YUV444PS=3, YUV422PS=4, YUV420PS=5)
        namespace = dict(GateEvidence=GateEvidence, Path=Path, vs=vs, np=SimpleNamespace(__version__='MOCK'),
                         platform=platform, hashlib=hashlib, itertools=itertools, traceback=traceback, __file__=str(source))
        exec(compile(ast.Module(body=[suite], type_ignores=[]), str(source), 'exec'), namespace)
        cls = namespace['Suite']
        for method in ('parameters', 'twsc_fixed_geometry', 'invalid', 'extended', 'oracle', 'matrix', 'zero'):
            setattr(cls, method, lambda self, *args: None)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()) as output, patch.dict(os.environ, {}, clear=True):
            root = Path(root)
            plugin = root / 'mock-plugin'
            plugin.write_bytes(b'not a real plugin')
            out = (root / 'evidence').resolve()
            first = cls(out, plugin)
            self.assertTrue(first.run())
            historic = (first.out / 'summary.json').read_bytes()
            second = cls(out, plugin)
            def fail():
                raise AssertionError('intentional harness failure')
            second.parameters = fail
            self.assertFalse(second.run())
            self.assertNotEqual(first.out, second.out)
            self.assertEqual(first.out.parent, out)
            self.assertEqual(second.out.parent, out)
            self.assertFalse((out / 'summary.json').exists())
            self.assertEqual((first.out / 'summary.json').read_bytes(), historic)
            records = [json.loads(line.split(' ', 1)[1]) for line in output.getvalue().splitlines()
                       if line.startswith('NSS_GATE_COMPLETED ')]
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]['invocation_id'], records[1]['invocation_id'])
            self.assertEqual(records[1]['summary_path'], str(second.out / 'summary.json'))
            self.assertFalse(records[1]['passed'])
            self.assertEqual(len(json.loads((second.out / 'summary.json').read_text())['cases']), 84)

    def test_file_output_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'file'
            path.write_text('preserve')
            with self.assertRaisesRegex(ValueError, '--out must be a directory'):
                GateEvidence(path)
            self.assertEqual(path.read_text(), 'preserve')

    def test_legacy_summary_is_not_current(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            stale = Path(root) / 'summary.json'
            stale.write_text('{"passed":true}')
            run = GateEvidence(root)
            run.finish(dict(passed=False))
            self.assertEqual(stale.read_text(), '{"passed":true}')
            self.assertNotEqual(run.record['summary_path'], str(stale))

    def test_strict_results_reject_unexecuted_stale_and_wrong_binary(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            root = Path(root)
            run = GateEvidence(root / 'plugin')
            run.finish(dict(passed=True, plugin_sha256='candidate', cases=[dict(name='one', passed=True)]))
            summary_path = run.path / 'summary.json'
            good_summary = summary_path.read_text()
            required = required_tests(['avx2', 'avx3'])
            tree = ET.Element('testsuite')
            for name in required:
                case = ET.SubElement(tree, 'testcase', name=name, status='run')
                if name == 'test_full_image_plugin':
                    output = 'NSS_GATE_COMPLETED ' + json.dumps(dict(run.record, passed=True))
                else:
                    isa = name.rsplit('_', 1)[1].upper()
                    output = f'verified Highway target={isa} lanes={8 if isa == "AVX2" else 16}'
                    if 'twsc_svd_validation' in name:
                        output += '\n' + json.dumps(dict(passed=True, target=isa, available_8=True))
                ET.SubElement(case, 'system-out').text = output
            junit = root / 'test.xml'
            def validate():
                ET.ElementTree(tree).write(junit)
                return validate_results(junit, required, run.invocation_id, 'candidate', expected_cases=1)
            self.assertEqual(len(validate()['tests']), 9)
            first = tree[0]
            for tag in ('skipped', 'failure', 'error'):
                node = ET.SubElement(first, tag)
                with self.assertRaises(ValueError): validate()
                first.remove(node)
            first.set('status', 'notrun')
            with self.assertRaises(ValueError): validate()
            first.set('status', 'run')
            tree.remove(first)
            with self.assertRaises(ValueError): validate()
            tree.insert(0, first)
            for key, value in [('completed', False), ('passed', False), ('invocation_id', 'old'), ('plugin_sha256', 'wrong'), ('cases', [])]:
                report = json.loads(good_summary)
                report[key] = value
                summary_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError, msg=key): validate()
            summary_path.write_text(good_summary)
            native = tree[1].find('system-out')
            original = native.text
            native.text = ''
            with self.assertRaises(ValueError): validate()
            native.text = original
            twsc = tree[4].find('system-out')
            twsc.text = twsc.text.replace('"available_8": true', '"available_8": false')
            with self.assertRaises(ValueError): validate()


if __name__ == '__main__':
    unittest.main()
