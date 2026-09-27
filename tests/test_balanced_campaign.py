#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Offline evidence-integrity checks for the balanced campaign (no VS/cloud)."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import balanced_campaign as campaign


class CampaignIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures = self.root/'fixtures'; self.fixtures.mkdir()
        self.case = dict(id='scene-s25-n0', image='scene', split='search', sigma=25,
                         noise_repeat=0, channels=1)
        campaign.save_json(self.fixtures/'fixtures.json', dict(cases=[self.case]))
        campaign.save_json(self.fixtures/'candidates.json', dict(algorithms={'NLM': [dict(id='default', parameters={})]}))
        campaign.save_json(self.fixtures/'nlh-v4.json', dict(profiles={}))
        self.plugin = self.root/'plugin.so'; self.plugin.write_bytes(b'fake plugin; never loaded')
        self.args = argparse.Namespace(fixtures=str(self.fixtures), plugin=str(self.plugin),
            out=str(self.root/'results'), split='search', sigmas=None, noise_repeat=None,
            candidates=None, finalists=None, algorithms=['NLM'], repeats=1, timeout=1.,
            twsc_timeout=1., target_seconds=.3, max_frames=256, candidate_ids=None,
            budget_seconds=10.)

    def invoke(self):
        with patch.object(campaign.os, 'sched_getaffinity', return_value={0}, create=True):
            with contextlib.redirect_stdout(io.StringIO()): campaign.run(self.args)

    def test_scene_leak_is_rejected(self):
        campaign.save_json(self.fixtures/'fixtures.json', dict(cases=[self.case,
            dict(self.case, id='same-scene-validation', split='validation')]))
        with self.assertRaisesRegex(ValueError, 'scene leaks'):
            campaign.selected_cases(self.args)

    def test_timeout_is_terminal_and_resume_identity_includes_v4(self):
        with patch.object(campaign.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['fake'], 1.)) as run:
            self.invoke(); self.assertEqual(run.call_count, 1)
        row = json.loads((Path(self.args.out)/'results.jsonl').read_text())
        self.assertEqual(row['status'], 'timeout'); self.assertFalse(row['ok'])
        with patch.object(campaign.subprocess, 'run') as run:
            self.invoke(); run.assert_not_called()
        campaign.save_json(self.fixtures/'nlh-v4.json', dict(profiles={'changed': {}}))
        with self.assertRaisesRegex(ValueError, 'resume identity differs'): self.invoke()

    def test_sealed_requires_frozen_selection(self):
        self.args.split = 'sealed'
        with self.assertRaisesRegex(ValueError, 'requires frozen'): self.invoke()

    def test_contaminated_timing_keeps_quality_but_no_speedup(self):
        out = Path(self.args.out); out.mkdir()
        campaign.save_json(out/'identity.json', dict(split='search'))
        base = dict(case='scene', algorithm='NLM', candidate='default', repeat=0, ok=True,
            status='complete', supplied_parameters={}, output_sha256='base', timing_eligible=False,
            seconds=.1, quality=dict(psnr_db=30., ssim=.9), noisy_quality=dict(psnr_db=20., ssim=.5),
            noisy_sha256='input', host=dict(boot_id='same'))
        cand = dict(base, candidate='tuned', supplied_parameters={'h': 15}, output_sha256='candidate',
                    seconds=.01, quality=dict(psnr_db=31., ssim=.91))
        (out/'results.jsonl').write_text('\n'.join(map(json.dumps,[base,cand]))+'\n')
        campaign.report(argparse.Namespace(results=str(out)))
        result = json.loads((out/'summary.json').read_text())['comparisons'][0]
        self.assertIsNone(result['speedup']); self.assertEqual(result['valid_pairs'], 0)
        self.assertEqual(result['psnr_delta_db'], 1.)

    def test_changed_parameters_across_repeats_are_rejected(self):
        out = Path(self.args.out); out.mkdir()
        campaign.save_json(out/'identity.json', {})
        base = dict(case='scene', algorithm='NLH', candidate='default', repeat=0, ok=True,
                    status='complete', supplied_parameters={}, output_sha256='base')
        candidate = dict(base, candidate='v4', supplied_parameters={'hard_strength': 1})
        changed = dict(candidate, repeat=1, supplied_parameters={'hard_strength': .5})
        (out/'results.jsonl').write_text('\n'.join(map(json.dumps,[base,candidate,changed]))+'\n')
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            campaign.report(argparse.Namespace(results=str(out)))


if __name__ == '__main__': unittest.main()
