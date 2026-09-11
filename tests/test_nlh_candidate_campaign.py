"""A noisy frame-count pilot must never become accepted timing evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from nlh_candidate_campaign import InvalidTiming, run_worker, main


class MeasurementRoleTests(unittest.TestCase):
    def test_parameter_comparison_uses_per_frame_times_and_separate_hashes(self):
        def worker(plugin, job, base, frames, mode=None, require_quiet=True):
            duration = 4. if plugin == 'old.so' else 1.
            return dict(seconds_per_frame=duration, seconds=duration*frames,
                        frames=frames, input_sha256='same-input',
                        host=dict(hostname='same-host', boot_id='same-boot'),
                        output_sha256='old-output' if plugin == 'old.so' else 'new-output')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root/'jobs.json'
            jobs.write_text(json.dumps(dict(jobs=[dict(label='fixture', input='pixels.npy',
                            parameters=dict(hard_strength=1), candidate_parameters=dict(hard_strength=2))])))
            arguments = ['campaign', 'bench', '--baseline', 'old.so', '--candidate', 'new.so',
                         '--jobs', str(jobs), '--out', str(root/'result'),
                         '--comparison', 'parameters', '--pairs', '2', '--target-seconds', '8']
            with patch('nlh_candidate_campaign.run_worker', worker), \
                 patch('nlh_candidate_campaign.os.sched_getaffinity', return_value={0}, create=True), \
                 patch('sys.argv', arguments):
                main()
            row = json.loads((root/'result/summary.json').read_text())['rows'][0]
        self.assertEqual(row['frames'], dict(baseline=2, candidate=8))
        self.assertEqual(row['median_speedup'], 4.)
        self.assertFalse(row['numerical_equivalence_required'])
        self.assertNotEqual(row['expected_output_sha256']['baseline'], row['expected_output_sha256']['candidate'])

    def test_pilot_and_measurement_have_different_acceptance_roles(self):
        record = dict(schema='nss.nlh-candidate-profile.v1', timed_source_fills=0,
                      affinity=[0], cpu_activity=dict(cpu0=dict(steal_ticks=0),
                      cpu1=dict(steal_ticks=0, busy_fraction=.025)))

        def execute(command, stdout, stderr):
            stdout.write(json.dumps(record)+'\n')
            stdout.flush()
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory, patch('nlh_candidate_campaign.subprocess.run', execute):
            root = Path(directory)
            job = dict(input='fixture.npy', parameters={})
            pilot = run_worker('plugin.so', job, root/'pilot', 1, require_quiet=False)
            self.assertFalse(pilot['cpu_valid'])
            self.assertFalse(pilot['accepted_for_timing'])
            with self.assertRaises(InvalidTiming):
                run_worker('plugin.so', job, root/'measured', 1)
            excluded = json.loads((root/'measured.json').read_text())
            self.assertFalse(excluded['accepted_for_timing'])
            self.assertIn('exclusion_reason', excluded)
            record['cpu_activity']['cpu1']['busy_fraction'] = 0.
            accepted = run_worker('plugin.so', job, root/'quiet', 1)
            self.assertTrue(accepted['accepted_for_timing'])


if __name__ == '__main__':
    unittest.main()
