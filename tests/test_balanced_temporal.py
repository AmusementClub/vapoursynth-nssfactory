#!/usr/bin/env python3
"""Offline guards for temporal source coverage and noise attribution."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from balanced_temporal import moving_input, residual_variation, temporal_layout


class TemporalInputChecks(unittest.TestCase):
    def test_composed_source_halo(self):
        self.assertEqual(temporal_layout('NLM', 'single', 2, 7)['interior_frames'], [2, 3, 4])
        for algorithm in ('BM3D', 'WNNM', 'MCWNNM', 'TWSC', 'NLH', 'NCSR'):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, 'full-halo'):
                    temporal_layout(algorithm, 'single', 2, 7)
                self.assertEqual(temporal_layout(algorithm, 'single', 2, 11)['interior_frames'], [4, 5, 6])
        with self.assertRaisesRegex(ValueError, 'full-halo'):
            temporal_layout('BM3D', 'basic-final', 2, 11)
        self.assertEqual(temporal_layout('BM3D', 'basic-final', 2, 19)['interior_frames'], [8, 9, 10])
        spatial = temporal_layout('LSSC', 'single', 2, 7)
        self.assertEqual((spatial['effective_radius'], spatial['source_halo']), (0, 0))

    def test_unequal_noise_cannot_be_relabelled_homogeneous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case, manifest in [({'channel_sigma': [5, 25, 50]}, {}),
                                   ({}, {'channel_sigma': [50, 25, 5]}),
                                   ({}, {'do_not_pool_with_homogeneous_noise': True})]:
                (root/'fixtures.json').write_text(json.dumps(manifest))
                # Reject before touching an absent clean image or its shape.
                with self.assertRaisesRegex(ValueError, 'homogeneous'):
                    moving_input(root, case, 7)

    def test_extended_sequence_has_exact_motion_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = np.arange(3*40*42, dtype=np.float32).reshape(3, 40, 42)/6000
            parent.tofile(root/'clean.f32')
            (root/'fixtures.json').write_text('{"cases": []}')
            case = dict(clean='clean.f32', channels=3, height=40, width=42,
                        sigma=25, seed=123, id='offline-homogeneous')
            clean, noisy, offsets, seed = moving_input(root, case, 11)
            second = moving_input(root, case, 11)
            self.assertEqual(clean.shape, (11, 3, 30, 32))
            np.testing.assert_array_equal(noisy, second[1])
            self.assertEqual(seed, second[3])
            self.assertFalse(np.array_equal(noisy[0]-clean[0], noisy[1]-clean[1]))
            variation = residual_variation(clean, noisy, clean, offsets)
            self.assertEqual(variation['output_temporal_std_rms'], 0)
            self.assertGreater(variation['noisy_temporal_std_rms'], 0)
            for frames in (5, 6, 8):
                with self.assertRaisesRegex(ValueError, 'odd source frames'):
                    moving_input(root, case, frames)


if __name__ == '__main__':
    unittest.main()
