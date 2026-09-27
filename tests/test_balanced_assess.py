#!/usr/bin/env python3
"""Offline selection-policy guards, including censored cold references."""
import unittest
from balanced_assess import summarize, metadata


def cell(**overrides):
    row=dict(case='scene-s5-n0',candidate='candidate',image='scene',ok=True,sigma=5,
        sentinel=True,quality=dict(psnr_db=35.,ssim=.94),baseline_quality=dict(psnr_db=35.,ssim=.94),
        noisy_quality=dict(psnr_db=34.,ssim=.9),paired_speed_ratios=[2.,2.,2.],valid_pairs=3,
        baseline_kind='paired-warm',seconds=.01,peak_bytes=1024,
        supplied_parameters={'sigma':5},output_sha256='stable')
    row.update(overrides);return row


class AssessmentPolicy(unittest.TestCase):
    def assess(self,items,**kwargs):return summarize(items,3,kwargs.get('ms'),kwargs.get('mb'))

    def test_measured_gate_needs_all_pairs(self):
        row=cell(paired_speed_ratios=[2.],valid_pairs=1)
        r=self.assess([row]);self.assertFalse(r['timing_complete'])
        self.assertNotIn('meets measured',r['recommendation'])

    def test_cold_quality_never_supplies_timing(self):
        r=self.assess([cell(baseline_kind='cold-quality-only',paired_speed_ratios=[],valid_pairs=0)])
        self.assertEqual(r['quality_only_baseline_cases'],1)
        self.assertTrue(r['quality_complete']);self.assertIsNone(r['median_speedup'])
        self.assertFalse(r['timing_complete'])

    def test_missing_reference_does_not_become_passing_loss_bound(self):
        r=self.assess([cell(baseline_quality=None,baseline_kind=None,paired_speed_ratios=[],valid_pairs=0)])
        self.assertFalse(r['quality_complete']);self.assertFalse(r['quality_bounds_pass'])
        self.assertIn('loss bound unavailable',r['recommendation'])

    def test_low_noise_degradation_overrides_gain_against_bad_default(self):
        row=cell(quality=dict(psnr_db=33.,ssim=.88),baseline_quality=dict(psnr_db=30.,ssim=.7))
        r=self.assess([row]);self.assertEqual(r['low_noise_worse_than_noisy_cases'],['scene-s5-n0'])
        self.assertFalse(r['no_worse_than_noisy']);self.assertIn('reject',r['recommendation'])

    def test_memory_limit_blocks_otherwise_passing_speed_recipe(self):
        r=self.assess([cell(peak_bytes=2*1024*1024)],mb=1)
        self.assertFalse(r['optional_resource_limits_pass'])
        self.assertNotIn('meets measured',r['recommendation'])

    def test_repeated_case_does_not_inflate_scene_or_case_count(self):
        r=self.assess([cell(),cell()])
        self.assertEqual(r['scheduled_cases'],1);self.assertEqual(r['scenes'],1)
        with self.assertRaisesRegex(ValueError,'differs across runs'):
            self.assess([cell(),cell(output_sha256='changed')])

    def test_quiet_rgb_channel_is_guarded_even_if_overall_psnr_improves(self):
        row=cell(sigma=25,channel_sigma=[5,25,50],
            quality=dict(psnr_db=35.,ssim=.94,planes=[{'psnr_db':32.},{'psnr_db':35.},{'psnr_db':35.}]),
            noisy_quality=dict(psnr_db=30.,ssim=.8,planes=[{'psnr_db':34.},{'psnr_db':30.},{'psnr_db':30.}]))
        r=self.assess([row]);self.assertFalse(r['no_worse_than_noisy'])
        self.assertEqual(r['low_noise_channel_degradations'][0]['plane'],0)
        self.assertIn('reject',r['recommendation'])

    def test_unequal_input_requires_matching_vector_and_separate_family(self):
        known={'case':dict(channels=3,sigma=25,channel_sigma=[5.,25.,50.])}
        row=dict(case='case',ok=True,shape=[3,256,256],sigma=25,
                 supplied_parameters={'sigma':[5.,25.,50.]})
        value=metadata(row,known)
        self.assertEqual(value['noise_family'],'unequal-rgb-5-25-50')
        row['supplied_parameters']['sigma']=25
        with self.assertRaisesRegex(ValueError,'differs from supplied'):
            metadata(row,known)

    def test_missing_baseline_cannot_hide_a_quiet_channel_failure(self):
        row=cell(sigma=25,channel_sigma=[5,25,50],baseline_quality=None,baseline_kind=None,
            paired_speed_ratios=[],valid_pairs=0,
            quality=dict(psnr_db=35.,ssim=.94,planes=[{'psnr_db':28.},{'psnr_db':35.},{'psnr_db':35.}]),
            noisy_quality=dict(psnr_db=25.,ssim=.8,planes=[{'psnr_db':34.},{'psnr_db':25.},{'psnr_db':20.}]))
        r=self.assess([row]);self.assertFalse(r['quality_complete'])
        self.assertFalse(r['no_worse_than_noisy']);self.assertIn('reject',r['recommendation'])
        self.assertNotIn('provisional effective',r['recommendation'])

    def test_psnr_gain_cannot_label_ssim_failure_a_quality_cost_choice(self):
        row=cell(sigma=25,quality=dict(psnr_db=35.5,ssim=.91))
        r=self.assess([row]);self.assertFalse(r['quality_bounds_pass'])
        self.assertTrue(r['no_worse_than_noisy']);self.assertIn('reject',r['recommendation'])


if __name__=='__main__':unittest.main()
