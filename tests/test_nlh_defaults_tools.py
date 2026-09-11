#!/usr/bin/env python3
"""Contracts that guard the NLH experiment against leakage and invalid ranking."""
import copy
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from alignment_reference_nlh_fast import denoise,stage_parameters,query_indices
from alignment_reference import spatial_reference
from nlh_defaults_inputs import validate_manifest,selected_div2k,OLD_IMAGES,load_case,file_sha
from nlh_defaults_search import nondominated,shortlist,strength_variants,baseline,evaluation_cases,fitted_structures
from nlh_defaults_rank import summarize
from nlh_defaults_response import response
from nlh_defaults_decision import paired_interval
from nlh_defaults_evaluate import cohort_cases
from nlh_defaults_compare import compare


class ExperimentTests(unittest.TestCase):
    def test_quality_comparison_rejects_changed_inputs_and_records_subsets(self):
        record = dict(case='case', image='image', dataset='DIV2K', sigma=25,
                      sigma_mode='explicit', shape=[1, 32, 32], score_size=16,
                      input_sha256='input', output_sha256='output', seconds=1.,
                      quality=dict(psnr_db=30., ssim=.8))
        candidate = dict(a=record, b=dict(record, case='second'))
        reference = dict(a=record)
        with self.assertRaises(ValueError):
            compare(candidate, reference)
        summary, rows = compare(candidate, reference, allow_subset=True)
        self.assertEqual(summary['omitted_candidate_cases'], ['b'])
        self.assertEqual(rows[0]['psnr_delta'], 0.)
        candidate['a'] = dict(record, input_sha256='different')
        with self.assertRaises(ValueError):
            compare(candidate, reference, allow_subset=True)

    def test_controls_never_enter_selection_or_test(self):
        cases = [dict(id='old', split='development', control=True, format='gray', sigma=25),
                 dict(id='new', split='development', control=False, format='gray', sigma=25)]
        manifest = dict(cases=cases)
        self.assertEqual([c['id'] for c in cohort_cases(manifest, 'gray-low', 'development')], ['new'])
        self.assertEqual([c['id'] for c in cohort_cases(manifest, 'gray-low', 'development', True)], ['old'])
        for split in ('selection', 'test'):
            with self.assertRaises(ValueError):
                cohort_cases(manifest, 'gray-low', split, True)

    def test_paired_intervals_preserve_originals_and_input_identity(self):
        rows, ids = {}, [[], []]
        for original in ('a', 'b', 'c'):
            for sigma in (5, 75):
                for seed in (0, 1):
                    case = f'{original}-{sigma}-{seed}'
                    for variant in (0, 1):
                        key = f'{variant}-{case}'; ids[variant].append(key)
                        rows[key] = dict(case=case, image=original, dataset='DIV2K',
                                         sigma=sigma, sigma_mode='explicit', shape=[1, 32, 32],
                                         input_sha256=case, score_size=16,
                                         quality=dict(psnr_db=30+2*variant, ssim=.8+.05*variant),
                                         seconds=2. if variant == 0 else 1.)
        reference, candidate = [dict(cases=keys) for keys in ids]
        result = paired_interval(candidate, reference, rows, draws=64)
        self.assertEqual(result['originals'], dict(DIV2K=3))
        self.assertEqual(result['psnr_delta']['ci95'], [2., 2.])
        np.testing.assert_allclose(result['latency_ratio']['ci95'], [.5, .5])
        rows[ids[1][0]]['input_sha256'] = 'changed-input'
        with self.assertRaises(ValueError):
            paired_interval(candidate, reference, rows, draws=64)

    def test_paired_response_measures_output_noise_change(self):
        rng = np.random.default_rng(307)
        first, second = rng.normal(size=(2, 1, 8, 9))
        self.assertEqual(response(first, second, first, second)['ratio'], 1.)
        self.assertEqual(response(first, second, np.zeros_like(first), np.zeros_like(second))['ratio'], 0.)
        value = response(first, second, first*.25+7, second*.25+7, 4)
        self.assertAlmostEqual(value['ratio'], .25)
        with self.assertRaises(ValueError):
            response(first, first, first, second)

    def test_strength_fitting_precedes_latency_ranking(self):
        calibrated = baseline('gray-low')
        suppressed = dict(calibrated, hard_strength=8.)
        sparse = copy.deepcopy(calibrated); sparse['block_step'] = [8,8]
        rows = [dict(id='calibrated', parameters=calibrated, mean_psnr=31., mean_ssim=.9, geomean_seconds=1.),
                dict(id='suppressed', parameters=suppressed, mean_psnr=12., mean_ssim=.2, geomean_seconds=.2),
                dict(id='sparse', parameters=sparse, mean_psnr=30.8, mean_ssim=.88, geomean_seconds=.4)]
        self.assertEqual({r['id'] for r in fitted_structures(rows)}, {'calibrated','sparse'})
        self.assertEqual(shortlist(rows)['speed']['id'], 'sparse')
        self.assertEqual(shortlist(rows)['balanced']['id'], 'sparse')

    def test_blind_mode_keeps_identical_injected_noise(self):
        case = dict(id='synthetic', image='image', dataset='DIV2K', format='rgb',
                    sigma=25, seed=1234)
        blind = evaluation_cases([case], 'blind')[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = (np.arange(32*33*3).reshape(32,33,3) % 256).astype(np.uint8)
            Image.fromarray(pixels).save(root/'clean.png')
            manifest = dict(images=[dict(id='image', clean='clean.png', clean_sha256=file_sha(root/'clean.png'))])
            clean, noisy, crop = load_case(root, manifest, case, 16)
            blind_clean, blind_noisy, blind_crop = load_case(root, manifest, blind, 16)
        np.testing.assert_array_equal(clean, blind_clean)
        np.testing.assert_array_equal(noisy, blind_noisy)
        self.assertEqual(crop, blind_crop)
        self.assertNotEqual(case['id'], blind['id'])
        self.assertEqual(case['sigma'], blind['sigma'])
        self.assertTrue(blind['estimate_sigma'])
        self.assertNotIn('estimate_sigma', case)

    def test_dataset_and_noise_level_weights(self):
        def row(dataset, sigma, psnr, seconds):
            return dict(dataset=dataset, sigma=sigma, image=f'{dataset}-{sigma}',
                        quality=dict(psnr_db=psnr, ssim=.8), seconds=seconds)
        measurements = [row('DIV2K', 5, 40, 1), row('DIV2K', 25, 30, 4),
                        row('CC', None, 20, 8)]
        expected = summarize(measurements)
        repeated = summarize(measurements + [measurements[0]] * 10)
        self.assertEqual(expected['mean_psnr'], 27.5)
        self.assertEqual(expected['geomean_seconds'], 4.)
        self.assertEqual(expected['mean_psnr'], repeated['mean_psnr'])
        self.assertEqual(expected['geomean_seconds'], repeated['geomean_seconds'])

    def test_stage_shapes_and_sampling(self):
        rng=np.random.default_rng(204)
        image=rng.integers(16,200,(1,20,21)).astype(np.float32)/256
        for block in range(2,17):
            other=18-block
            p=dict(block_size=[block,other],block_step=[block,other],q=[min(4,block*block),2],
                   group_size=[64,8],search_window=[3,20],basic_iters=1)
            actual,logs=denoise(image,25,parameters=p)
            self.assertTrue(np.isfinite(actual).all())
            self.assertEqual(actual.shape,image.shape)
            self.assertEqual([r['parameters']['block_size'] for r in logs],[block,other])
        # Very small windows have different actual power-of-two group sizes.
        p=dict(block_size=3,block_step=3,q=2,group_size=8,search_window=3,basic_iters=1)
        _,logs=denoise(image,25,parameters=p)
        self.assertEqual(set(logs[-1]['actual_groups']),{4,8})

    def test_reference_defaults_preserved(self):
        rng=np.random.default_rng(17)
        for channels in (1,3):
            image=rng.integers(10,200,(channels,16,17)).astype(np.float32)/256
            actual,_=denoise(image,25)
            expected=spatial_reference(image,25,'NLH')
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-4)

    def test_zero_and_ties(self):
        image=np.full((1,19,20),.375,np.float32)
        exact,_=denoise(image,0,parameters=dict(block_size=[2,16],q=[2,16]))
        np.testing.assert_array_equal(exact,image)
        p=dict(block_size=[4,7],block_step=[4,7],q=[2,4],group_size=[8,16],basic_iters=1,
               wiener_sigma_scale=0)
        exact,_=denoise(image,25,parameters=p)
        np.testing.assert_allclose(exact,image,atol=1e-6,rtol=0)
        picks=query_indices(image,3,8,3,0,1,[0,2],[0])
        self.assertEqual(picks[0].tolist(),[0,1,18,19])
        self.assertEqual(picks[1][0],2)

    def test_reject_invalid_shapes(self):
        for p in (dict(block_size=1),dict(block_size=17),dict(q=3),dict(group_size=3),
                  dict(block_size=2,q=8),dict(block_step=9,block_size=8),dict(wiener_iters=0),
                  dict(lambda_basic=float('nan')),dict(block_size=[3,4,5])):
            with self.assertRaises(ValueError): stage_parameters(p,False,25)

    def test_split_leakage_and_old_images(self):
        self.assertEqual(len(selected_div2k()),30)
        self.assertFalse(set(selected_div2k()) & OLD_IMAGES)
        rows=[dict(id='a',group='scene',clean_sha256='sha-a',split='development'),
              dict(id='b',group='scene',clean_sha256='sha-b',split='test')]
        with self.assertRaises(ValueError): validate_manifest(dict(images=rows,cases=[]))
        rows[1].update(group='other',clean_sha256='sha-a')
        with self.assertRaises(ValueError): validate_manifest(dict(images=rows,cases=[]))
        rows[1].update(clean_sha256='sha-b',control=True)
        with self.assertRaises(ValueError): validate_manifest(dict(images=rows,cases=[]))

    def test_no_gain_gate_or_sigma_tuning(self):
        rows=[dict(id='a',mean_psnr=30.,mean_ssim=.9,geomean_seconds=1.),
              dict(id='b',mean_psnr=30.001,mean_ssim=.9,geomean_seconds=1.),
              dict(id='c',mean_psnr=29.99,mean_ssim=.89,geomean_seconds=.5)]
        self.assertEqual({r['id'] for r in nondominated(rows)},{'b','c'})
        self.assertEqual(shortlist(rows)['quality']['id'],'b')
        profile=baseline('rgb');before=copy.deepcopy(profile)
        for variant in strength_variants(profile):
            self.assertNotIn('sigma',variant)
            self.assertEqual(variant['block_size'],profile['block_size'])
        self.assertEqual(profile,before)


if __name__=='__main__':
    unittest.main()
