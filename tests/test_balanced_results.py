#!/usr/bin/env python3
"""Offline evidence guards; no plugin, fixture pixels, or cloud access."""
import json
from pathlib import Path
import tempfile
import unittest

from balanced_results import DEFAULTS_SHA, METRICS_SHA, coverage, formal_timing, recipe_coverage, sha, timing_eligible, verify_stage


def timed(candidate='candidate', repeat=0, ratio=2.):
    return dict(case='case',algorithm='NLM',candidate=candidate,repeat=repeat,status='complete',
        seconds=ratio if candidate=='default' else 1.,elapsed_seconds=.8,frames=8,
        host={'boot_id':'same'},noisy_sha256='noise',cpu0_thread_siblings='0-1',
        cpu_activity={'cpu0':dict(total_ticks=10,busy_fraction=1.,steal_ticks=0),
                      'cpu1':dict(total_ticks=10,busy_fraction=.01,steal_ticks=0)})


class ResultsPolicy(unittest.TestCase):
    def test_coverage_preserves_missing_timeout_and_duplicate(self):
        jobs=[dict(case='case',algorithm='NLM',candidate='candidate',repeat=i) for i in range(3)]
        stage=dict(jobs=jobs,temporal=False)
        rows=[timed(repeat=0),dict(timed(repeat=1),status='timeout')]
        result=coverage(stage,rows)
        self.assertFalse(result['all_successful'])
        self.assertEqual(result['status_counts'],{'complete':1,'timeout':1})
        self.assertEqual(result['missing'],[jobs[2]])
        self.assertTrue(coverage(stage,rows+[rows[0]])['duplicates'])

    def test_original_timing_boundary_accepts_subsecond_windows(self):
        row=timed();self.assertTrue(timing_eligible(row))
        row['cpu_activity']['cpu1']['busy_fraction']=.01001
        self.assertFalse(timing_eligible(row))
        row=timed();row['cpu_activity']['cpu1']['total_ticks']=9
        self.assertFalse(timing_eligible(row))
        row=timed();row['cpu_activity']['cpu0']['steal_ticks']=1
        self.assertFalse(timing_eligible(row))

    def test_seven_valid_pairs_ci_and_actual_windows(self):
        rows=[timed(c,n) for n in range(7) for c in ('default','candidate')]
        result=formal_timing(rows,7)[0]
        self.assertTrue(result['seven_complete_valid_pairs'])
        self.assertTrue(result['ci_lower_bound_above_one'])
        self.assertEqual(result['median_speed_ratio_ci95'],[2.,2.])
        self.assertEqual(result['actual_window_seconds']['default']['below_one_second'],7)
        rows[-1]['cpu_activity']['cpu1']['total_ticks']=9
        result=formal_timing(rows,7)[0]
        self.assertEqual(result['valid_pairs'],6)
        self.assertIsNone(result['median_speed_ratio_ci95'])
        self.assertFalse(result['ci_lower_bound_above_one'])

    def test_median_speed_does_not_override_uncertain_ci(self):
        rows=[timed(c,n,ratio) for n,ratio in enumerate([.8,.9,1.,2.,2.,2.,2.]) for c in ('default','candidate')]
        result=formal_timing(rows,7)[0]
        self.assertGreaterEqual(result['median_speed_ratio'],1.2)
        self.assertLessEqual(result['median_speed_ratio_ci95'][0],1.)
        self.assertFalse(result['ci_lower_bound_above_one'])

    def test_unrelated_timeout_does_not_hide_complete_recipe(self):
        stage=dict(jobs=[dict(case='case',algorithm=a,candidate=c,repeat=0)
                         for a,c in [('NLM','default'),('NLM','candidate'),('TWSC','diagnostic')]])
        checked=dict(coverage={'missing':[]},integrity_pass=True,
            terminal_failures=[dict(case='case',algorithm='TWSC',candidate='diagnostic',repeat=0,status='timeout')])
        self.assertTrue(recipe_coverage(stage,checked,'NLM','candidate')['all_successful'])
        self.assertFalse(recipe_coverage(stage,checked,'TWSC','diagnostic')['all_successful'])

    def test_config_and_input_identity_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fixtures256').mkdir();directory=root/'run';directory.mkdir()
            config=root/'recipe.json';config.write_text(json.dumps({'algorithms':{'NLM':[{'id':'candidate','parameters':{}}]}}))
            (root/'fixtures256/nlh-v4.json').write_text('{}')
            case=dict(id='case',sigma=25,channels=1,clean_sha256='clean',noisy_sha256='expected-noise')
            template=root/'fixtures256/fixtures.json';template.write_text(json.dumps({'cases':[case]}))
            manifest=root/'materialized.json';manifest.write_text(json.dumps({'cases':[case],'parent_full_fixture_manifest_sha256':sha(template)}))
            stage=dict(name='test',phase='sealed',temporal=False,size=256,split='sealed',config='recipe.json',
                config_sha256=sha(config),cases=[case],jobs=[dict(case='case',algorithm='NLM',candidate='candidate',repeat=0)],
                algorithms=['NLM'],repeats=1,timeout=180,twsc_timeout=60,target_seconds=.25,finalists=None,
                fixture_manifest='materialized.json',source_template_sha256=sha(template))
            identity=dict(plugin_sha256='plugin',harness_sha256='worker',candidates_sha256=sha(config),
                metrics_sha256=METRICS_SHA,defaults_helper_sha256=DEFAULTS_SHA,split='sealed',cases=['case'],algorithms=['NLM'],
                repeats=1,timeout=180,twsc_timeout=60,target_seconds=.25,max_frames=4096,candidate_ids=None,finalists_sha256=None,
                nlh_v4_sha256=sha(root/'fixtures256/nlh-v4.json'),fixtures_sha256=sha(manifest))
            (directory/'identity.json').write_text(json.dumps(identity))
            row=timed();row.update(plugin_sha256='plugin',harness_sha256='worker',supplied_parameters={},sigma=25,
                split='sealed',pipeline='single',shape=[1,256,256],clean_sha256='clean',timing_eligible=True,
                timed_source_fills=0,output_cache_disabled=True,output='not-downloaded.f32',output_sha256='pixels')
            (directory/'results.jsonl').write_text(json.dumps(row)+'\n')
            selection=dict(plugin_sha256='plugin',worker_sha256='worker')
            result=verify_stage(stage,directory,root,selection,root,root)
            self.assertFalse(result['integrity_pass'])
            self.assertIn('record noisy input differs',result['errors'])
            row['noisy_sha256']='expected-noise'
            (directory/'results.jsonl').write_text(json.dumps(row)+'\n')
            result=verify_stage(stage,directory,root,selection,root,root)
            self.assertTrue(result['integrity_pass'],result['errors'])
            config.write_text(config.read_text()+'\n')
            result=verify_stage(stage,directory,root,selection,root,root)
            self.assertIn('frozen config differs',result['errors'])


if __name__=='__main__':unittest.main()
