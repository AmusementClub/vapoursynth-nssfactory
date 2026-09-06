import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bm_numerics import compare
from avx2_review_evidence import review_phase

class ReviewEvidenceTests(unittest.TestCase):
    def test_replay_must_match_retained_arrays_and_keep_raw_record(self):
        with tempfile.TemporaryDirectory() as directory:
            phase=Path(directory);bench=phase/'bench';bench.mkdir()
            baseline=np.zeros((1,1,8,8),np.float32)
            candidate=baseline+.001
            for side,value in (('baseline',baseline),('candidate',candidate)):
                np.save(bench/f'0-{side}.npy',value)
            numeric=compare(baseline,candidate)
            rows=[dict(config=dict(name='case'),numerical=numeric,paired_speedup=1.1,
                       pairs=3,ci95=[1.09,1.11],environment=dict(valid=True),selected=False)]
            summary=bench/'summary.json';summary.write_text(json.dumps(rows));original=summary.read_bytes()
            self.assertEqual(review_phase(phase)['bench']['selected'],[])
            replay=phase/'crossover';replay.mkdir()
            report=dict(passed=True,original=numeric,reproduced_original=dict(baseline=True,candidate=True),
                        loss=dict(psnr=.001,ssim=.00001),output_hashes=dict(baseline='wrong',candidate='wrong'))
            p=replay/'replay.json';p.write_text(json.dumps(report))
            self.assertEqual(review_phase(phase)['bench']['selected'],[])
            report['output_hashes']={s:hashlib.sha256(v.tobytes()).hexdigest()
                                    for s,v in (('baseline',baseline),('candidate',candidate))}
            p.write_text(json.dumps(report))
            self.assertEqual(review_phase(phase)['bench']['selected'],['case'])
            self.assertEqual(summary.read_bytes(),original)
            (replay/'INVALID.txt').write_text('wrong replay parameters')
            self.assertEqual(review_phase(phase)['bench']['selected'],[])

    def test_speed_boundary_and_environment_remain_required(self):
        with tempfile.TemporaryDirectory() as directory:
            phase=Path(directory);bench=phase/'bench';bench.mkdir()
            rows=[dict(config=dict(name='boundary'),numerical=dict(passed=True),paired_speedup=1.02,
                       pairs=7,ci95=[1.01,1.03],environment=dict(valid=True)),
                  dict(config=dict(name='busy'),numerical=dict(passed=True),paired_speedup=1.5,
                       pairs=7,ci95=[1.4,1.6],environment=dict(valid=False))]
            (bench/'summary.json').write_text(json.dumps(rows))
            self.assertEqual(review_phase(phase)['bench']['selected'],[])

if __name__=='__main__':unittest.main()
