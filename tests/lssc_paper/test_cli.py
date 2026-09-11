import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from audit import verify
from run import sha


def run_command(tmp_path, clean_value, name):
    image = .4 + np.random.default_rng(13).normal(0,.05,(11,12))
    np.save(tmp_path/'input.npy',image.astype(np.float32))
    np.save(tmp_path/'D.npy',np.eye(9))
    np.save(tmp_path/(name+'-clean.npy'),np.full_like(image,clean_value))
    command = [sys.executable,str(Path(__file__).with_name('run.py')),
               '--input',str(tmp_path/'input.npy'),'--dictionary',str(tmp_path/'D.npy'),
               '--sigma','12.75','--window','5','--clean',str(tmp_path/(name+'-clean.npy')),
               '--out',str(tmp_path/name)]
    environment = dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
    result = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=30)
    return result, command


def test_cli_ground_truth_cannot_change_output_and_saved_trace_recomputes(tmp_path):
    first,_ = run_command(tmp_path,0,'first')
    second,_ = run_command(tmp_path,1,'second')
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert sha(tmp_path/'first/output.f32') == sha(tmp_path/'second/output.f32')
    assert sha(tmp_path/'first/pilot.f32') == sha(tmp_path/'second/pilot.f32')
    assert verify(tmp_path/'first',tmp_path/'input.npy',tmp_path/'D.npy')['verified']
    meta = json.loads((tmp_path/'first/result.json').read_text())
    assert meta['complete_lssc'] is False and meta['dictionary_learning'] is False
    assert len(meta['unverified_choices']) >= 5


def test_existing_output_is_not_overwritten(tmp_path):
    first,command = run_command(tmp_path,.4,'first')
    assert first.returncode == 0, first.stderr
    before = sha(tmp_path/'first/output.f32')
    again = subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert again.returncode != 0
    assert sha(tmp_path/'first/output.f32') == before


def test_independent_audit_detects_incorrect_fit_even_if_file_hash_is_updated(tmp_path):
    execution,_ = run_command(tmp_path,.4,'first')
    assert execution.returncode == 0, execution.stderr
    trace = tmp_path/'first/trace.npz'
    with np.load(trace,allow_pickle=False) as data:
        arrays = dict(data)
    arrays['final_patches'][0,0] += .1
    np.savez_compressed(trace,**arrays)
    path = tmp_path/'first/result.json'
    metadata = json.loads(path.read_text())
    metadata['trace_sha256'] = sha(trace)
    path.write_text(json.dumps(metadata))
    with pytest.raises(AssertionError):
        verify(tmp_path/'first',tmp_path/'input.npy',tmp_path/'D.npy')
