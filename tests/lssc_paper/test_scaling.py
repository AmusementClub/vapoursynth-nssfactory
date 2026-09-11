import numpy as np
import pytest
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
from scipy.io import savemat

import p2_reference
import reference
from benchmark import digest
from scaling import fixture, stage_timers, statistics_for


@pytest.mark.parametrize('variant', ['p1', 'p2'])
def test_warm_instrumentation_exact_and_restored(variant):
    image = .4 + np.random.default_rng(5).normal(0, .02, (12, 13))
    dictionary = np.eye(9)
    original = (reference.somp, p2_reference.simultaneous_ols, reference.aggregate)
    call = (lambda: reference.denoise(image, dictionary, .02)) if variant == 'p1' else (
        lambda: p2_reference.denoise_diagnostic(image, dictionary, .02,
                                               solver='energy_gain', grouping='overlap'))
    expected = call()
    with stage_timers() as (costs, counts):
        actual = call()
    assert digest(actual.output) == digest(expected.output)
    assert (reference.somp, p2_reference.simultaneous_ols, reference.aggregate) == original
    assert counts['aggregate'] == 2
    assert counts['pilot_solve'] == 110
    assert costs['pilot_solve'] > 0 and costs['group_solve'] > 0
    report = statistics_for(actual)
    assert report['patches'] == 110 and report['all_final_groups_met_budget']


def test_timer_restoration_on_exception():
    original = reference.somp
    with pytest.raises(RuntimeError):
        with stage_timers():
            raise RuntimeError('test')
    assert reference.somp is original


def test_stress_fixture_labels_repetition_only_in_clean():
    native = np.arange(12).reshape(3, 4) / 12
    clean, noisy = fixture(native, 8, 6, 25, 12)
    assert clean.shape == noisy.shape == (6, 8)
    np.testing.assert_array_equal(clean[:3, :4], clean[3:, 4:])
    assert not np.array_equal(noisy[:3, :4], noisy[3:, 4:])
    np.testing.assert_array_equal(noisy, fixture(native, 8, 6, 25, 12)[1])
    with pytest.raises(ValueError):
        fixture(native, 0, 6, 25, 12)


def test_serial_worker_protocol_and_failure_evidence(tmp_path):
    clean = tmp_path / 'source.png'
    Image.fromarray(np.full((12, 12), 100, dtype=np.uint8)).save(clean)
    dictionary = tmp_path / 'dictionary.mat'
    savemat(dictionary, {'D': np.eye(9)})
    out = tmp_path / 'campaign'
    command = [sys.executable, str(Path(__file__).with_name('scaling.py')), '--clean', str(clean),
               '--dictionary', str(dictionary), '--cases', '12x12:25', '--pairs', '2', '--out', str(out)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    result = json.loads((out / '12x12-s25' / 'result.json').read_text())
    assert result['complete'] and not result['formal_c4_gate']
    assert [r['variant'] for r in result['rows']] == ['p1', 'p2', 'p2', 'p1']
    assert all(r['output_sha256'] == result['warm'][r['variant']]['output_sha256']
               for r in result['rows'])
    assert all(r['maxrss_bytes'] > 0 for r in result['rows'])
    # Invalid dictionary fails in the worker; parent preserves the failure and
    # closes child processes instead of hanging or claiming a completed sample.
    savemat(dictionary, {'D': np.ones((9, 9))})
    failed = tmp_path / 'failed'
    command[-1] = str(failed)
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    failure = json.loads((failed / '12x12-s25' / 'result.json').read_text())
    assert not failure['complete'] and failure['failure']['type'] == 'RuntimeError'
